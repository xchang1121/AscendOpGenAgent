#!/usr/bin/env python3
"""
Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python scripts/engine/pipeline.py <task_dir>

This script does:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. record_round → KEEP/DISCARD/FAIL (workflow library, in-process)
    4. settle → update plan.md, advance (ACTIVE)
    5. compute next phase → write .phase
    6. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from quick_check import check_editable_files, _run_smoke_test as _run_smoke
from task_config import load_task_config, run_eval
from utils.failure_extractor import extract_failure_signals, format_for_stdout
from utils.json_io import sanitize_floats
from workflow import PhaseController, PlanStore, record_round
from phase_machine import (
    get_active_item,
    get_guidance, auto_rollback, load_progress, edit_marker_path,
    pending_settle_path, FINISH,
)


def _run_settle(task_dir: str, kd_json: dict) -> tuple:
    """Settle the active plan item in-process. Returns
    ``(ok: bool, error_tail: str, settle_json: dict | None)``.

    Used to subprocess `engine/settle.py` and parse its stdout; now
    inlined here (`engine/settle.py` was deleted). For manual recovery
    after a failed settle, the operator re-runs pipeline.py — the
    `.pending_settle.json` sentinel makes the re-run skip quick_check
    / eval / record_round and replay this function only. The `settle_
    json` returned here carries the `settled_item` id, which the caller
    needs for the status report — `get_active_item()` AFTER settle
    points at the NEXT ACTIVE item, not the one we just settled.
    """
    try:
        decision = kd_json.get("decision", "FAIL")
        best_metric = kd_json.get("best_metric")
        # KEEP carries this round's metric; DISCARD/FAIL leave it None.
        metric_val = best_metric if decision == "KEEP" else None

        store = PlanStore(task_dir)
        if not store.exists():
            return False, "plan.md not found", None
        settled_id, _ = store.settle_active(decision, metric_val)
        return True, "", {
            "settled_item": settled_id,
            "decision": decision,
            "metric": metric_val,
        }
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _persist_pending_settle(task_dir: str, kd_json: dict) -> None:
    path = pending_settle_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(kd_json), f)


def _clear_pending_settle(task_dir: str) -> None:
    path = pending_settle_path(task_dir)
    if os.path.exists(path):
        os.remove(path)


def _emit_settle_failure(task_dir: str, error_tail: str) -> None:
    print(f"[PIPELINE] SETTLE FAILED. plan.md was NOT updated. "
          f"progress.json + history.jsonl already moved during this round; "
          f"re-running this script will RETRY SETTLE ONLY (kd_json was "
          f"persisted to .ar_state/.pending_settle.json) — it will NOT "
          f"re-run quick_check/eval/record_round.\n"
          f"\n"
          f"Recovery options (do NOT hand-edit plan.md):\n"
          f"  1. Fix the underlying cause from the error tail below, "
          f"then re-run pipeline.py — the replay-only path will retry "
          f"settle on the same kd_json.\n"
          f"  2. If the failure is structural (plan.md malformed, no "
          f"(ACTIVE) item, etc.) and settle cannot recover, run "
          f"create_plan.py to write a fresh plan.md. While "
          f"pending_settle.json exists, hooks/guard_bash allows "
          f"create_plan.py in EDIT phase as a recovery path; on "
          f"successful create_plan validation hooks/post_bash clears "
          f"pending_settle.json. The orphan history.jsonl row stays "
          f"(audit trail) but no longer corresponds to any plan item.\n"
          f"\n"
          f"error: {error_tail}", file=sys.stderr)


def _post_settle(task_dir: str, decision: str, settled_id: str) -> None:
    """Common path after a successful settle: advance phase, clear edit
    marker, print status. Runs whether settle succeeded the first time or
    on the replay-only retry."""
    next_phase = PhaseController(task_dir).on_round_settled()
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        os.remove(marker)

    # FINISH is a one-way terminal transition — generate the deterministic
    # report.md (summary tables + inline SVG curve) here so it's on disk
    # before the FINISH guidance announces its path.
    if next_phase == FINISH:
        try:
            from report import write_report
            rp = write_report(task_dir)
            if rp:
                print(f"[PIPELINE] Report written: "
                      f"{os.path.relpath(rp, task_dir)}")
        except Exception as e:
            print(f"[PIPELINE] Report generation failed: {e}",
                  file=sys.stderr)

    progress = load_progress(task_dir) or {}
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)

    improv = ""
    if (
        best is not None and baseline is not None
        and isinstance(best, (int, float))
        and isinstance(baseline, (int, float))
        and baseline != 0 and best != 0
    ):
        pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best}{improv} | Failures: {failures}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
    print(get_guidance(task_dir))


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <task_dir>")
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])

    # === Replay-only settle ===
    # If a previous pipeline.py invocation got past record_round but
    # settle.py failed, the kd_json was persisted to .pending_settle.json.
    # Re-running pipeline.py from scratch would re-eval and double-write
    # progress/history; instead, we ONLY retry settle here. Fix the
    # underlying cause (the agent saw the failure reason in stderr), then
    # invoke pipeline.py — same command, no flags — and this branch handles
    # the retry deterministically. Lives BEFORE task.yaml load so retry
    # works even if task config has drifted (settle only touches .ar_state).
    pending_path = pending_settle_path(task_dir)
    if os.path.exists(pending_path):
        try:
            with open(pending_path, "r", encoding="utf-8") as f:
                kd_json = json.load(f)
        except Exception as e:
            print(f"[PIPELINE] pending settle file unreadable ({e}). "
                  f"Removing it and bailing — please re-run pipeline.py "
                  f"to start a fresh round.", file=sys.stderr)
            _clear_pending_settle(task_dir)
            sys.exit(1)
        print(f"[PIPELINE] Retrying settle from {os.path.basename(pending_path)} "
              f"(skipping quick_check/eval/record_round).", flush=True)
        ok, error_tail, settle_json = _run_settle(task_dir, kd_json)
        if not ok:
            _emit_settle_failure(task_dir, error_tail)
            sys.exit(1)
        _clear_pending_settle(task_dir)
        # Use _run_settle's reported settled_item, not get_active_item — by
        # this point ACTIVE has already advanced to the NEXT pending item.
        settled_id = (settle_json or {}).get("settled_item") or "?"
        _post_settle(task_dir, kd_json.get("decision", "?"), settled_id)
        return

    config = load_task_config(task_dir)
    if config is None:
        print("[PIPELINE] ERROR: task.yaml not found")
        sys.exit(1)

    progress = load_progress(task_dir) or {}
    active = get_active_item(task_dir)
    # Persist the full description — dashboards/logs do their own display-time
    # truncation based on terminal width.
    desc = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None

    # === Step 1: Quick check ===
    # In-process: check_editable_files + smoke test. quick_check.py
    # stays as a standalone CLI for manual replay; here we call the
    # same helpers directly, no subprocess / rc-decoding / stdout-parse.
    print("[PIPELINE] Running quick_check...", flush=True)
    try:
        file_issues = check_editable_files(task_dir, config)
        smoke_errors = _run_smoke(task_dir, config)
    except Exception as exc:
        file_issues = [{"file": "(internal)",
                        "report": f"quick_check crashed: "
                                  f"{type(exc).__name__}: {exc}",
                        "errors": []}]
        smoke_errors = []

    if file_issues or smoke_errors:
        auto_rollback(task_dir)
        # Clear edit marker — rollback means we're back to clean state
        marker = edit_marker_path(task_dir)
        if os.path.exists(marker):
            os.remove(marker)
        blob: dict = {"ok": False}
        if file_issues:
            blob["file_issues"] = file_issues
        if smoke_errors:
            blob["smoke_errors"] = smoke_errors
        print(f"[PIPELINE] QUICK CHECK FAIL: "
              f"{json.dumps(blob, ensure_ascii=False)[:200]}")
        print(f"[PIPELINE] Auto-rolled back. Fix and re-edit.")
        print(get_guidance(task_dir))
        sys.exit(0)

    print("[PIPELINE] Quick check PASS", flush=True)

    # === Step 2: Eval ===
    # Direct in-process call (was a subprocess to eval_wrapper.py + last-line-JSON
    # parse). eval_client owns its own per-shape timeout via task.yaml's
    # `eval_timeout`; no outer wall-clock cap is needed here because the
    # subprocess crash isolation lives inside utils.eval_runner.local_eval
    # (the two `eval_kernel.py` subprocesses it spawns).
    print("[PIPELINE] Running eval...", flush=True)
    try:
        result = run_eval(task_dir, config)
    except Exception as e:
        auto_rollback(task_dir)
        print(f"[PIPELINE] EVAL ERROR: run_eval raised "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    eval_json = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        eval_json["failure_signals"] = extract_failure_signals(
            result.raw_output).to_dict()
        eval_json["raw_output_tail"] = (result.raw_output or "")[-4000:]

    correctness = eval_json.get("correctness", False)
    metrics = eval_json.get("metrics", {})
    print(f"[PIPELINE] Eval: correctness={correctness}, metrics={metrics}", flush=True)

    # infra_fail: eval pipeline broke before kernel was meaningfully
    # exercised. Roll back and skip the round — recording a FAIL here
    # would mislead later DIAGNOSE / KEEP / DISCARD.
    if eval_json.get("outcome") == "infra_fail":
        auto_rollback(task_dir)
        print(f"[PIPELINE] INFRA_FAIL: {eval_json.get('error', 'no data')}. "
              f"Rolled back, not recording round.", flush=True)
        sys.exit(0)

    # Surface structured failure signals (UB overflow, aivec trap, OOM, ...)
    # extracted from the eval subprocess's raw log. Without this, Claude
    # sees only a generic "verify failed" string and has nothing to act
    # on. Fall back
    # through increasingly coarse sources so *something* always reaches the
    # user on failure.
    if not correctness or eval_json.get("error"):
        if eval_json.get("error"):
            print(f"[PIPELINE] Error: {eval_json['error']}", flush=True)
        pretty = format_for_stdout(eval_json.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_json.get("raw_output_tail"):
            # No known pattern matched — dump the tail raw so Claude still
            # has something concrete to work with.
            print("[PIPELINE] Eval log tail (no structured signals matched):",
                  flush=True)
            print(eval_json["raw_output_tail"], flush=True)

    # === Step 3: Keep or discard ===
    # Direct library call (was a subprocess + last-line-JSON parse before
    # the deleted keep_or_discard.py shell wrapper was removed). In-process
    # eliminates the JSON-protocol seam between pipeline and the round
    # writer.
    kd_json = record_round(task_dir, eval_json,
                           description=desc, plan_item=plan_item)
    if kd_json.get("decision") == "ERROR":
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd_json.get('error')}")
        sys.exit(1)

    decision = kd_json.get("decision", "FAIL")

    # === Step 4: Settle (update plan.md) ===
    # progress.json + history.jsonl were already mutated by record_round;
    # plan.md is the only state piece settle.py owns. If settle fails, the
    # kd_json is persisted to .pending_settle.json so the NEXT invocation
    # of pipeline.py retries settle alone (no second eval, no duplicate
    # history row). The advance-phase block is gated on settle success.
    ok, error_tail, _settle_json = _run_settle(task_dir, kd_json)
    if not ok:
        _persist_pending_settle(task_dir, kd_json)
        _emit_settle_failure(task_dir, error_tail)
        sys.exit(1)
    _clear_pending_settle(task_dir)

    # === Step 5+6: Advance phase + status report ===
    settled_id = active["id"] if active else "?"
    _post_settle(task_dir, decision, settled_id)


if __name__ == "__main__":
    main()
