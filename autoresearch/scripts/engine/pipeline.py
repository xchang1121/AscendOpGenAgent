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
    get_guidance, auto_rollback, load_progress, load_state, save_state,
    edit_marker_path, FINISH,
    require_state_consistency, format_state_inconsistency,
    replay_intent,
)


def _run_settle(task_dir: str, kd_json: dict) -> tuple:
    """Settle the active plan item in-process. Returns
    ``(ok: bool, error_tail: str, settle_json: dict | None)``.

    Idempotent w.r.t. plan_item: when kd_json's `plan_item` already
    appears in plan.md's Settled History table, settle is considered
    already-done (replay-safe). Otherwise it actually runs.
    """
    try:
        decision = kd_json.get("decision", "FAIL")
        best_metric = kd_json.get("best_metric")
        # KEEP carries this round's metric; DISCARD/FAIL leave it None.
        metric_val = best_metric if decision == "KEEP" else None

        store = PlanStore(task_dir)
        if not store.exists():
            return False, "plan.md not found", None

        expected_item = kd_json.get("plan_item")
        if expected_item:
            current_active = get_active_item(task_dir)
            current_id = (current_active or {}).get("id")
            if current_id != expected_item:
                # Plan moved past expected_item. Only safe when
                # expected_item is in Settled History (legitimate
                # replay of a prior successful settle); otherwise
                # plan.md is malformed / rewritten and we must NOT
                # clear the sentinel.
                settled_rows = store.parse_settled_history() or ""
                if f"| {expected_item} |" in settled_rows:
                    return True, "", {
                        "settled_item": expected_item,
                        "decision": decision,
                        "metric": metric_val,
                        "already_settled": True,
                    }
                return False, (
                    f"plan.md ACTIVE is {current_id!r}, kd_json "
                    f"expected {expected_item!r}, and {expected_item} "
                    f"does NOT appear in Settled History. Plan is "
                    f"either malformed or was rewritten by an "
                    f"unrelated create_plan — pretending settle "
                    f"succeeded would lose this round's kd_json. "
                    f"Refusing; state.pending_settle retained for "
                    f"manual inspection."
                ), None

        settled_id, _ = store.settle_active(decision, metric_val)
        return True, "", {
            "settled_item": settled_id,
            "decision": decision,
            "metric": metric_val,
        }
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _clear_pending_settle(task_dir: str) -> None:
    """Atomic state-write that nulls the pending_settle field. The
    sentinel used to be a separate .pending_settle.json file; it now
    lives in state.pending_settle so the clear is one save_state."""
    state = load_state(task_dir)
    if state is None or state.get("pending_settle") is None:
        return
    state["pending_settle"] = None
    save_state(task_dir, state)


def _emit_settle_failure(task_dir: str, error_tail: str) -> None:
    print(f"[PIPELINE] SETTLE FAILED. plan.md was NOT updated. "
          f"history.jsonl + state.json already moved during this "
          f"round; re-running this script will RETRY SETTLE ONLY "
          f"(kd_json was persisted to state.pending_settle) — it "
          f"will NOT re-run quick_check/eval/record_round.\n"
          f"\n"
          f"Recovery options (do NOT hand-edit plan.md):\n"
          f"  1. Fix the underlying cause from the error tail below, "
          f"then re-run pipeline.py — the replay-only path will "
          f"retry settle on the same kd_json.\n"
          f"  2. If the failure is structural (plan.md malformed, "
          f"no (ACTIVE) item, etc.) and settle cannot recover, run "
          f"create_plan.py to write a fresh plan.md. While "
          f"state.pending_settle is non-null, hooks/guard_bash "
          f"allows create_plan.py in EDIT phase as a recovery path; "
          f"on successful create_plan validation hooks/post_bash "
          f"clears state.pending_settle. The orphan history.jsonl "
          f"row stays (audit trail).\n"
          f"\n"
          f"error: {error_tail}", file=sys.stderr)


def _post_settle(task_dir: str, decision: str, settled_id: str) -> None:
    """Common path after a successful settle: advance phase, clear
    edit marker, print status. Runs whether settle succeeded the
    first time or on the replay-only retry."""
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

    # === Journal replay ===
    # A prior writer (record_round / baseline) journals its intent
    # before touching bodies, then clears the journal after state.json
    # commits. A crash in the window leaves intent.json behind;
    # replay_intent inspects it and the actual artifacts to either
    # rebuild state (bodies landed, state didn't), discard (bodies
    # never landed), or clear (state already caught up). After this
    # returns, the consistency gate below is meaningful: any remaining
    # inconsistency is a genuine off-flow corruption, not a normal
    # crash window we know how to heal.
    replayed = replay_intent(task_dir)
    if replayed is not None:
        print(f"[PIPELINE] intent.json {replayed['action']}: "
              f"{replayed['detail']}", file=sys.stderr)

    # === Cross-file consistency gate ===
    # state.json is the commit barrier; plan.md + history.jsonl are
    # durable bodies written ahead of it. With the journal in place,
    # any inconsistency that reaches here is off-flow (manual file
    # edits, external rewrites) and the operator must fix the
    # specific artifact named in the report — re-running the writer
    # is no longer a generic recovery.
    report = require_state_consistency(task_dir, on_inconsistent="report")
    if not report["consistent"]:
        print(f"[PIPELINE] REFUSING TO RUN — "
              f"{format_state_inconsistency(report)}", file=sys.stderr)
        sys.exit(1)

    # === Replay-only settle ===
    # If a previous pipeline.py invocation got past record_round but
    # settle failed (or was killed before _post_settle / _clear_pending_
    # settle ran), record_round persisted the kd_json into
    # state.pending_settle. Re-running pipeline.py from scratch would
    # re-eval and double-write history; instead, we ONLY retry settle.
    # Lives BEFORE task.yaml load so retry works even if task config
    # has drifted (settle only touches .ar_state).
    pending_state = load_state(task_dir)
    kd_json = (pending_state or {}).get("pending_settle")
    if kd_json:
        print(f"[PIPELINE] Retrying settle from state.pending_settle "
              f"(skipping quick_check/eval/record_round).", flush=True)
        ok, error_tail, settle_json = _run_settle(task_dir, kd_json)
        if not ok:
            _emit_settle_failure(task_dir, error_tail)
            sys.exit(1)
        # Order: settle wrote plan.md → _post_settle writes phase →
        # _clear_pending_settle removes the sentinel LAST. A crash
        # before the clear leaves state.pending_settle non-null and
        # the next pipeline.py invocation re-enters this branch
        # idempotently.
        settled_id = (settle_json or {}).get("settled_item") or "?"
        _post_settle(task_dir, kd_json.get("decision", "?"), settled_id)
        _clear_pending_settle(task_dir)
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
    # record_round writes history.jsonl + state.json (incl. pending_
    # settle = kd_json) in one atomic save_state. Returns the kd_json
    # we then hand to _run_settle.
    kd_json = record_round(task_dir, eval_json,
                           description=desc, plan_item=plan_item)
    if kd_json.get("decision") == "ERROR":
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd_json.get('error')}")
        sys.exit(1)

    decision = kd_json.get("decision", "FAIL")

    # === Step 4: Settle (update plan.md) ===
    # state.json + history.jsonl already moved during record_round and
    # state.pending_settle holds the sentinel. A crash between here
    # and _clear_pending_settle is recoverable via the replay branch
    # at top-of-main.
    ok, error_tail, _settle_json = _run_settle(task_dir, kd_json)
    if not ok:
        _emit_settle_failure(task_dir, error_tail)
        sys.exit(1)

    # === Step 5+6: Advance phase + clear sentinel ===
    # Order: plan.md is already written (_run_settle), .phase lands
    # next (_post_settle), sentinel clear LAST. Any crash before the
    # clear leaves state.pending_settle non-null; pipeline.py's
    # replay branch re-enters _run_settle (sees expected_item in
    # Settled History → already_settled), re-runs _post_settle
    # (idempotent on same phase), clears.
    settled_id = active["id"] if active else "?"
    _post_settle(task_dir, decision, settled_id)
    _clear_pending_settle(task_dir)


if __name__ == "__main__":
    main()
