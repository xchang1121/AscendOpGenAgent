#!/usr/bin/env python3
"""
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

The only commands that advance phase from this hook are those Claude runs
directly via the Bash tool:
  - `export AR_TASK_DIR=...`  → activate task, compute starting phase
                                (fresh task always lands at BASELINE; both
                                reference.py and kernel.py are guaranteed
                                present because /autoresearch requires both)
  - `baseline.py`             → PLAN on success or on kernel_fail;
                                infra_fail leaves phase untouched
  - `pipeline.py`             → whatever phase pipeline.py itself wrote
  - `create_plan.py`          → EDIT on plan validation pass
                                (called from PLAN / DIAGNOSE / REPLAN)

The inner pipeline steps (quick_check subprocess + in-process run_eval +
in-process record_round + settle subprocess) run beneath pipeline.py and
never re-enter this hook, so they don't need their own phase constants
or branches here.
"""
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status, emit_todowrite_context
from workflow import PhaseController
from phase_machine import (
    read_phase, get_guidance, compute_resume_phase,
    get_task_dir, set_task_dir, get_active_item, touch_heartbeat,
    load_progress, update_progress,
    parse_invoked_ar_script,
    history_path, plan_path, edit_marker_path, state_record_path,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN,
)


def _activation_target(command: str) -> str | None:
    r"""Extract the path from `export AR_TASK_DIR=<path>`. Uses shlex
    so quoted values with spaces (`AR_TASK_DIR="/path with space"`)
    survive — the earlier `[^"\';\s&]+` regex truncated at the first
    space."""
    if "AR_TASK_DIR=" not in command:
        return None
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    for tok in tokens:
        if tok.startswith("AR_TASK_DIR="):
            return tok[len("AR_TASK_DIR="):] or None
    return None


# Script-invocation parsing lives in phase_machine.parse_invoked_ar_script,
# a thin view over `classify(command)` — returns the AR script basename
# only when the classifier sees a canonical AR shape (and None otherwise,
# including for non-canonical AR-mentions which PreToolUse already
# rejected). Under that contract the basename returned here is
# unambiguous, and shapes like `python --version ...X.py` or
# `python -c ... .../X.py` no longer falsely advance phase.


def _clean_stale_edit_marker(task_dir: str):
    """Remove .edit_started if git is clean (nothing to resume)."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    from utils.git_utils import is_working_tree_clean
    if is_working_tree_clean(task_dir):
        try:
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
        except OSError:
            pass


def _handle_activation(new_task_dir: str):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    # set_task_dir returns False when refusing to overwrite a
    # heartbeat-fresh pointer (= another live Claude/batch session
    # owns the active task). Without checking the return value, the
    # transcript here would claim B is now activated while .active_task
    # still points at A — guard_bash then evaluates phase against A
    # while B's commands name B's task_dir, and the two sessions
    # cross-write each other's state. Bail loudly.
    if not set_task_dir(new_task_dir):
        emit_status(
            f"[AR] ERROR: refused to activate {new_task_dir} — another "
            f"session is active on a different task (heartbeat fresh). "
            f"Stop the other session, or rm autoresearch/.active_task to "
            f"force takeover."
        )
        return
    _clean_stale_edit_marker(new_task_dir)

    # state.json is the single source of truth — its existence is
    # the new "has progress / has phase" signal.
    has_state = os.path.exists(state_record_path(new_task_dir))

    pc = PhaseController(new_task_dir)
    if has_state:
        # Cross-file consistency gate. plan.md / history.jsonl ahead
        # of state.expected_* means a writer landed body bytes but
        # never committed state.json — every downstream computation
        # would be based on inconsistent state. Refuse to advance,
        # hand the operator the recovery message, exit.
        from phase_machine import (
            require_state_consistency as _require,
            format_state_inconsistency as _fmt_inconsistency,
        )
        try:
            rep = _require(new_task_dir, on_inconsistent="report")
        except Exception:
            rep = None
        if rep is not None and not rep["consistent"]:
            emit_status("[AR] " + _fmt_inconsistency(rep))
            return
        phase = read_phase(new_task_dir)
        # Stale-planning recovery: phase file says PLAN or REPLAN but
        # plan.md + progress.json show a validated plan with an active
        # item. This is the state left by a create_plan.py that finished
        # both disk writes but crashed before PostToolUse advanced
        # .phase to EDIT. Without recovery, the agent re-runs
        # create_plan, bumps to vN+1, and loses the pending items of vN.
        #
        # DIAGNOSE is deliberately NOT in this list: it has its own
        # gate (diagnose_state.action requires the subagent's
        # diagnose_v<N>.md artifact). compute_resume_phase doesn't
        # model that gate — it would happily return EDIT for a DIAGNOSE
        # task whose plan still carries the pre-DIAGNOSE active item,
        # skipping the diagnosis the agent was about to do. Leave
        # DIAGNOSE to the normal PostToolUse(Task) flow; if the
        # operator's session died mid-DIAGNOSE the agent just continues
        # the artifact loop on resume.
        if phase in (PLAN, REPLAN):
            recomputed = pc.on_activation_resume()
            if recomputed != phase:
                emit_status(
                    f"[AR] Phase file was {phase} but plan.md + "
                    f"progress.json show round-ready state — "
                    f"advancing to {recomputed} (create_plan.py "
                    f"likely crashed before PostToolUse could advance).")
                phase = recomputed
        emit_status(f"[AR] Resuming. Phase: {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    else:
        # No state.json yet → fresh-scaffolded task. Initial phase
        # transition lands inside _fresh_start (which calls
        # PhaseController.on_activation_ready).
        _fresh_start(new_task_dir)


def _fresh_start(task_dir: str):
    """Pick initial phase for a fresh task. With /autoresearch requiring
    both --ref and --kernel, scaffold has already gated on reference
    runnability and written the user's seed kernel; the next legal step
    is always BASELINE. baseline.py exercises the kernel; on failure the
    hook routes to PLAN so the agent rewrites via plan->edit."""
    PhaseController(task_dir).on_activation_ready()
    emit_status(f"[AR] Fresh start. Phase -> BASELINE. {get_guidance(task_dir)}")


def _baseline_message(outcome, new_phase, progress, guidance):
    if outcome == "infra_fail":
        if getattr(progress, "baseline_error_source", None) == "ref":
            return ("[AR] Baseline INFRA_FAIL (ref): reference.py is broken. "
                    "Fix the source file passed via --ref and re-run "
                    "/autoresearch from scratch — reference.py is not "
                    "editable from EDIT. Phase stays at BASELINE.")
        return ("[AR] Baseline INFRA_FAIL: eval pipeline broken (no per-shape "
                "data). Do NOT edit kernel.py. Fix env / device / eval.timeout "
                "and re-run baseline.py.")
    if outcome != "ok":
        reason = ("seed kernel produced no timing"
                  if progress.seed_metric is None
                  else "seed kernel failed correctness / profile")
        return (f"[AR] Baseline failed: {reason}. Phase -> PLAN. Plan a "
                f"kernel fix/rewrite via the standard plan->edit loop. "
                f"{guidance}")
    return f"[AR] Baseline complete. Phase -> PLAN. {guidance}"


def _reset_failures_for_diagnose(task_dir: str, phase: str):
    """Zero consecutive_failures only on DIAGNOSE replan validation
    (PLAN/REPLAN keep the streak — failures led to the replan)."""
    if phase == DIAGNOSE:
        update_progress(task_dir, consecutive_failures=0)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    # Activation arrives as its own Bash call under the canonical-form
    # gate (any chain is rejected at PreToolUse), so we can return as
    # soon as `_handle_activation` has set up the task pointer + emitted
    # guidance — there is no AR-script invocation in the same command
    # to dispatch on.
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py" and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            emit_status("[AR] Baseline failed (no progress.json). Retry.")
        else:
            # baseline.py / workflow.run_baseline_init already advanced the
            # phase via PhaseController.on_baseline_settled before exiting.
            # Read it back here — do NOT re-run the transition (used to be
            # called from both sides, masked by the both-paths-land-on-PLAN
            # accident for the success case).
            new_phase = read_phase(task_dir)
            outcome = getattr(progress, "baseline_outcome", None)
            emit_status(_baseline_message(outcome, new_phase, progress,
                                          get_guidance(task_dir)))

    elif invoked == "pipeline.py":
        # pipeline.py writes .phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN, EDIT):
        from phase_machine import (
            validate_plan, load_state, save_state,
            check_state_consistency,
        )
        # PLAN/DIAGNOSE/REPLAN: normal plan-creation flow.
        # EDIT: only legal as a recovery path when settle kept failing
        # on a malformed plan.md (gated in hooks/guard_bash by
        # state.pending_settle being non-null). The new plan retires
        # the broken plan_version, so the orphan kd_json is no longer
        # actionable; clear it as part of the post-create_plan
        # transition.
        state = load_state(task_dir) or {}
        pending = state.get("pending_settle")
        if phase == EDIT and not pending:
            # Defense-in-depth: guard_bash should have blocked this,
            # but if it slipped through, refuse to advance state.
            emit_status("[AR] create_plan.py in EDIT phase requires a "
                        "pending settle recovery state; nothing to do.")
            sys.exit(0)
        ok, err = validate_plan(task_dir)
        # Cross-file consistency: plan.md version vs
        # state.expected_plan_version. create_plan.py is the only
        # writer that should bump both; if they disagree, the
        # operator should re-run create_plan with the same XML.
        if ok:
            rep = check_state_consistency(task_dir)
            if not rep["consistent"]:
                emit_status(
                    f"[AR] state.json and plan.md/history.jsonl are "
                    f"out of sync. Issues: {rep['issues']}. Re-run "
                    f"create_plan.py with the same XML — both files "
                    f"will reconverge.")
                sys.exit(0)
            _reset_failures_for_diagnose(task_dir, phase)
            PhaseController(task_dir).on_plan_validated()
            if phase == EDIT:
                # Recovery completed: discard the orphan kd_json.
                state = load_state(task_dir) or {}
                state["pending_settle"] = None
                save_state(task_dir, state)
                emit_status(f"[AR] Pending settle abandoned; new plan "
                            f"installed. Phase -> EDIT. {get_guidance(task_dir)}")
            else:
                emit_status(f"[AR] Plan validated. Phase -> EDIT. {get_guidance(task_dir)}")
            emit_todowrite_context(task_dir, "[AR] Plan validated. Phase -> EDIT.")
        else:
            emit_status(f"[AR] Plan not valid yet: {err}")

    sys.exit(0)


def _print_resume_context(task_dir: str):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            emit_status(f"[AR] Active item: {active['id']}: {active['description'][:50]}")
        emit_status("[AR] Read .ar_state/plan.md and .ar_state/history.jsonl for full context.")


if __name__ == "__main__":
    main()
