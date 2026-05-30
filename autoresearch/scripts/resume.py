#!/usr/bin/env python3
"""
Resume an existing autoresearch task.

Usage:
    python scripts/resume.py [task_dir]

If task_dir is omitted, auto-detects the most recently active task.
Validates state files. Prints the task_dir on success, exits with error if incompatible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_machine import (
    load_progress, plan_path, edit_marker_path,
    has_pending_items, find_active_task_dir,
    is_task_fresh, task_summary,
)


def _validate(task_dir: str) -> tuple[bool, str]:
    """Check task state is resumable. Returns (ok, error_message)."""
    if not os.path.isdir(task_dir):
        return False, f"Not a directory: {task_dir}"

    # Required files
    for rel in ("task.yaml",):
        if not os.path.exists(os.path.join(task_dir, rel)):
            return False, f"Missing required file: {rel}"

    progress = load_progress(task_dir)
    if progress is None:
        # Either state.json is missing entirely (task was never
        # claimed) or baseline never landed (progress_initialized is
        # False). Both mean "nothing to resume" — start fresh.
        return False, ("No measured progress yet (state.json missing or "
                       "baseline never committed). Run /autoresearch "
                       "without --resume to start a fresh task.")

    required_fields = {"task", "eval_rounds", "max_rounds"}
    missing = required_fields - set(progress.keys())
    if missing:
        return False, f"state.json progress fields missing: {missing}"

    # Validate plan.md if present. A fully-consumed plan (0 pending) is legal —
    # compute_resume_phase routes it to REPLAN. validate_plan would reject it
    # for lacking an ACTIVE item, so only validate when pending items exist.
    if os.path.exists(plan_path(task_dir)) and has_pending_items(task_dir):
        from phase_machine import validate_plan
        ok, err = validate_plan(task_dir)
        if not ok:
            return False, f"plan.md invalid: {err}"

    # Cross-file consistency: plan.md / history.jsonl vs state.json's
    # expected_*. Refuse to attach if the previous writer landed body
    # bytes but state.json wasn't committed — operator re-runs the
    # original writer first.
    from phase_machine import (
        require_state_consistency as _require,
        format_state_inconsistency as _fmt_inconsistency,
    )
    rep = _require(task_dir, on_inconsistent="report")
    if not rep["consistent"]:
        return False, _fmt_inconsistency(rep)

    return True, ""


def _check_active_lock(task_dir: str, force: bool) -> None:
    """Refuse to attach when another Claude Code session is actively
    driving this task. "Active" = state.last_touched is within
    `heartbeat_fresh_seconds`. The retired `.heartbeat` sidecar is
    gone; heartbeats are bumped in-place on state.json by
    touch_heartbeat, and `is_task_fresh` is the only thing callers
    should ask about freshness."""
    if not is_task_fresh(task_dir):
        return
    if force:
        print(f"[resume] WARNING: Task is fresh (state.last_touched within "
              f"the heartbeat window). Forcing takeover (--force).",
              file=sys.stderr)
        return
    summary = task_summary(task_dir) or {}
    owner = summary.get("owner") or {}
    print(f"[resume] ERROR: Task is currently active "
          f"(state.last_touched={summary.get('last_touched')}, "
          f"owner.session_id={owner.get('session_id') or '<none>'}).",
          file=sys.stderr)
    print(f"[resume] Another Claude Code session may be running it.",
          file=sys.stderr)
    print(f"[resume] If you're sure no other session is running, add --force:",
          file=sys.stderr)
    print(f"[resume]   /autoresearch --resume --force", file=sys.stderr)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    task_dir = args[0] if args else None

    if task_dir:
        task_dir = os.path.abspath(task_dir)
    else:
        task_dir = find_active_task_dir() or ""
        if not task_dir:
            print("[resume] ERROR: No existing task found in ar_tasks/", file=sys.stderr)
            sys.exit(1)

    ok, err = _validate(task_dir)
    if not ok:
        print(f"[resume] ERROR: Cannot resume {task_dir}", file=sys.stderr)
        print(f"[resume] {err}", file=sys.stderr)
        print("[resume] This task may be from an incompatible version. Start fresh.", file=sys.stderr)
        sys.exit(1)

    _check_active_lock(task_dir, force)

    # Clean stale edit marker (git clean means marker is stale)
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        from utils.git_utils import is_working_tree_clean
        if is_working_tree_clean(task_dir):
            try:
                os.remove(marker)
                print("[resume] Cleaned stale edit marker.", file=sys.stderr)
            except OSError:
                pass

    # task_summary is the only thing this script needs from the state
    # store — it's the single shape callers should agree on after the
    # state.json unification. Reaching into individual fields used to
    # be the source of resume vs dashboard skew on schema changes.
    summary = task_summary(task_dir) or {}
    print(f"[resume] Task: {summary.get('task')}")
    print(f"[resume] Round: {summary.get('eval_rounds')}/{summary.get('max_rounds')}")
    print(f"[resume] Best: {summary.get('best_metric')} | "
          f"Baseline: {summary.get('baseline_metric')}")
    print(f"[resume] Phase: {summary.get('phase')}")

    # Print task_dir on last line (for easy parsing)
    print(task_dir)


if __name__ == "__main__":
    main()
