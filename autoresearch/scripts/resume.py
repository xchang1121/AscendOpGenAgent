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
    ALL_PHASES, load_progress,
    plan_path, state_path, edit_marker_path,
    has_pending_items, find_active_task_dir,
)
from utils.settings import heartbeat_fresh_seconds  # noqa: E402


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
        return False, "Missing or corrupt .ar_state/progress.json — task was never initialized"

    required_fields = {"task", "eval_rounds", "max_rounds"}
    missing = required_fields - set(progress.keys())
    if missing:
        return False, f"progress.json missing fields: {missing} (incompatible version)"

    # Validate .phase if present. Go through phase_machine.read_phase
    # — it owns the on-disk format (today `PHASE|<txn>`, formerly
    # `PHASE`) and the corrupt-content handling. Reading raw here
    # would re-parse the format incorrectly the next time the writer
    # side evolves (which already happened once: the .txn refactor
    # added the `|<txn>` suffix and this bypass would have rejected
    # every fresh task as "Unknown phase 'EDIT|7'").
    phase_file = state_path(task_dir, ".phase")
    if os.path.exists(phase_file):
        from phase_machine import read_phase, INIT
        phase = read_phase(task_dir)
        # read_phase falls back to INIT on corrupt content (with a
        # stderr warning); INIT here means the file existed but was
        # unparseable, which is the same incompatible-version signal
        # the old check produced.
        if phase == INIT:
            return False, ("Unparseable .phase file (read_phase fell "
                           "back to INIT — see stderr for details)")

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
    """Check if another Claude Code instance is actively running this task.

    Uses .ar_state/.heartbeat file mtime — if updated in last 3 minutes, warn.
    """
    heartbeat = state_path(task_dir, ".heartbeat")
    if not os.path.exists(heartbeat):
        return

    import time
    age = time.time() - os.path.getmtime(heartbeat)
    if age < heartbeat_fresh_seconds():  # config.yaml resume.heartbeat_fresh_seconds
        if force:
            print(f"[resume] WARNING: Task was active {age:.0f}s ago. Forcing takeover (--force).",
                  file=sys.stderr)
            return
        print(f"[resume] ERROR: Task is currently active (heartbeat updated {age:.0f}s ago).",
              file=sys.stderr)
        print(f"[resume] Another Claude Code session may be running it.", file=sys.stderr)
        print(f"[resume] If you're sure no other session is running, add --force:", file=sys.stderr)
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

    progress = load_progress(task_dir) or {}
    print(f"[resume] Task: {progress.get('task')}")
    print(f"[resume] Round: {progress.get('eval_rounds')}/{progress.get('max_rounds')}")
    print(f"[resume] Best: {progress.get('best_metric')} | Baseline: {progress.get('baseline_metric')}")
    phase_file = state_path(task_dir, ".phase")
    if os.path.exists(phase_file):
        # Same reason as the validator above — go through read_phase
        # so the displayed string is the phase NAME, not the raw
        # `PHASE|<txn>` on-disk form.
        from phase_machine import read_phase
        print(f"[resume] Phase: {read_phase(task_dir)}")

    # Print task_dir on last line (for easy parsing)
    print(task_dir)


if __name__ == "__main__":
    main()
