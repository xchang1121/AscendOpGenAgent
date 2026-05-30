"""State storage layer.

Single owner of `<task_dir>/.ar_state/` and `autoresearch/.active_task`.
No other module reads/writes these files directly — go through the helpers
here.

What lives in this module:
  - Phase enum constants (used as keys / values throughout).
  - Canonical file basenames inside `.ar_state/` (PHASE_FILE, etc.).
  - Path builders (`state_path`, `plan_path`, `progress_path`, …).
  - Phase I/O (`read_phase`, `write_phase`).
  - Progress I/O (`load_progress` -> Progress, `save_progress`,
    `update_progress`). Progress is a typed dataclass (see models.py)
    so writers construct full objects and the field set is validated.
  - History append (`append_history`).
  - Active-task pointer (`get_task_dir`, `set_task_dir`).
  - Heartbeat touch.
  - JSON-tail parser used by every subprocess output.

Why phase constants live here and not in phase_policy: `read_phase` needs
`ALL_PHASES` to validate; phase_policy in turn needs `compute_next_phase`
to read progress, which lives here. Putting the constants at the bottom
of the dependency stack avoids the cycle.
"""
import json
import os
import sys
from typing import Optional, Union

# state_store is imported by hook code that may run before scripts/ is
# on sys.path (no editable install). Make the import work either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_HERE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from utils.json_io import sanitize_floats  # noqa: E402

from .models import Progress


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

INIT = "INIT"
BASELINE = "BASELINE"
PLAN = "PLAN"
EDIT = "EDIT"
DIAGNOSE = "DIAGNOSE"
REPLAN = "REPLAN"
FINISH = "FINISH"

ALL_PHASES = {INIT, BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, FINISH}


# ---------------------------------------------------------------------------
# Canonical filenames inside <task_dir>/.ar_state/
# ---------------------------------------------------------------------------

PHASE_FILE = ".phase"
PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
PLAN_ITEMS_FILE = "plan_items.xml"  # canonical XML payload path under .ar_state/
EDIT_MARKER_FILE = ".edit_started"
PENDING_SETTLE_FILE = ".pending_settle.json"  # kd_json saved when settle.py fails
HEARTBEAT_FILE = ".heartbeat"
ACTIVE_TASK_FILE = ".active_task"  # under autoresearch/, not .ar_state/

# DIAGNOSE artifact contract — see CLAUDE.md invariant #10.
# The DIAGNOSE phase is gated on a structured report at this path before
# create_plan.py / Stop become legal. The ar-diagnosis subagent is the
# intended writer (per its prompt + read-only tool isolation), but hook
# payloads do NOT distinguish main agent from subagent — provenance is
# not enforced. Only the artifact's CONTENT is validated. The marker is
# plan-version-aware so a stale prior diagnose can't be replayed across
# REPLAN boundaries.
DIAGNOSE_ARTIFACT_TEMPLATE = "diagnose_v{}.md"
DIAGNOSE_MARKER_TEMPLATE = "[AR DIAGNOSE COMPLETE marker_v{}]"
DIAGNOSE_ATTEMPTS_CAP = 5


# ---------------------------------------------------------------------------
# Project root resolution + active-task pointer
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """The autoresearch project root (contains scripts/, config.yaml,
    .claude/, ar_tasks/, .active_task). Derived from this file's
    fixed location: <autoresearch_root>/scripts/phase_machine/state_store.py.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


_PROJECT_ROOT = _find_project_root()
_ACTIVE_TASK_FILE = os.path.join(_PROJECT_ROOT, ACTIVE_TASK_FILE)
_TASK_DIR_POINTERS = os.path.join(_PROJECT_ROOT, ".task_dir_pointers")


def task_dir_pointer_path(op_name: str) -> str:
    """Filesystem path of the per-op task_dir pointer.

    scaffold writes this immediately after creating <repo>/ar_tasks/
    <op>_<ts>_<rand>; batch/run.py reads it instead of mtime-scanning.
    The mtime scan still works as a fallback (tasks scaffolded before
    this change have no pointer), but the pointer is the authoritative
    answer when present.
    """
    safe = op_name.replace("/", "_").replace("\\", "_")
    return os.path.join(_TASK_DIR_POINTERS, safe)


def write_task_dir_pointer(op_name: str, task_dir: str) -> None:
    """Atomic write of the per-op pointer. Tmp + os.replace so a racing
    reader sees either nothing or the full path - never a half line."""
    path = task_dir_pointer_path(op_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(task_dir) + "\n")
    os.replace(tmp, path)


def read_task_dir_pointer(op_name: str) -> Optional[str]:
    """Return the absolute task_dir for op_name, or None when no pointer
    exists / contents are stale (the dir no longer exists)."""
    path = task_dir_pointer_path(op_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    if td and os.path.isdir(td):
        return td
    return None


def get_task_dir() -> str:
    """Get active task_dir. Reads from autoresearch/.active_task file.

    Falls back to AR_TASK_DIR env var for backward compat.
    Returns "" if no active task.
    """
    if os.path.exists(_ACTIVE_TASK_FILE):
        with open(_ACTIVE_TASK_FILE, "r") as f:
            td = f.read().strip()
        if td and os.path.isdir(td):
            return td
    return os.environ.get("AR_TASK_DIR", "")


def clear_active_task(*, force: bool = False) -> bool:
    """Remove the repo-wide .active_task pointer if it's safe to do so.
    Returns True when the pointer is gone after the call (deleted or
    already absent), False when refused.

    "Safe" means: the pointer is missing, dangling (points at a deleted
    dir), or its task_dir's heartbeat is older than
    heartbeat_fresh_seconds. A fresh heartbeat means another live
    session (manual Claude, another batch) is actively writing the
    task — silently unlinking would let our caller create its own
    activation and the two sessions would then cross-write state.
    Pass force=True to override.

    Used by batch/run.py between ops (where the prior op's pointer is
    expected to be stale) and any other "supervisor" that knows it
    owns the activation lifecycle. Hooks should keep using
    set_task_dir (which calls into the same heartbeat check on the
    write path).
    """
    if not os.path.exists(_ACTIVE_TASK_FILE):
        return True
    try:
        with open(_ACTIVE_TASK_FILE, "r") as f:
            pointed = f.read().strip()
    except OSError:
        pointed = ""
    if not force and pointed and os.path.isdir(pointed):
        try:
            _scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _scripts not in sys.path:
                sys.path.insert(0, _scripts)
            from utils.settings import heartbeat_fresh_seconds as _hb
            fresh_window = _hb()
        except Exception:
            fresh_window = 180
        import time as _time
        hb_path = state_path(pointed, HEARTBEAT_FILE)
        try:
            age = _time.time() - os.path.getmtime(hb_path)
        except OSError:
            age = float("inf")
        if age < fresh_window:
            print(f"[state_store] WARNING: refusing to clear .active_task "
                  f"— currently points at {pointed} with a fresh heartbeat "
                  f"({age:.0f}s ago, window={fresh_window}s). Another "
                  f"Claude or batch session looks active. Pass "
                  f"force=True only if you've verified that session is "
                  f"truly done.", file=sys.stderr)
            return False
    try:
        os.remove(_ACTIVE_TASK_FILE)
    except OSError:
        pass
    return True


def set_task_dir(task_dir: str, *, force: bool = False) -> bool:
    """Write active task_dir to autoresearch/.active_task. Returns True
    on success, False when the pointer is refused (see below).

    The .active_task pointer is repo-wide and unscoped — two Claude or
    batch sessions sharing the same checkout used to silently clobber
    each other's pointer, and the loser's hooks would then read the
    wrong task_dir and gate / edit against unrelated files. Without
    Claude Code passing a session id to hooks we can't fix the surface
    fully, so the next-best guard: refuse to overwrite when the
    EXISTING pointer's task_dir still has a fresh heartbeat (= some
    other process is actively writing it). Pass force=True to override
    when the caller explicitly knows it's taking over (batch/run.py
    unlinks the pointer first and re-acquires).
    """
    import time as _time
    new_abs = os.path.abspath(task_dir)
    if not force and os.path.exists(_ACTIVE_TASK_FILE):
        try:
            with open(_ACTIVE_TASK_FILE, "r") as f:
                existing = f.read().strip()
        except OSError:
            existing = ""
        if (existing and existing != new_abs and os.path.isdir(existing)):
            # `from .settings import` would resolve to phase_machine.settings
            # which doesn't exist (I shipped this exact bug class in 447da0f
            # and just repeated it in 8baf094) — fallback would always fire
            # and silently lock the freshness window to 180s regardless of
            # what the operator put in config.yaml. utils.settings is the
            # canonical home; reach into it via absolute import.
            try:
                _scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if _scripts not in sys.path:
                    sys.path.insert(0, _scripts)
                from utils.settings import heartbeat_fresh_seconds as _hb
                fresh_window = _hb()
            except Exception as _e:
                import sys as _sys
                print(f"[state_store] WARNING: utils.settings.heartbeat_"
                      f"fresh_seconds() unavailable ({_e}); falling back "
                      f"to 180s. .active_task guard may be tighter than "
                      f"you configured.", file=_sys.stderr)
                fresh_window = 180
            hb_path = state_path(existing, HEARTBEAT_FILE)
            try:
                age = _time.time() - os.path.getmtime(hb_path)
            except OSError:
                age = float("inf")
            if age < fresh_window:
                import sys as _sys
                print(f"[state_store] WARNING: refusing to overwrite "
                      f".active_task — currently points at {existing} "
                      f"with a fresh heartbeat ({age:.0f}s ago, "
                      f"window={fresh_window}s). Another Claude or batch "
                      f"session looks active on that task. To take over "
                      f"explicitly, delete {_ACTIVE_TASK_FILE} first, or "
                      f"call set_task_dir(..., force=True).",
                      file=_sys.stderr)
                return False
    os.makedirs(os.path.dirname(_ACTIVE_TASK_FILE), exist_ok=True)
    with open(_ACTIVE_TASK_FILE, "w") as f:
        f.write(new_abs)
    touch_heartbeat(task_dir)
    return True


def find_active_task_dir() -> Optional[str]:
    """Single source of truth for "which task is currently active".

    Resume / dashboard / batch each historically maintained their own
    rule for this lookup (resume trusted the .active_task pointer first,
    dashboard scanned mtimes first, batch looked only at heartbeats).
    Three rules for one question is a divergence risk — concurrent
    sessions, stale pointers, or batch/manual mixes can make the tools
    disagree about which task is "current". Everyone goes through this
    helper now.

    Resolution rule:
      1. If `autoresearch/.active_task` points at an existing task dir
         → return it. Stale pointers (the task dir was deleted) get
         removed in passing so the next call falls straight to step 2.
      2. Otherwise scan `ar_tasks/` and pick the dir with the most-recent
         signal — heartbeat first (touched every hook fire, freshest),
         then progress.json mtime, then the dir's own mtime. Task dirs
         missing `task.yaml` are ignored (not a real task).
    Returns None if no task is found.
    """
    if os.path.exists(_ACTIVE_TASK_FILE):
        try:
            with open(_ACTIVE_TASK_FILE, "r") as f:
                td = f.read().strip()
        except OSError:
            td = ""
        if td and os.path.isdir(td):
            return td
        # Stale pointer — clean it so future callers skip step 1.
        try:
            os.remove(_ACTIVE_TASK_FILE)
        except OSError:
            pass

    tasks_root = os.path.join(_PROJECT_ROOT, "ar_tasks")
    if not os.path.isdir(tasks_root):
        return None

    best: Optional[str] = None
    best_mt: float = -1.0
    for name in os.listdir(tasks_root):
        full = os.path.join(tasks_root, name)
        if not os.path.isdir(full):
            continue
        if not os.path.exists(os.path.join(full, "task.yaml")):
            continue
        mt = -1.0
        for candidate in (
            state_path(full, HEARTBEAT_FILE),
            progress_path(full),
            full,
        ):
            if os.path.exists(candidate):
                try:
                    mt = max(mt, os.path.getmtime(candidate))
                except OSError:
                    pass
        if mt > best_mt:
            best_mt = mt
            best = full
    return best


def touch_heartbeat(task_dir: str):
    """Update .ar_state/.heartbeat file to signal this task is active.

    Called from every hook invocation. resume.py checks mtime to detect
    conflicting concurrent Claude Code sessions. A failed touch is reported
    to stderr — silently swallowing it would make the session look dead in
    a way that's nearly impossible to debug.
    """
    try:
        heartbeat = state_path(task_dir, HEARTBEAT_FILE)
        os.makedirs(os.path.dirname(heartbeat), exist_ok=True)
        import time
        with open(heartbeat, "w") as f:
            f.write(f"{int(time.time())}\n")
    except Exception as e:
        print(f"[AR] WARNING: heartbeat write failed ({e}); resume.py may "
              f"misreport this task as inactive.", file=sys.stderr)


# ---------------------------------------------------------------------------
# State file path builders
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Path to a file under <task_dir>/.ar_state/. Centralized so no module
    hand-builds state paths."""
    return os.path.join(task_dir, ".ar_state", name)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, PLAN_FILE)


def progress_path(task_dir: str) -> str:
    return state_path(task_dir, PROGRESS_FILE)


def history_path(task_dir: str) -> str:
    return state_path(task_dir, HISTORY_FILE)


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, EDIT_MARKER_FILE)


def pending_settle_path(task_dir: str) -> str:
    """Sidecar holding the kd_json from a settle.py invocation that failed.

    pipeline.py persists the kd_json here when settle returns non-zero, then
    its NEXT invocation detects this file and retries settle ONLY (skipping
    quick_check/eval/keep_or_discard). Without this replay-only path, a
    re-run of pipeline.py would double-mutate progress.json (eval_rounds++)
    and history.jsonl (duplicate row) before the original settle even gets
    a second chance.

    Removed by pipeline.py on successful settle.
    """
    return state_path(task_dir, PENDING_SETTLE_FILE)


def diagnose_artifact_path(task_dir: str, plan_version: int) -> str:
    """Path to the DIAGNOSE artifact for a given plan_version. The subagent
    Writes to this exact path; the validator reads from it. Plan-version
    suffix prevents stale artifacts from satisfying a later DIAGNOSE round."""
    return state_path(task_dir, DIAGNOSE_ARTIFACT_TEMPLATE.format(plan_version))


def diagnose_marker(plan_version: int) -> str:
    return DIAGNOSE_MARKER_TEMPLATE.format(plan_version)


# ---------------------------------------------------------------------------
# Phase file I/O
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Read current phase. Returns INIT if the phase file is missing.

    If the file exists but its content is unrecognised (truncated by a
    crashed mid-write, hand-edited typo, or out-of-date phase name from
    a downgrade), emit a stderr warning AND fall back to INIT — silent
    fallback was hiding mid-write corruption from the operator until
    resume.py also rejected it with a confusing "incompatible state".
    The warning makes the corrupt-state path visible without crashing
    every hook on the way to recovery.
    """
    path = state_path(task_dir, PHASE_FILE)
    if not os.path.exists(path):
        return INIT
    with open(path, "r") as f:
        phase = f.read().strip()
    if phase in ALL_PHASES:
        return phase
    import sys as _sys
    print(f"[state_store] WARNING: {path} contains unrecognised phase "
          f"{phase!r}; treating as INIT. If a hook or pipeline crashed "
          f"mid-write, restoring from .ar_state/history.jsonl or "
          f"deleting {path} and re-running /autoresearch may recover.",
          file=_sys.stderr)
    return INIT


def write_phase(task_dir: str, phase: str):
    """Write phase to .ar_state/.phase atomically.

    tmp + os.replace prevents read_phase from ever seeing an empty /
    half-written file, which previously cascaded into the
    INIT-fallback path above and gated the agent out of all AR
    scripts. PhaseController owns the only callsite, so there is no
    cross-writer race to worry about — atomicity is purely about
    crash-mid-write.
    """
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    path = state_path(task_dir, PHASE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(phase)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Progress + history I/O
# ---------------------------------------------------------------------------

def load_progress(task_dir: str) -> Optional[Progress]:
    """Read .ar_state/progress.json into a typed Progress, or None if
    absent/corrupt. Single canonical reader.

    Existing read sites use `progress.get("X", default)`; Progress.get
    mirrors dict.get so they keep working without any rewrite. New code
    should prefer attribute access (`progress.eval_rounds`).
    """
    path = progress_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return Progress.from_dict(data)


def save_progress(task_dir: str, progress: Union[Progress, dict],
                  *, stamp: bool = True):
    """Write progress to .ar_state/progress.json atomically. Accepts
    Progress or a plain dict (the dict path stays for batch/manifest.py
    which has its own schema and predates the dataclass).

    Atomicity: tmp + os.replace. Earlier non-atomic rewrites occasionally
    let `load_progress` see an empty file mid-write and `compute_next_
    phase` then short-circuit to FINISH well before max_rounds.
    """
    from datetime import datetime, timezone
    path = progress_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(progress, Progress):
        if stamp:
            progress = progress.apply(
                last_updated=datetime.now(timezone.utc).isoformat())
        payload = progress.to_dict()
    else:
        payload = dict(progress)
        if stamp:
            payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(payload), f, indent=2)
    os.replace(tmp_path, path)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Single canonical writer
    used by keep_or_discard and _baseline_init."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_floats(record), ensure_ascii=False) + "\n")


def update_progress(task_dir: str, **fields) -> Optional[Progress]:
    """Load Progress, .apply(**fields), save. Returns the new Progress.

    Field-name validation is delegated to Progress.apply, so a typo here
    becomes TypeError instead of a silently-dropped attribute (which is
    what `progress["typo"] = ...` produced in the dict-era code).

    Returns None only when progress.json does not yet exist (pre-scaffold,
    legitimate no-op). Save failures (disk full, permission, racing
    rename) re-raise after a loud stderr warning — earlier callers
    silently lost DIAGNOSE attempt counts and consecutive_failures
    resets when the underlying write failed, producing infinite-retry
    loops or repeated DIAGNOSE entry that the operator couldn't trace
    back to the dropped write.
    """
    progress = load_progress(task_dir)
    if progress is None:
        return None
    new_progress = progress.apply(**fields)
    try:
        save_progress(task_dir, new_progress, stamp=False)
    except Exception as e:
        import sys as _sys
        print(f"[state_store] CRITICAL: failed to save progress.json for "
              f"{task_dir}: {type(e).__name__}: {e}. fields={list(fields)}. "
              f"The in-memory update is lost; the next round may see stale "
              f"state (wrong consecutive_failures, diagnose_attempts, etc). "
              f"Free disk space / fix permissions and re-run the failed "
              f"action.", file=_sys.stderr)
        raise
    return new_progress


# Subprocess output parser lives in utils.json_io now (was duplicated
# here and in task_config.eval_client). Importers that previously did
# `from phase_machine import parse_last_json_line` should switch to
# `from utils.json_io import parse_last_json_line`.
