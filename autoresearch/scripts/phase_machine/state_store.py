"""State storage layer.

Single per-task state file at ``<task_dir>/.ar_state/state.json``.
Every piece of "control state" (current phase, ownership, heartbeat,
all progress accounting, pending-settle sentinel) lives in this one
record. Atomic write of state.json IS the transaction commit; there's
no separate marker file. Cross-file consistency with the two durable
artifacts that DON'T live inside state.json (``plan.md`` and
``history.jsonl``) is checked by comparing their current shape against
``state.expected_plan_version`` / ``state.expected_history_round``.

Files this module owns (writes):
  - <task_dir>/.ar_state/state.json   single source of truth (this file)

Files this module reads (artifacts written elsewhere):
  - <task_dir>/.ar_state/history.jsonl   append-only round records
  - <task_dir>/.ar_state/plan.md         agent-facing plan
  - <task_dir>/.ar_state/diagnose_v<N>.md / plan_items.xml / .edit_started

No backwards-compat: callers that still reference the retired sidecars
(.phase / progress.json / .pending_settle.json / .heartbeat / .txn /
.active_task) get an ImportError on missing constants. Migrate to the
state.json helpers.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
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
# Filenames inside <task_dir>/.ar_state/
# ---------------------------------------------------------------------------

STATE_FILE = "state.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
PLAN_ITEMS_FILE = "plan_items.xml"  # agent-written XML, validated by create_plan
EDIT_MARKER_FILE = ".edit_started"

# DIAGNOSE artifact — see CLAUDE.md invariant #10.
DIAGNOSE_ARTIFACT_TEMPLATE = "diagnose_v{}.md"
DIAGNOSE_MARKER_TEMPLATE = "[AR DIAGNOSE COMPLETE marker_v{}]"
DIAGNOSE_ATTEMPTS_CAP = 5


# ---------------------------------------------------------------------------
# Project root + per-op pointer (scaffold -> batch/run.py handoff)
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """The autoresearch project root. Derived from this file's fixed
    location: <autoresearch_root>/scripts/phase_machine/state_store.py.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


_PROJECT_ROOT = _find_project_root()
_TASK_DIR_POINTERS = os.path.join(_PROJECT_ROOT, ".task_dir_pointers")


def task_dir_pointer_path(op_name: str) -> str:
    """Per-op pointer file path. scaffold writes the task_dir here
    immediately after creating <repo>/ar_tasks/<op>_<ts>_<rand>;
    batch/run.py reads it instead of mtime-scanning."""
    safe = op_name.replace("/", "_").replace("\\", "_")
    return os.path.join(_TASK_DIR_POINTERS, safe)


def write_task_dir_pointer(op_name: str, task_dir: str) -> None:
    """Atomic write."""
    path = task_dir_pointer_path(op_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(task_dir))
    os.replace(tmp, path)


def read_task_dir_pointer(op_name: str) -> Optional[str]:
    """Returns absolute task_dir, or None when missing / dangling."""
    path = task_dir_pointer_path(op_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    return td if td and os.path.isdir(td) else None


# ---------------------------------------------------------------------------
# Path builders for .ar_state/ files
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Generic path builder for any file under <task_dir>/.ar_state/."""
    return os.path.join(task_dir, ".ar_state", name)


def state_record_path(task_dir: str) -> str:
    return state_path(task_dir, STATE_FILE)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, PLAN_FILE)


def history_path(task_dir: str) -> str:
    return state_path(task_dir, HISTORY_FILE)


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, EDIT_MARKER_FILE)


def diagnose_artifact_path(task_dir: str, plan_version: int) -> str:
    return state_path(task_dir, DIAGNOSE_ARTIFACT_TEMPLATE.format(plan_version))


def diagnose_marker(plan_version: int) -> str:
    return DIAGNOSE_MARKER_TEMPLATE.format(plan_version)


# ---------------------------------------------------------------------------
# state.json — load / save / update primitives
# ---------------------------------------------------------------------------
# Schema (every key documented; missing keys at load → default value):
#
#   phase                      str    one of ALL_PHASES; defaults to INIT
#   owner                      dict|None  {session_id, pid, claimed_at};
#                                         None when no Claude session is
#                                         driving the task
#   last_touched               ISO    bumped by touch_heartbeat / save_state
#
#   # Progress accounting (was progress.json; Progress dataclass fields)
#   task, eval_rounds, max_rounds, consecutive_failures,
#   best_metric, best_commit, baseline_metric, baseline_source,
#   baseline_outcome, baseline_error_source, baseline_per_shape_us,
#   baseline_fingerprint, seed_metric, plan_version, next_pid,
#   num_cases, per_shape_descs, diagnose_attempts,
#   diagnose_attempts_for_version, last_diagnose_failure_reason
#
#   # Pending settle (was .pending_settle.json; None when no replay needed)
#   pending_settle             dict|None  kd_json from a round whose
#                                         settle hasn't committed yet
#
#   # Cross-file artifact expectations (subsumes the per-file _txn_id
#   # markers that lived in plan.md / history.jsonl / etc.)
#   expected_plan_version      int    plan.md's "# Plan vN" must match
#   expected_history_round     int    history.jsonl last row's "round"
#
# Atomic save_state(td, state) is the transaction commit. Body artifacts
# (plan.md / history.jsonl) are written FIRST, then state.json with the
# updated expected_* fields. A crash before save_state leaves the body
# artifacts ahead of state's expectations — check_state_consistency
# reports it, the recovery path is to re-run the writer (idempotent).


_PROGRESS_FIELD_NAMES = {f.name for f in Progress.__dataclass_fields__.values()}


def load_state(task_dir: str) -> Optional[dict]:
    """Read state.json into a dict, or None when missing / corrupt.
    Single canonical reader."""
    path = state_record_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[state_store] WARNING: {path} corrupt ({e}); treating "
              f"as missing. Delete the file and re-run to recover.",
              file=sys.stderr)
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(task_dir: str, state: dict) -> None:
    """Atomic write of the full state record. Bumps last_touched.
    Callers pass the COMPLETE new state dict — partial updates go
    through update_state below."""
    state = dict(state)
    state["last_touched"] = _now_iso()
    path = state_record_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(state), f, indent=2)
    os.replace(tmp, path)


def update_state(task_dir: str, **fields) -> dict:
    """Load → merge fields → atomic save. Returns the post-merge state.
    Convenience for single-field updates (touch_heartbeat, owner
    changes); transactional callers that mutate many fields should
    build the dict explicitly and call save_state."""
    state = load_state(task_dir) or _fresh_state()
    state.update(fields)
    save_state(task_dir, state)
    return state


def _fresh_state() -> dict:
    """Default state record for a task that has none yet on disk."""
    return {
        "phase": INIT,
        "owner": None,
        "last_touched": _now_iso(),
        "task": "",
        "eval_rounds": 0,
        "max_rounds": 0,
        "consecutive_failures": 0,
        "best_metric": None,
        "best_commit": None,
        "baseline_metric": None,
        "baseline_source": None,
        "baseline_outcome": None,
        "baseline_error_source": None,
        "baseline_per_shape_us": None,
        "baseline_fingerprint": None,
        "seed_metric": None,
        "plan_version": 0,
        "next_pid": 0,
        "num_cases": 1,
        "per_shape_descs": None,
        "diagnose_attempts": 0,
        "diagnose_attempts_for_version": None,
        "last_diagnose_failure_reason": None,
        "pending_settle": None,
        "expected_plan_version": 0,
        "expected_history_round": 0,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Ownership (was .active_task at repo level; now embedded in state.owner)
# ---------------------------------------------------------------------------
# .active_task is gone. "Which task is the current Claude session
# driving?" is answered by scanning ar_tasks/ and matching state.json's
# owner.session_id against the env's CLAUDE_CODE_SESSION_ID. Supervisors
# (batch/run.py) have no Claude session and pass expected_task_dir to
# clear_active_task to release a task they themselves spawned.

def _our_session_id() -> str:
    """Caller's Claude Code session id (empty when not inside an agent
    process — supervisors like batch/run.py)."""
    return os.environ.get("CLAUDE_CODE_SESSION_ID", "")


def _heartbeat_fresh(state: dict) -> bool:
    """True iff state.last_touched is within heartbeat_fresh_seconds.
    Loud-fallback to 180s if settings is unreachable."""
    try:
        from utils.settings import heartbeat_fresh_seconds as _hb
        window = _hb()
    except Exception as e:
        print(f"[state_store] WARNING: heartbeat_fresh_seconds() "
              f"unavailable ({e}); falling back to 180s.", file=sys.stderr)
        window = 180
    age = _age_seconds(state.get("last_touched"))
    return age < window


def _age_seconds(iso_str: Optional[str]) -> float:
    if not iso_str:
        return float("inf")
    try:
        ts = datetime.fromisoformat(iso_str).timestamp()
        return time.time() - ts
    except (ValueError, TypeError):
        return float("inf")


def _iter_task_dirs():
    """Yield absolute paths of ar_tasks/<dir>/ whose .ar_state/state.json
    exists. Order is undefined; caller sorts as needed."""
    root = os.path.join(_PROJECT_ROOT, "ar_tasks")
    if not os.path.isdir(root):
        return
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        full = os.path.join(root, name)
        if os.path.isdir(full) and os.path.exists(state_record_path(full)):
            yield full


def get_task_dir() -> str:
    """Return the task_dir owned by the current Claude session, or ""
    when none is found. Falls back to AR_TASK_DIR env var (used by
    legacy scripts that pass task_dir via env)."""
    session = _our_session_id()
    if session:
        for td in _iter_task_dirs():
            st = load_state(td)
            if not st:
                continue
            owner = st.get("owner") or {}
            if owner.get("session_id") == session:
                return td
    return os.environ.get("AR_TASK_DIR", "")


def set_task_dir(task_dir: str, *, force: bool = False) -> bool:
    """Claim `task_dir` for the current session. Returns True on
    success, False when refused.

    Refuse-overwrite logic (mirrors clear_active_task):
      1. force=True                                           → write
      2. no existing owner on `task_dir`                      → write
      3. existing owner.session_id == our session             → write
         (legitimate re-claim by the same agent)
      4. existing owner.session_id != ours, but state's
         last_touched is older than heartbeat_fresh_seconds   → write
         (prior owner is silent → presumed dead)
      5. otherwise (different session, fresh heartbeat)       → refuse
    """
    if not os.path.isdir(task_dir):
        return False
    state = load_state(task_dir) or _fresh_state()
    if not force:
        existing = state.get("owner") or {}
        our_session = _our_session_id()
        existing_sid = existing.get("session_id") or ""
        same_session = our_session and existing_sid == our_session
        if existing_sid and not same_session and _heartbeat_fresh(state):
            print(f"[state_store] WARNING: refusing to claim {task_dir} "
                  f"— owned by session_id={existing_sid} "
                  f"(heartbeat fresh). Our session_id="
                  f"{our_session or '<none>'}. Stop the other session "
                  f"or rm state.json's owner to take over.",
                  file=sys.stderr)
            return False
    state["owner"] = {
        "session_id": _our_session_id(),
        "pid":        os.getpid(),
        "claimed_at": _now_iso(),
    }
    save_state(task_dir, state)
    return True


def clear_active_task(expected_task_dir: Optional[str] = None,
                      *, force: bool = False) -> bool:
    """Release ownership.

    Two caller patterns:
      - in-session hook releasing its own task: pass expected_task_dir
        = our owned task, or omit to auto-find via session match
      - supervisor (batch/run.py) releasing a task it spawned: pass
        expected_task_dir = the task that just finished

    Decision (each step short-circuits to clear+True):
      1. force=True with a target → clear unconditionally
      2. session match            → clear (mine via session)
      3. expected_task_dir match  → clear (mine via supervisor claim)
      4. heartbeat stale          → clear (prior owner dead)
      5. otherwise                → refuse (live different session)
    """
    targets = []
    if expected_task_dir:
        if os.path.isdir(expected_task_dir):
            targets = [os.path.abspath(expected_task_dir)]
    else:
        # No explicit target — scan for one owned by our session.
        for td in _iter_task_dirs():
            st = load_state(td)
            if not st:
                continue
            owner = st.get("owner") or {}
            if owner.get("session_id") == _our_session_id():
                targets.append(td)

    if not targets:
        return True  # nothing to clear

    cleared_any = False
    our_session = _our_session_id()
    for td in targets:
        state = load_state(td)
        if not state or not state.get("owner"):
            cleared_any = True
            continue

        if force:
            state["owner"] = None
            save_state(td, state)
            cleared_any = True
            continue

        owner = state["owner"] or {}
        owner_sid = owner.get("session_id") or ""
        same_session = our_session and owner_sid == our_session
        supervisor_claim = (expected_task_dir
                            and os.path.abspath(expected_task_dir) == td)

        if same_session or supervisor_claim or not _heartbeat_fresh(state):
            state["owner"] = None
            save_state(td, state)
            cleared_any = True
            continue

        print(f"[state_store] WARNING: refusing to clear ownership of "
              f"{td} — owned by session_id={owner_sid} (heartbeat "
              f"fresh, neither session-match nor supervisor claim). "
              f"Pass force=True only if you've verified that session "
              f"is truly done.", file=sys.stderr)
        return False

    return cleared_any or True


def find_active_task_dir() -> Optional[str]:
    """Pick the "current" task — used by dashboard / resume / batch
    monitor when no specific task is in mind.

    Priority:
      1. A task whose owner.session_id matches our env (we're inside
         that agent)
      2. The task with the most-recent last_touched among those with
         an owner (= last live activation, regardless of which agent)
      3. The most-recently touched task overall (no owner anywhere,
         dashboard fallback)
      4. None
    """
    our_session = _our_session_id()
    owned_match: Optional[tuple] = None
    owned_freshest: Optional[tuple] = None
    any_freshest: Optional[tuple] = None
    for td in _iter_task_dirs():
        st = load_state(td)
        if not st:
            continue
        last_touched = _age_seconds(st.get("last_touched"))
        cand = (last_touched, td, st)
        if any_freshest is None or last_touched < any_freshest[0]:
            any_freshest = cand
        owner = st.get("owner") or {}
        if owner.get("session_id"):
            if owned_freshest is None or last_touched < owned_freshest[0]:
                owned_freshest = cand
            if our_session and owner.get("session_id") == our_session:
                if owned_match is None or last_touched < owned_match[0]:
                    owned_match = cand
    for pick in (owned_match, owned_freshest, any_freshest):
        if pick is not None:
            return pick[1]
    return None


def touch_heartbeat(task_dir: str):
    """Bump state.last_touched. Cheap atomic write — every hook fire
    calls this. Failed touch is reported to stderr — silently
    swallowing would make the session look dead in a hard-to-debug
    way."""
    try:
        state = load_state(task_dir) or _fresh_state()
        save_state(task_dir, state)
    except Exception as e:
        print(f"[AR] WARNING: heartbeat write failed ({e}); resume.py "
              f"may misreport this task as inactive.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase R/W (was .phase; now state.json.phase)
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Return the current phase from state.json, or INIT when state
    is missing / phase value is corrupt (with a stderr warning so the
    corrupt-state path is visible during recovery)."""
    state = load_state(task_dir)
    if state is None:
        return INIT
    phase = state.get("phase")
    if phase in ALL_PHASES:
        return phase
    print(f"[state_store] WARNING: state.json has unrecognised phase "
          f"{phase!r}; treating as INIT. Recovery options: re-run "
          f"baseline.py / pipeline.py to advance, or delete "
          f"{state_record_path(task_dir)} to start over.",
          file=sys.stderr)
    return INIT


def write_phase(task_dir: str, phase: str):
    """Write phase into state.json. Atomic single-file commit; no
    cross-file coordination needed here."""
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    state = load_state(task_dir) or _fresh_state()
    state["phase"] = phase
    save_state(task_dir, state)


# ---------------------------------------------------------------------------
# Progress R/W (was progress.json; now Progress fields embedded in state)
# ---------------------------------------------------------------------------

def load_progress(task_dir: str) -> Optional[Progress]:
    """Read the Progress dataclass view from state.json, or None when
    state is missing. Existing read sites use `progress.get("X",
    default)`; Progress.get mirrors dict.get so they keep working."""
    state = load_state(task_dir)
    if state is None:
        return None
    progress_fields = {k: v for k, v in state.items()
                       if k in _PROGRESS_FIELD_NAMES}
    return Progress.from_dict(progress_fields)


def save_progress(task_dir: str, progress: Union[Progress, dict],
                  *, stamp: bool = True):
    """Merge progress fields into state.json and atomically save.
    `stamp=True` updates the in-state `last_updated` field (Progress
    schema's own timestamp, distinct from state.last_touched)."""
    state = load_state(task_dir) or _fresh_state()
    if isinstance(progress, Progress):
        if stamp:
            progress = progress.apply(last_updated=_now_iso())
        payload = progress.to_dict()
    else:
        payload = dict(progress)
        if stamp:
            payload["last_updated"] = _now_iso()
    for k, v in payload.items():
        if k in _PROGRESS_FIELD_NAMES:
            state[k] = v
    save_state(task_dir, state)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Append-only artifact;
    each row is self-contained and immutable. Cross-file consistency
    with state.json's `expected_history_round` is the caller's
    responsibility (typically: bump expected_history_round in the
    same save_state that wraps this round's body writes)."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_floats(record), ensure_ascii=False) + "\n")


def update_progress(task_dir: str, **fields) -> Optional[Progress]:
    """Load Progress, .apply(**fields), save. Returns the new Progress.

    Field-name validation is delegated to Progress.apply, so a typo
    here becomes TypeError instead of a silently-dropped attribute.

    Returns None only when state.json doesn't exist (pre-scaffold,
    legitimate no-op). Save failures re-raise after a loud stderr
    warning — earlier callers silently lost DIAGNOSE attempt counts
    and consecutive_failures resets when the write failed, producing
    infinite-retry loops the operator couldn't trace back."""
    progress = load_progress(task_dir)
    if progress is None:
        return None
    new_progress = progress.apply(**fields)
    try:
        save_progress(task_dir, new_progress, stamp=False)
    except Exception as e:
        print(f"[state_store] CRITICAL: failed to save state.json for "
              f"{task_dir}: {type(e).__name__}: {e}. fields="
              f"{list(fields)}. The in-memory update is lost; the next "
              f"round may see stale state. Free disk space / fix "
              f"permissions and re-run the failed action.",
              file=sys.stderr)
        raise
    return new_progress


# ---------------------------------------------------------------------------
# Cross-file consistency check
# ---------------------------------------------------------------------------
# state.json is the commit barrier. plan.md and history.jsonl are
# durable artifacts written outside state.json — a writer that landed
# them but didn't commit state.json leaves a detectable gap. The check
# below compares state.expected_plan_version / expected_history_round
# against the actual artifact contents.

def _read_plan_version_from_disk(task_dir: str) -> Optional[int]:
    """Parse plan.md's `# Plan vN` header; None when plan.md missing
    or unparseable. (PlanStore.parse_version_on_disk exists too but
    we keep this local to avoid the workflow→phase_machine import
    cycle.)"""
    path = plan_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return None
    import re as _re
    m = _re.match(r"^#\s*Plan\s+v(\d+)\b", first)
    return int(m.group(1)) if m else None


def _read_last_history_round(task_dir: str) -> Optional[int]:
    """Parse history.jsonl's last row's `round` field; None when no
    history yet or the last row is unparseable."""
    path = history_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            try:
                size = os.path.getsize(path)
                if size == 0:
                    return None
                f.seek(-1, os.SEEK_END)
                while f.tell() > 0:
                    if f.read(1) == b"\n":
                        if f.tell() == size:
                            f.seek(-2, os.SEEK_END)
                            continue
                        break
                    f.seek(-2, os.SEEK_CUR)
                last = f.readline().decode("utf-8", errors="replace").strip()
            except OSError:
                return None
        if not last:
            return None
        row = json.loads(last)
        v = row.get("round")
        return int(v) if isinstance(v, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def check_state_consistency(task_dir: str) -> dict:
    """Return a report:
        {"consistent": bool,
         "state": <full state dict or None>,
         "issues": [<human-readable issue strings>]}

    issues is empty when state is consistent; otherwise lists each
    artifact-vs-state mismatch.
    """
    state = load_state(task_dir)
    if state is None:
        # No state → nothing to compare. Treat as consistent (fresh
        # task or pre-write state).
        return {"consistent": True, "state": None, "issues": []}

    issues = []
    expected_plan = int(state.get("expected_plan_version") or 0)
    plan_on_disk = _read_plan_version_from_disk(task_dir)
    if plan_on_disk is not None and plan_on_disk != expected_plan:
        issues.append(
            f"plan.md is at v{plan_on_disk} but state.expected_plan_"
            f"version={expected_plan}. The plan writer (create_plan.py "
            f"or pipeline.py's settle path) landed plan.md but did not "
            f"commit the matching state.json. Re-run the original "
            f"writer with the same input — both reach the same target "
            f"version idempotently.")

    expected_round = int(state.get("expected_history_round") or 0)
    last_round = _read_last_history_round(task_dir)
    if last_round is not None and last_round != expected_round:
        issues.append(
            f"history.jsonl last row is round {last_round} but state."
            f"expected_history_round={expected_round}. Round writer "
            f"appended history but did not commit state.json. Re-run "
            f"pipeline.py — the pending_settle path will reconcile.")

    return {"consistent": not issues, "state": state, "issues": issues}


def format_state_inconsistency(report: dict) -> str:
    """Render a check_state_consistency report into a recovery
    message suitable for hook stderr / agent transcript."""
    if report["consistent"]:
        return ""
    state = report.get("state") or {}
    phase = state.get("phase") or "<unknown>"
    head = (f".ar_state is inconsistent (phase={phase}). Writer landed "
            f"body artifacts but never committed state.json:")
    return head + "\n  - " + "\n  - ".join(report["issues"])


def require_state_consistency(task_dir: str,
                              *, on_inconsistent: str = "raise") -> dict:
    """Cross-file consistency gate for activation hooks / resume. On
    inconsistency:
      - on_inconsistent="raise" (default): RuntimeError with the
        recovery message. Pipelines fail loud.
      - on_inconsistent="report": return the report; caller surfaces
        the message its own way.
    """
    report = check_state_consistency(task_dir)
    if report["consistent"]:
        return report
    if on_inconsistent == "raise":
        raise RuntimeError(format_state_inconsistency(report))
    return report
