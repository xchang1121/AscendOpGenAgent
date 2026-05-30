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
TXN_FILE = ".txn"  # monotonic per-task transaction marker — see below

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


# ---------------------------------------------------------------------------
# Active-task ownership record
# ---------------------------------------------------------------------------
# `.active_task` is repo-wide and unscoped: any process can write it,
# and there's no a-priori answer to "who owns this pointer?" without
# an embedded identity. Earlier iterations stitched the question
# together from indirect signals — heartbeat freshness, same task_dir
# string, hook pid — and each indirect signal broke a different
# legitimate caller. The structural fix is to store identity IN the
# record so the question becomes a direct match.
#
# Record format (JSON; written atomically via tmp + os.replace):
#
#   {"task_dir":    "<absolute path>",
#    "session_id":  "<CLAUDE_CODE_SESSION_ID or ''>",
#    "owner_pid":   <int, getpid() of writer>,
#    "claimed_at":  "<ISO 8601 UTC>"}
#
# session_id is the primary ownership token: Claude Code injects
# CLAUDE_CODE_SESSION_ID into every hook process, so hooks of the
# same agent session always match. Supervisors (batch/run.py) have
# no Claude session of their own; they can still claim ownership by
# passing expected_task_dir to clear_active_task — they know which
# task_dir they just spawned + waited on.
#
# Backward compat: a single-line `.active_task` (plain task_dir) loads
# as session_id="" / owner_pid=0. Clear/set then fall back to the
# heartbeat defence so existing checkouts mid-upgrade keep working.

def _load_active_record() -> Optional[dict]:
    """Read `.active_task` into a normalised dict, or None when missing /
    unreadable / corrupt. Legacy (plain task_dir) records load with
    empty session_id and owner_pid=0."""
    if not os.path.exists(_ACTIVE_TASK_FILE):
        return None
    try:
        with open(_ACTIVE_TASK_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[state_store] WARNING: {_ACTIVE_TASK_FILE} JSON "
                  f"corrupt ({e}); treating as no active task. Delete "
                  f"the file and re-run to recover.", file=sys.stderr)
            return None
        return {
            "task_dir":   str(data.get("task_dir", "") or ""),
            "session_id": str(data.get("session_id", "") or ""),
            "owner_pid":  int(data.get("owner_pid") or 0),
            "claimed_at": str(data.get("claimed_at", "") or ""),
        }
    # Legacy single-line form.
    return {"task_dir": raw, "session_id": "", "owner_pid": 0,
            "claimed_at": ""}


def _write_active_record(task_dir: str) -> None:
    """Atomic write of the ownership record. Auto-populates session_id
    from env (hooks inherit CLAUDE_CODE_SESSION_ID from the agent)
    and owner_pid from os.getpid()."""
    from datetime import datetime, timezone
    rec = {
        "task_dir":   os.path.abspath(task_dir),
        "session_id": os.environ.get("CLAUDE_CODE_SESSION_ID", ""),
        "owner_pid":  os.getpid(),
        "claimed_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(_ACTIVE_TASK_FILE), exist_ok=True)
    tmp = _ACTIVE_TASK_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, _ACTIVE_TASK_FILE)


def _our_session_id() -> str:
    """Empty when our caller isn't inside a Claude agent process (e.g.
    batch/run.py supervisor)."""
    return os.environ.get("CLAUDE_CODE_SESSION_ID", "")


def _heartbeat_age_seconds(task_dir: str) -> float:
    """Seconds since the task's `.heartbeat` was touched. inf when no
    such file exists."""
    import time as _time
    try:
        return _time.time() - os.path.getmtime(
            state_path(task_dir, HEARTBEAT_FILE))
    except OSError:
        return float("inf")


def _heartbeat_fresh(task_dir: str) -> bool:
    """True iff the task's heartbeat is younger than
    settings.heartbeat_fresh_seconds(). Loud-fallback to 180s if the
    settings import path is broken (which it shouldn't be — the
    fallback exists to keep the active_task guard usable rather than
    crashing a hook)."""
    try:
        _scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        from utils.settings import heartbeat_fresh_seconds as _hb
        window = _hb()
    except Exception as e:
        print(f"[state_store] WARNING: heartbeat_fresh_seconds() "
              f"unavailable ({e}); falling back to 180s.", file=sys.stderr)
        window = 180
    return _heartbeat_age_seconds(task_dir) < window


def _try_unlink_active() -> None:
    try:
        os.remove(_ACTIVE_TASK_FILE)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public active-task API
# ---------------------------------------------------------------------------

def get_task_dir() -> str:
    """Return the active task_dir from `.active_task`, or AR_TASK_DIR
    env fallback, or "". Reads both the JSON record and legacy single-
    line forms (compat with mid-upgrade checkouts).
    """
    rec = _load_active_record()
    if rec and rec["task_dir"] and os.path.isdir(rec["task_dir"]):
        return rec["task_dir"]
    return os.environ.get("AR_TASK_DIR", "")


def set_task_dir(task_dir: str, *, force: bool = False) -> bool:
    """Claim `.active_task` for `task_dir`. Returns True on write,
    False when refused.

    Decision order (each step short-circuits):
      1. force=True                            → write
      2. no existing record / empty record     → write
      3. existing.session_id == our session_id → write (same agent,
         legitimate focus change)
      4. existing.task_dir dangling            → write (prior dir gone,
         can't be live)
      5. heartbeat stale on existing.task_dir  → write (prior owner is
         no longer touching it)
      6. otherwise                             → refuse (live session
         with a different identity is actively driving a different
         task_dir; silent overwrite would cross-write state)
    """
    new_abs = os.path.abspath(task_dir)
    rec = _load_active_record()

    if force or rec is None or not rec["task_dir"]:
        _write_active_record(new_abs)
        touch_heartbeat(new_abs)
        return True

    our_session = _our_session_id()
    if our_session and rec["session_id"] == our_session:
        _write_active_record(new_abs)
        touch_heartbeat(new_abs)
        return True

    if not os.path.isdir(rec["task_dir"]):
        _write_active_record(new_abs)
        touch_heartbeat(new_abs)
        return True

    if not _heartbeat_fresh(rec["task_dir"]):
        _write_active_record(new_abs)
        touch_heartbeat(new_abs)
        return True

    age = _heartbeat_age_seconds(rec["task_dir"])
    print(f"[state_store] WARNING: refusing to overwrite .active_task — "
          f"held by session_id={rec['session_id'] or '<legacy>'} on "
          f"{rec['task_dir']} (heartbeat {age:.0f}s ago, still fresh). "
          f"Our session_id={our_session or '<none>'}. Stop the other "
          f"session, rm {_ACTIVE_TASK_FILE}, or call "
          f"set_task_dir(..., force=True).", file=sys.stderr)
    return False


def clear_active_task(expected_task_dir: Optional[str] = None,
                      *, force: bool = False) -> bool:
    """Remove `.active_task` if it's safe to do so. Returns True when
    the pointer is gone after the call (deleted or already absent),
    False when refused.

    Decision order (each step short-circuits to unlink + True):
      1. no existing record                          → True
      2. force=True                                  → unlink
      3. existing.session_id == our session_id       → unlink (mine via
         agent identity — primary path for hook-to-hook clears)
      4. existing.task_dir == expected_task_dir      → unlink (mine via
         supervisor claim — primary path for batch between-ops, where
         the supervisor itself isn't inside a Claude session)
      5. existing.task_dir dangling                  → unlink
      6. heartbeat stale on existing.task_dir        → unlink
      7. otherwise                                   → refuse (live
         session, different identity, no supervisor claim)
    """
    rec = _load_active_record()
    if rec is None:
        return True

    if force:
        _try_unlink_active()
        return True

    our_session = _our_session_id()
    if our_session and rec["session_id"] == our_session:
        _try_unlink_active()
        return True

    if (expected_task_dir and rec["task_dir"]
            and os.path.abspath(rec["task_dir"])
                 == os.path.abspath(expected_task_dir)):
        _try_unlink_active()
        return True

    if rec["task_dir"] and not os.path.isdir(rec["task_dir"]):
        _try_unlink_active()
        return True

    if rec["task_dir"] and not _heartbeat_fresh(rec["task_dir"]):
        _try_unlink_active()
        return True

    age = (_heartbeat_age_seconds(rec["task_dir"])
           if rec["task_dir"] else 0.0)
    print(f"[state_store] WARNING: refusing to clear .active_task — "
          f"held by session_id={rec['session_id'] or '<legacy>'} on "
          f"{rec['task_dir']} (heartbeat {age:.0f}s ago, fresh). Our "
          f"session_id={our_session or '<none>'}, expected_task_dir="
          f"{expected_task_dir!r}. Pass force=True only if you've "
          f"verified that session is truly done.", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Per-task transaction id (the .ar_state-wide invariant marker)
# ---------------------------------------------------------------------------
# A single monotonically-increasing integer that tags each "write a
# coordinated group of .ar_state files" operation. Each writer in the
# group embeds the txn id in its file; the LAST step writes
# `.ar_state/.txn` containing the id + a "by" label naming the
# operation. A reader inspecting the group can now answer "is this
# state self-consistent?" by a direct compare — no more stitching the
# answer together from heartbeat/pid/string heuristics.
#
# Convention:
#   begin_txn(task_dir, by) -> id
#       Allocates next id (read .txn, +1). NO disk write yet — the
#       writer is responsible for putting `id` into every body file
#       it then writes; commit_txn lands the marker.
#   commit_txn(task_dir, id, by)
#       Atomically writes .ar_state/.txn = {id, by, at}.
#       MUST be the last step of the transaction. If the writer
#       crashes before commit, the body files carry id but .txn
#       still names id-1 → check_txn_consistency reports `extra`
#       and tooling can identify which writer was in flight.
#   read_txn(task_dir) -> {"id": int, "by": str, "at": str}
#       Reads the committed-txn marker; returns id=0 for a fresh
#       task with no .txn yet.
#   check_txn_consistency(task_dir) -> report dict
#       Inspects progress.json / history.jsonl tail / plan.md
#       header / .phase / .pending_settle.json for embedded txn
#       ids, returns {consistent, current_txn, files, stale, extra}.
#       Used by post_bash activation to surface partial commits.

def read_txn(task_dir: str) -> dict:
    """Current committed transaction marker. Returns id=0 when no .txn
    file exists (fresh task or pre-txn-system state)."""
    path = state_path(task_dir, TXN_FILE)
    if not os.path.exists(path):
        return {"id": 0, "by": "", "at": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "id":  int(data.get("id") or 0),
            "by":  str(data.get("by", "") or ""),
            "at":  str(data.get("at", "") or ""),
        }
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[state_store] WARNING: {path} corrupt ({e}); treating "
              f"as id=0. Re-run the writer or delete the file to "
              f"recover.", file=sys.stderr)
        return {"id": 0, "by": "", "at": ""}


def begin_txn(task_dir: str) -> int:
    """Reserve the next txn id. Single-process convention: the writer
    that calls begin_txn is responsible for tagging every file it
    then writes with the returned id, and for calling commit_txn
    after the last body write. NO inter-process locking — concurrent
    writers from different processes are out of scope (the framework
    already routes everything through PhaseController + record_round
    + create_plan, each of which runs in one process at a time).
    The `by` label only lands at commit time."""
    return read_txn(task_dir)["id"] + 1


def commit_txn(task_dir: str, txn_id: int, by: str) -> None:
    """Atomically commit the transaction marker. Should be the LAST
    write of the transaction; body files that carry this id and
    landed BEFORE the commit are the durable record of what was
    committed."""
    from datetime import datetime, timezone
    path = state_path(task_dir, TXN_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"id": int(txn_id), "by": by,
           "at": datetime.now(timezone.utc).isoformat()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


def _txn_from_progress(task_dir: str) -> Optional[int]:
    """Extract `_txn_id` from progress.json, or None when missing /
    pre-txn-system / file corrupt."""
    path = progress_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("_txn_id")
        return int(v) if isinstance(v, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _txn_from_history_tail(task_dir: str) -> Optional[int]:
    """Extract `_txn_id` from the LAST history.jsonl row, or None.
    history.jsonl is append-only and per-line JSON, so the tail is
    the most-recent write."""
    path = history_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-1, os.SEEK_END)
                while f.tell() > 0:
                    if f.read(1) == b"\n":
                        if f.tell() == os.path.getsize(path):
                            f.seek(-2, os.SEEK_END)
                            continue
                        break
                    f.seek(-2, os.SEEK_CUR)
                last = f.readline().decode("utf-8", errors="replace").strip()
            except OSError:
                # Empty file
                return None
        if not last:
            return None
        row = json.loads(last)
        v = row.get("_txn_id")
        return int(v) if isinstance(v, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _txn_from_plan(task_dir: str) -> Optional[int]:
    """Extract txn from `# Plan vN  <!-- txn: M -->` first line.
    Returns None for pre-txn-system plan.md (just `# Plan vN`)."""
    path = plan_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
    except OSError:
        return None
    import re as _re
    m = _re.search(r"<!--\s*txn:\s*(\d+)\s*-->", first)
    return int(m.group(1)) if m else None


def _txn_from_phase(task_dir: str) -> Optional[int]:
    """Extract txn from `.phase` content `PHASE|N`. Pre-txn-system
    .phase files just hold the phase name with no `|` — returns None."""
    path = state_path(task_dir, PHASE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            content = f.read().strip()
    except OSError:
        return None
    if "|" not in content:
        return None
    try:
        return int(content.split("|", 1)[1].strip())
    except (ValueError, IndexError):
        return None


def _txn_from_pending_settle(task_dir: str) -> Optional[int]:
    path = pending_settle_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("_txn_id")
        return int(v) if isinstance(v, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def check_txn_consistency(task_dir: str) -> dict:
    """Compare per-file embedded txn ids against the committed marker.

    The only inconsistency is `extra` (file claims a newer txn than
    .txn says is committed) — that's the in-flight signature: body
    landed but the commit marker did not. Earlier-txn tags ("stale")
    are normal: a file that wasn't part of the latest transaction
    legitimately keeps the id of whichever earlier transaction last
    touched it (e.g. plan.md keeps its create_plan txn id across
    many subsequent rounds that only touch progress/history). The
    reader exposes the stale list for diagnostics but doesn't treat
    it as an inconsistency.

    Files with txn=None (pre-txn-system data on disk) contribute no
    constraint — surfaced under `untagged` for the operator's
    information, doesn't affect `consistent`.
    """
    current = read_txn(task_dir)["id"]
    files = {
        "progress.json":          _txn_from_progress(task_dir),
        "history.jsonl":          _txn_from_history_tail(task_dir),
        "plan.md":                _txn_from_plan(task_dir),
        ".phase":                 _txn_from_phase(task_dir),
        ".pending_settle.json":   _txn_from_pending_settle(task_dir),
    }
    extra = [name for name, t in files.items()
             if isinstance(t, int) and t > current]
    stale = [name for name, t in files.items()
             if isinstance(t, int) and t < current]
    untagged = [name for name, t in files.items() if t is None
                and os.path.exists(_path_for(task_dir, name))]
    return {
        "consistent": not extra,
        "current_txn": current,
        "by": read_txn(task_dir)["by"],
        "files": files,
        "extra": extra,
        "stale": stale,        # informational only
        "untagged": untagged,  # informational only
    }


def _path_for(task_dir: str, file_label: str) -> str:
    """Map check_txn_consistency's label back to its on-disk path —
    used only to filter the `untagged` list to files that actually
    exist."""
    mapping = {
        "progress.json":        progress_path(task_dir),
        "history.jsonl":        history_path(task_dir),
        "plan.md":              plan_path(task_dir),
        ".phase":               state_path(task_dir, PHASE_FILE),
        ".pending_settle.json": pending_settle_path(task_dir),
    }
    return mapping.get(file_label, "")


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
    rec = _load_active_record()
    if rec is not None:
        td = rec["task_dir"]
        if td and os.path.isdir(td):
            return td
        # Stale pointer (dangling or empty) — clean it so future
        # callers skip step 1.
        _try_unlink_active()

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

    Tolerates two on-disk forms:
      - `PHASE`             (legacy / no txn tag)
      - `PHASE|<txn_id>`    (txn-tagged — strip the suffix)

    If the file exists but the (stripped) content is unrecognised
    (truncated mid-write, hand-edited typo, out-of-date phase name
    from a downgrade), emit a stderr warning AND fall back to INIT —
    silent fallback was hiding mid-write corruption from the operator
    until resume.py also rejected it with a confusing "incompatible
    state". The warning makes the corrupt-state path visible without
    crashing every hook on the way to recovery.
    """
    path = state_path(task_dir, PHASE_FILE)
    if not os.path.exists(path):
        return INIT
    with open(path, "r") as f:
        content = f.read().strip()
    phase = content.split("|", 1)[0].strip() if "|" in content else content
    if phase in ALL_PHASES:
        return phase
    import sys as _sys
    print(f"[state_store] WARNING: {path} contains unrecognised phase "
          f"{content!r}; treating as INIT. If a hook or pipeline "
          f"crashed mid-write, restoring from .ar_state/history.jsonl "
          f"or deleting {path} and re-running /autoresearch may "
          f"recover.", file=_sys.stderr)
    return INIT


def write_phase(task_dir: str, phase: str, *, txn_id: int):
    """Write phase to .ar_state/.phase atomically. On-disk content is
    `PHASE|<txn_id>` so check_txn_consistency can extract the tag
    without a separate sidecar.

    tmp + os.replace prevents read_phase from ever seeing an empty /
    half-written file, which previously cascaded into the INIT-
    fallback path above and gated the agent out of all AR scripts.
    PhaseController owns the only callsite, so there is no cross-
    writer race — atomicity is purely about crash-mid-write.
    """
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    path = state_path(task_dir, PHASE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(f"{phase}|{int(txn_id)}")
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
                  *, stamp: bool = True, txn_id: int):
    """Write progress to .ar_state/progress.json atomically. Accepts
    Progress or a plain dict (the dict path stays for batch/manifest.py
    which has its own schema and predates the dataclass).

    `txn_id` lands as the `_txn_id` field in the payload;
    check_txn_consistency reads it back to verify this write belongs
    to the currently-committed transaction (or is "extra" if a crash
    left it ahead). Required — every save_progress must participate
    in a transaction so the cross-file consistency invariant holds.

    Atomicity: tmp + os.replace. Earlier non-atomic rewrites
    occasionally let `load_progress` see an empty file mid-write and
    `compute_next_phase` then short-circuit to FINISH well before
    max_rounds.
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
    payload["_txn_id"] = int(txn_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(payload), f, indent=2)
    os.replace(tmp_path, path)


def append_history(task_dir: str, record: dict, *, txn_id: int):
    """Append one JSON record to history.jsonl. `txn_id` lands as a
    `_txn_id` field on the row; check_txn_consistency reads the LAST
    row's value to verify the most-recent append belongs to the
    committed transaction. Required — every history write must
    participate in a transaction."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {**record, "_txn_id": int(txn_id)}
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
    # update_progress is a standalone helper (hook reset paths,
    # DIAGNOSE counter increments) — not part of a coordinated outer
    # transaction. Allocate a micro-txn so the resulting progress.json
    # is still tagged consistently with .txn.
    txn = begin_txn(task_dir)
    try:
        save_progress(task_dir, new_progress, stamp=False, txn_id=txn)
    except Exception as e:
        import sys as _sys
        print(f"[state_store] CRITICAL: failed to save progress.json for "
              f"{task_dir}: {type(e).__name__}: {e}. fields={list(fields)}. "
              f"The in-memory update is lost; the next round may see stale "
              f"state (wrong consecutive_failures, diagnose_attempts, etc). "
              f"Free disk space / fix permissions and re-run the failed "
              f"action.", file=_sys.stderr)
        raise
    commit_txn(task_dir, txn, by="update_progress")
    return new_progress


# Subprocess output parser lives in utils.json_io now (was duplicated
# here and in task_config.eval_client). Importers that previously did
# `from phase_machine import parse_last_json_line` should switch to
# `from utils.json_io import parse_last_json_line`.
