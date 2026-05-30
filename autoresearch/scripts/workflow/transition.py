"""PhaseController — single owner of `.ar_state/.phase` writes. Callers
invoke `on_*` events; the controller decides the target phase + writes.
A new event must land here, not in the caller.

Transactional model: callers that already hold an outer txn (e.g.
record_round inside the round transaction, create_plan inside the
plan transaction) pass `txn_id=N` so the phase write inherits the
group's id. Standalone events (activation, baseline-settled, etc.)
auto-allocate a micro-txn (begin → write_phase → commit) so the
.phase file always carries a valid tag and check_txn_consistency
never sees an untagged .phase mid-flight."""
from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    BASELINE, EDIT, FINISH, PLAN,
    begin_txn, commit_txn,
    compute_next_phase, compute_resume_phase, load_progress, read_phase,
    write_phase,
)
from task_config.metric_policy import STUCK_BASELINE_OUTCOMES  # noqa: E402


class PhaseController:
    def __init__(self, task_dir: str, *, txn_id: Optional[int] = None):
        self.task_dir = task_dir
        # When set by the caller, _write uses this id (the phase write
        # participates in the caller's outer transaction). When None,
        # _write allocates a micro-txn for each event so .phase still
        # ends up tagged.
        self._outer_txn = txn_id

    # ---- Activation -----------------------------------------------------
    def on_activation_resume(self) -> str:
        phase = compute_resume_phase(self.task_dir)
        return self._write(phase)

    def on_activation_ready(self) -> str:
        return self._write(BASELINE)

    def on_baseline_settled(self) -> str:
        """ok / kernel_fail → PLAN; STUCK_BASELINE_OUTCOMES (infra_fail)
        → leave phase as-is. Missing outcome (legacy progress) is treated
        as kernel_fail so the agent gets pushed through PLAN."""
        progress = load_progress(self.task_dir)
        if progress is None:
            return read_phase(self.task_dir)
        outcome = progress.baseline_outcome or "kernel_fail"
        if outcome in STUCK_BASELINE_OUTCOMES:
            return read_phase(self.task_dir)
        return self._write(PLAN)

    def on_plan_validated(self) -> str:
        return self._write(EDIT)

    def on_round_settled(self) -> str:
        return self._write(compute_next_phase(self.task_dir))

    def _write(self, phase: str) -> str:
        if self._outer_txn is not None:
            write_phase(self.task_dir, phase, txn_id=self._outer_txn)
        else:
            txn = begin_txn(self.task_dir)
            write_phase(self.task_dir, phase, txn_id=txn)
            commit_txn(self.task_dir, txn,
                       by=f"PhaseController->{phase}")
        return phase

    # Re-export so callers don't need a separate `from phase_machine import FINISH`.
    FINISH = FINISH
