"""Round-0 SEED eval recorder. `run_baseline_init(task_dir, eval_data)`
is called in-process by engine/baseline.py and returns that script's
exit code (see `_EXIT_FOR`). Owns the post-baseline phase transition
via PhaseController.on_baseline_settled — the post-Bash hook only
emits guidance off the phase already on disk."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, load_progress, load_state, save_state,
)
from task_config import EvalOutcome, load_task_config  # noqa: E402
from utils.git_utils import current_head_short  # noqa: E402

from .progress_reducer import reduce_baseline_init
from .transition import PhaseController


# Outcome → exit code. Binary: 0 = task is activatable (kernel may need
# rewrite via PLAN, but state machine handles that), non-zero = task is
# NOT activatable. scaffold.py's "rc != 0 → surface error" stays accurate;
# the slash command's "non-zero exit → stop and report" gates only on
# INFRA_FAIL now. The full 3-way outcome lives in progress.baseline_outcome
# for downstream readers (post_bash, dashboard, stop_save).
_EXIT_FOR = {
    EvalOutcome.OK: 0,
    EvalOutcome.KERNEL_FAIL: 0,
    EvalOutcome.INFRA_FAIL: 4,
}


def run_baseline_init(task_dir: str, eval_data: dict) -> int:
    """Library entry point. engine/baseline.py calls this after
    run_eval finishes; the return value becomes that script's exit
    code. Side effects (progress, history, phase) are durable on disk
    before this returns."""
    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        return 1

    existing = load_progress(task_dir) or Progress()
    head_commit = current_head_short(task_dir) or "unknown"
    reduction = reduce_baseline_init(
        existing, config, eval_data, best_commit=head_commit)

    if reduction.dropped_seed_metric is not None:
        print(f"[baseline] dropping wrong-output seed timing "
              f"(latency_us={reduction.dropped_seed_metric:.1f}) — "
              f"kernel failed correctness "
              f"so its measurement cannot anchor best_metric.",
              file=sys.stderr)
    if reduction.anchor.message:
        print(f"[baseline] {reduction.anchor.message}", file=sys.stderr)

    # Append SEED history row first (durable artifact).
    append_history(task_dir, {
        "round": 0,
        "description": "seed kernel initial eval",
        "decision": "SEED",
        "metrics": reduction.metrics,
        "outcome": reduction.outcome.value,
        "correctness": reduction.correctness,
        "commit": head_commit,
    })

    # Single atomic commit: merge progress fields + bump
    # expected_history_round so the consistency check matches the row
    # we just appended. progress_initialized flips on here — this is the
    # discriminator load_progress() uses to distinguish "claimed by a
    # session but never measured" from "has baseline data". Resume /
    # dashboard rely on it to avoid offering a Round 0/0 view on a task
    # that hasn't run baseline yet.
    state = load_state(task_dir) or {}
    for k, v in reduction.progress.to_dict().items():
        state[k] = v
    state["expected_history_round"] = 0
    state["progress_initialized"] = True
    save_state(task_dir, state)

    # Phase transition (PLAN for kernel_fail, untouched for infra_fail)
    # is owned by on_baseline_settled.
    if reduction.outcome != EvalOutcome.OK:
        PhaseController(task_dir).on_baseline_settled()
        print(f"[baseline] {reduction.outcome.value}: "
              f"{eval_data.get('error') or '(no detail)'}",
              file=sys.stderr)
        return _EXIT_FOR[reduction.outcome]

    if reduction.seed_metric is None:
        # Degenerate: outcome=OK but no primary metric. Leave phase at
        # BASELINE so the agent retries.
        print(f"[baseline] ERROR: outcome=OK but no valid "
              f"{config.primary_metric}; treating as kernel-no-timing.",
              file=sys.stderr)
        return 2

    PhaseController(task_dir).on_baseline_settled()
    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={reduction.seed_metric}, "
          f"baseline({reduction.anchor.source})={reduction.anchor.metric}, "
          f"commit={head_commit}", file=sys.stderr)
    return 0
