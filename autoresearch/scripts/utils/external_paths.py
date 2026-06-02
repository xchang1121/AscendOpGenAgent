"""Single source for paths to content OUTSIDE autoresearch/.

The framework depends on sibling skill trees living next to autoresearch/
at the repo root:

    <repo_root>/
      autoresearch/        <- this package (scripts/ lives here)
      skills/triton/kernel-verifier/scripts/    verify.py, benchmark.py,
                                                validate_triton_impl.py
      skills/triton/latency-optimizer/references/   *.md perf-tuning docs

This module is the one place that encodes the `<repo>/autoresearch +
<repo>/skills` layout assumption — a tree move is a one-line fix here
instead of every consumer.
"""
import os

# external_paths.py → utils/ → scripts/ → autoresearch/ → repo root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

_SKILLS_ROOT = os.path.join(_REPO_ROOT, "skills")


def kernel_verifier_dir() -> str:
    """Dir holding the kernel-verifier skill scripts (verify.py,
    benchmark.py, validate_triton_impl.py)."""
    return os.path.join(_SKILLS_ROOT, "triton", "kernel-verifier", "scripts")


def latency_refs_dir() -> str:
    """Dir holding the latency-optimizer reference markdown."""
    return os.path.join(
        _SKILLS_ROOT, "triton", "latency-optimizer", "references")
