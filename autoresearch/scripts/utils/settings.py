"""Shared accessors for config.yaml.

config.yaml is the SINGLE SOURCE OF TRUTH for the framework reference
tables and tunable knobs below. This module reads it once per process and
exposes typed accessors. There are NO in-code defaults: a missing section
or key is a hard error, because config.yaml ships with every key present.
Retune by editing config.yaml — never by editing values here.
"""
from functools import lru_cache
import os
from typing import Dict

import yaml

# __file__ now lives in scripts/utils/; climb two levels to reach autoresearch/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and every accessor below depends on it."""
    return _load_yaml(_CONFIG_PATH)


def _get(section: str, key: str):
    """Read config.yaml[section][key]. config.yaml is the single source of
    truth, so a missing section/key raises — there is no in-code default."""
    sect = _raw().get(section)
    if not isinstance(sect, dict) or key not in sect:
        raise KeyError(
            f"{_CONFIG_PATH}: missing required key '{section}.{key}'")
    return sect[key]


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))


# --- task defaults -----------------------------------------------------
def default_max_rounds() -> int:
    """Default optimization-round budget when a task doesn't specify one.
    Single source for scaffold (new task.yaml) and loader (TaskConfig
    fallback) so the two cannot drift."""
    return _get("defaults", "max_rounds")


# --- eval timing measurement (read where the timing runs: on remote eval
#     that is the WORKER's config.yaml) ----------------------------------
def eval_warmup() -> int:
    return _get("eval", "warmup")


def eval_repeats() -> int:
    return _get("eval", "repeats")


# --- remote worker -----------------------------------------------------
def worker_port() -> int:
    """Worker TCP port. Single source for ar_cli (tunnel/status) and
    worker.server (bind) so the two cannot drift."""
    return _get("worker", "port")


# --- batch pre-flight verification timeouts (seconds) ------------------
def batch_tier1_timeout() -> int:
    return _get("batch", "tier1_timeout")


def batch_tier2_timeout() -> int:
    return _get("batch", "tier2_timeout")


# --- resume heartbeat freshness window (seconds) ----------------------
def heartbeat_fresh_seconds() -> int:
    return _get("resume", "heartbeat_fresh_seconds")


# --- speedup classification thresholds (x vs ref) ---------------------
def speedup_improved_above() -> float:
    return _get("metrics", "speedup_improved_above")


def speedup_regress_below() -> float:
    return _get("metrics", "speedup_regress_below")


def classify_speedup(v: float) -> str:
    """'improved' / 'on-par' / 'regress' per the configured thresholds.
    Single owner for the batch reporters (summarize, monitor)."""
    if v > speedup_improved_above():
        return "improved"
    if v < speedup_regress_below():
        return "regress"
    return "on-par"
