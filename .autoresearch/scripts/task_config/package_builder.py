"""Build a tar.gz package for the remote worker.

The worker receives task.yaml + editable_files (kernel) + ref_file
(reference). The worker has its own copy of the AscendOpGenAgent
repo, so it doesn't need eval_kernel.py / utils / skill files shipped
across — only the per-task artifacts.

Returns raw bytes; the HTTP transport posts them as multipart/form-data.
"""
from __future__ import annotations

import io
import os
import tarfile
from typing import Iterable

from .loader import TaskConfig


def _add_file(tar: tarfile.TarFile, task_dir: str, name: str) -> None:
    """Add task_dir/name into the archive at top-level name. Silently
    skips if the file is missing — the worker's load_task_config /
    eval_kernel will surface the error with the right context."""
    src = os.path.join(task_dir, name)
    if not os.path.isfile(src):
        return
    tar.add(src, arcname=name)


def build_package(task_dir: str, config: TaskConfig,
                  extra_files: Iterable[str] = ()) -> bytes:
    """Pack task.yaml + editable + ref + optional extras into tar.gz.

    `extra_files` is a list of bare basenames inside task_dir (e.g.
    `["inputs.json"]`) that the ref may import / read at runtime.
    Falsey entries are ignored.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        _add_file(tar, task_dir, "task.yaml")
        for ef in config.editable_files or []:
            _add_file(tar, task_dir, ef)
        _add_file(tar, task_dir, config.ref_file or "reference.py")
        for ex in extra_files:
            if ex:
                _add_file(tar, task_dir, ex)
    return buf.getvalue()
