"""AutoResearch worker — single HTTP endpoint that runs eval locally.

Endpoints:
    GET  /api/v1/status   server-side hardware info (no auth, no body)
    POST /api/v1/run      verify + profile in one round trip

The client ships a tar.gz built by `task_config.package_builder`. We
`safe_extract` it into a tempdir, pick a device slot from an asyncio
queue, and hand the extracted dir to `utils.eval_runner.local_eval` —
the exact code path direct local eval uses. The (verify_resp,
profile_resp) tuple it returns is JSON-serialised back to the client,
which feeds them to `task_config.eval_assemble.assemble_eval_result`.

Device assignment is internal to /run, so clients cannot leak a slot
(no acquire/release endpoints).

Non-finite floats (inf / -inf / nan) from a 0us-latency kernel or a
crashed-profile parse are recursively rewritten to `null` before
serialisation — FastAPI's JSON encoder rejects them with HTTP 500
otherwise.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

# worker/server.py → autoresearch/scripts/ is one parent up.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from task_config import load_task_config  # noqa: E402
from utils.eval_runner import local_eval_async  # noqa: E402
from utils.json_io import sanitize_floats as _sanitize_floats  # noqa: E402
from utils.settings import (  # noqa: E402
    worker_port as _worker_port,
    default_eval_timeout as _default_eval_timeout,
)

logger = logging.getLogger(__name__)

_state: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_devices(s: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        logger.warning("WORKER_DEVICES=%r unparseable; defaulting to [0]", s)
        return [0]


def _safe_extract_tar(tar_bytes: bytes, dest_dir: str) -> None:
    """Reject path-traversal entries (members whose resolved path escapes
    `dest_dir`) before unpacking. Refuses symlinks / devices outright."""
    import io
    dest_abs = os.path.realpath(dest_dir)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        for m in tar.getmembers():
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                raise ValueError(f"unsafe tar member type: {m.name}")
            target = os.path.realpath(os.path.join(dest_abs, m.name))
            if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
                raise ValueError(f"path traversal blocked: {m.name}")
        tar.extractall(dest_abs)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.environ.get("WORKER_BACKEND", "ascend")
    arch = os.environ.get("WORKER_ARCH", "")
    devices = _parse_devices(os.environ.get("WORKER_DEVICES", "0"))
    q: asyncio.Queue = asyncio.Queue()
    for d in devices:
        q.put_nowait(d)
    _state.update(backend=backend, arch=arch, devices=devices, queue=q)
    logger.info("worker ready: backend=%s arch=%s devices=%s",
                backend, arch, devices)
    yield
    logger.info("worker shutting down")


app = FastAPI(title="AutoResearch Worker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/status")
async def status():
    if "queue" not in _state:
        return {"status": "initializing"}
    return {
        "status": "ready",
        "backend": _state["backend"],
        "arch": _state["arch"],
        "devices": _state["devices"],
        "free": _state["queue"].qsize(),
    }


@app.post("/api/v1/run")
async def run(
    request: Request,
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(_default_eval_timeout()),
    override_base_us: Optional[float] = Form(None),
    override_base_per_shape_us: Optional[str] = Form(None),
):
    """Verify + profile in one call.

    Device is picked server-side from the queue, used as the DEVICE_ID
    for the eval subprocess, and released in the finally block — clients
    cannot leak a slot.

    The eval runs as an asyncio task; concurrently we watch
    `request.is_disconnected()`. Whichever finishes first wins:

      - eval done   → cancel the watcher, return result
      - client gone → cancel the eval (cascades into a SIGTERM on the
                      eval subprocess group via
                      `utils.eval_runner._run_subprocess_async`), release
                      the device, and return HTTP 499. This is what
                      keeps a `claude --print` killed by its wall-clock
                      cap from leaving an orphan eval pinning the device
                      until the eval finishes naturally.
    """
    if "queue" not in _state:
        raise HTTPException(status_code=503, detail="worker not initialised")

    package_bytes = await package.read()

    per_shape: Optional[list] = None
    if override_base_per_shape_us:
        try:
            per_shape = json.loads(override_base_per_shape_us)
            if not (isinstance(per_shape, list) and per_shape):
                per_shape = None
        except Exception as e:
            logger.warning("[%s] bad override_base_per_shape_us JSON: %s",
                           task_id, e)
            per_shape = None

    device_id = await _state["queue"].get()
    try:
        logger.info("[%s] run %s on device %d", task_id, op_name, device_id)
        eval_task = asyncio.create_task(
            _run_eval_async(package_bytes, task_id, op_name, timeout,
                            device_id, override_base_us, per_shape))
        watch_task = asyncio.create_task(_watch_disconnect(request))
        done, _pending = await asyncio.wait(
            [eval_task, watch_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if eval_task in done and not eval_task.cancelled():
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch_task
            try:
                return _sanitize_floats(eval_task.result())
            except Exception as e:
                # Internal eval failure (malformed task.yaml, loader
                # ValueError, response-assembly error). Package it as a
                # structured error response instead of letting it bubble to
                # FastAPI as an opaque HTTP 500 the client can't interpret.
                logger.exception("[%s] eval task raised", task_id)
                return _sanitize_floats(
                    _error_response(device_id, f"worker eval crashed: {e}"))
        # Disconnect (or eval crashed early without a result).
        logger.info("[%s] client gone before eval finished; cancelling, "
                    "releasing device %d", task_id, device_id)
        eval_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await eval_task
        # 499 is the conventional non-RFC "client closed request" code
        # nginx popularised; FastAPI happily relays it.
        raise HTTPException(status_code=499, detail="client disconnected")
    finally:
        await _state["queue"].put(device_id)


async def _watch_disconnect(request: Request) -> None:
    """Block until starlette reports the HTTP client has disconnected.

    `request.is_disconnected()` reads from the starlette receive queue —
    no network traffic, no curl-style heartbeat. uvicorn puts a
    `{"type": "http.disconnect"}` message in that queue the moment the
    TCP socket closes; this coroutine just observes it.
    """
    while not await request.is_disconnected():
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _run_eval_async(package_bytes: bytes, task_id: str, op_name: str,
                          timeout: int, device_id: int,
                          override_base_us: Optional[float],
                          override_base_per_shape_us: Optional[list]
                          ) -> dict:
    """Extract the package, look up kernel/ref filenames from task.yaml,
    dispatch to `local_eval_async`. Returns the dict the client expects.

    Mirrors the sync `_run_eval_sync` this replaced; the difference is
    the eval is `await`-ed (so cancellation propagates into the eval
    subprocess group) instead of run via `asyncio.to_thread` (which
    can't be cancelled — `asyncio.to_thread` runs in a real OS thread
    that doesn't respond to task cancellation, so a SIGTERM'd `claude
    --print` would close its socket and the worker would carry on
    blocking on `subprocess.run` for the rest of the eval).
    """
    with tempfile.TemporaryDirectory(prefix=f"ar_run_{task_id}_") as tmp:
        try:
            _safe_extract_tar(package_bytes, tmp)
        except Exception as e:
            return _error_response(device_id, f"extract failed: {e}")

        config = load_task_config(tmp)
        if config is None:
            return _error_response(device_id,
                                   "task.yaml missing in package")

        kernel_file = (config.editable_files[0].replace(".py", "")
                       if config.editable_files else "kernel")
        ref_file = config.ref_file.replace(".py", "")

        verify_resp, profile_resp = await local_eval_async(
            task_dir=tmp,
            op_name=op_name,
            kernel_file=kernel_file,
            ref_file=ref_file,
            timeout=timeout,
            device_id=device_id,
            override_base_time_us=override_base_us,
            override_base_per_shape_us=override_base_per_shape_us,
        )
        return {
            "device_id": device_id,
            "verify_resp": verify_resp,
            "profile_resp": profile_resp,
        }


def _error_response(device_id: int, msg: str) -> dict:
    # error_source="infra": every caller here is a worker-side
    # infrastructure failure (tar extract, missing task.yaml, internal
    # eval crash) — NOT a kernel defect. assemble_eval_result maps this to
    # INFRA_FAIL so the round isn't charged to the kernel as KERNEL_FAIL.
    return {
        "device_id": device_id,
        "verify_resp": {
            "success": False, "log": msg, "returncode": 1,
            "error_source": "infra", "verify_block": {}, "artifacts": {},
        },
        "profile_resp": {
            "success": False, "log": "", "artifacts": {},
            "gen_time": None, "base_time": None,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def start_server(host: Optional[str] = None, port: Optional[int] = None):
    # SSH-only by design: bind loopback so the worker is reachable ONLY via
    # an ssh -L tunnel (which forwards to the remote's 127.0.0.1). Never
    # bind a public interface — that would expose the eval endpoint on
    # every network.
    host = host or os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(port or os.environ.get("WORKER_PORT", str(_worker_port())))
    logger.info("starting worker on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    start_server()
