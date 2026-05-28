#!/usr/bin/env python3
"""ar_cli — AutoResearch CLI entry.

Currently a single subcommand: `worker` (start / stop / status of the
local HTTP worker daemon). The `--remote-host <alias>` flag delegates
the same command to a remote machine via SSH and (for --start) sets up
a local ssh -L tunnel so subsequent local-port commands transparently
hit the remote daemon. Remote host config lives in
`autoresearch/config.yaml:remote_worker.hosts`.

Examples:

    # local daemon control
    ar_cli worker --start --backend ascend --arch ascend910b3 \\
        --devices 6 --port 9111 --bg
    ar_cli worker --status --port 9111
    ar_cli worker --stop --port 9111

    # remote (the host alias is an entry in config.yaml; both sides
    # need their own AscendOpGenAgent checkout — the remote `repo_path`
    # should point at the autoresearch/ subdir, not the outer repo)
    ar_cli worker --remote-host my-npu --start --backend ascend \\
        --arch ascend910b3 --devices 6 --port 9111
    ar_cli worker --remote-host my-npu --status --port 9111
    ar_cli worker --remote-host my-npu --stop --port 9111
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
# Autoresearch project root: scripts/ → autoresearch/
AR_ROOT = SCRIPT_DIR.parent

STATE_DIR = Path(os.environ.get(
    "AR_STATE_DIR", str(Path.home() / ".autoresearch_state")))


# ---------------------------------------------------------------------------
# Tunnel pid file helpers (used by --remote-host in Phase 2C)
# ---------------------------------------------------------------------------

def _tunnel_pid_path(port: int) -> Path:
    return STATE_DIR / "tunnels" / f"{port}.pid"


# ---------------------------------------------------------------------------
# Worker daemon — local
# ---------------------------------------------------------------------------

def _worker_log_path(port: int) -> str:
    return f"/tmp/ar_worker_{port}.log"


def _curl_status(host: str, port: int, timeout: float = 5.0) -> Optional[dict]:
    url = f"http://{host}:{port}/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, socket.timeout, ConnectionError, Exception):
        return None


def _find_pid_on_port(port: int) -> Optional[int]:
    """POSIX-only: returns the PID listening on TCP `port`, or None.

    Tries `ss -ltnp` first (modern), falls back to `lsof -iTCP:port`.
    Both produce process info; we parse the first numeric pid we see.
    """
    if os.name != "posix":
        return None

    # `ss -ltnp` output line example:
    #   LISTEN 0  2048  0.0.0.0:9111  0.0.0.0:*  users:(("python",pid=12345,fd=7))
    try:
        out = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in out.stdout.splitlines():
            if f":{port} " in ln or ln.rstrip().endswith(f":{port}"):
                # extract pid=NNN
                m = ln.split("pid=")
                if len(m) > 1:
                    pid_part = m[1].split(",")[0].split(")")[0]
                    return int(pid_part)
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
    except Exception:
        pass

    return None


def _cmdline_contains_worker(pid: int) -> bool:
    """Safety check before SIGTERM — only kill if /proc/<pid>/cmdline
    looks like our worker. POSIX-only."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
        return "worker.server" in cmd or "worker/server.py" in cmd
    except Exception:
        return False


def cmd_worker_start(args) -> int:
    env = os.environ.copy()
    env["WORKER_BACKEND"] = args.backend
    env["WORKER_ARCH"] = args.arch
    env["WORKER_DEVICES"] = args.devices
    env["WORKER_HOST"] = args.host
    env["WORKER_PORT"] = str(args.port)

    cmd = [sys.executable, "-m", "worker.server"]

    if args.bg:
        if os.name != "posix":
            print("[ar_cli] --bg requires POSIX (detached process group). "
                  "On Windows, run without --bg and redirect output.",
                  file=sys.stderr)
            return 2

        log_path = _worker_log_path(args.port)
        log_fh = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Poll /status until ready or timeout — gives a clear failure
        # message instead of "fork succeeded, daemon crashed silently".
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break  # process died during boot
            if _curl_status("127.0.0.1", args.port, timeout=1.5):
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            rc = proc.poll()
            tail = ""
            try:
                with open(log_path, "rb") as f:
                    tail = f.read()[-1500:].decode(errors="replace")
            except Exception:
                pass
            print(f"[ar_cli] worker on :{args.port} did not become ready "
                  f"within 20s (process rc={rc}). Log tail:\n{tail}",
                  file=sys.stderr)
            return 1

        print("=" * 56)
        print("AutoResearch Worker (daemon)")
        print("-" * 56)
        print(f"  Backend : {args.backend}")
        print(f"  Arch    : {args.arch}")
        print(f"  Devices : {args.devices}")
        print(f"  Host    : {args.host}")
        print(f"  Port    : {args.port}")
        print(f"  PID     : {proc.pid}")
        print(f"  Log     : {log_path}")
        print("-" * 56)
        print(f"  Stop: ar_cli worker --stop --port {args.port}")
        print("=" * 56)
        return 0

    # Foreground
    return subprocess.call(cmd, cwd=str(SCRIPT_DIR), env=env)


def cmd_worker_stop(args) -> int:
    if os.name != "posix":
        print("[ar_cli] --stop requires POSIX (uses ss/lsof + SIGTERM). "
              "On Windows, find the process manually.", file=sys.stderr)
        return 2

    pid = _find_pid_on_port(args.port)
    if pid is None:
        print(f"[ar_cli] no listener on :{args.port}", file=sys.stderr)
        return 1

    if not args.force and not _cmdline_contains_worker(pid):
        print(f"[ar_cli] pid {pid} on :{args.port} does not look like a "
              f"worker (cmdline missing 'worker.server'). Pass --force "
              f"to kill anyway.", file=sys.stderr)
        return 1

    try:
        # SIGTERM, wait briefly, SIGKILL if still alive.
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print(f"[ar_cli] stopped worker pid={pid} on :{args.port}")
                return 0
        os.kill(pid, signal.SIGKILL)
        print(f"[ar_cli] SIGKILLed unresponsive worker pid={pid} on :{args.port}",
              file=sys.stderr)
        return 0
    except ProcessLookupError:
        print(f"[ar_cli] pid {pid} already gone")
        return 0
    except PermissionError as e:
        print(f"[ar_cli] cannot signal pid {pid}: {e}", file=sys.stderr)
        return 1


def cmd_worker_status(args) -> int:
    st = _curl_status(args.host, args.port)
    if st is None:
        print(f"Worker on {args.host}:{args.port} unreachable.")
        return 1
    print(json.dumps(st, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ar_cli",
        description="AutoResearch CLI (worker daemon control).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser(
        "worker",
        help="Start / stop / check the AutoResearch worker daemon.",
        description=(
            "Start / stop / check the AutoResearch worker on this "
            "machine (or, with --remote-host, on a config-defined "
            "remote machine via SSH)."
        ),
    )
    action = w.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true",
                        help="Start the worker on this machine.")
    action.add_argument("--stop", action="store_true",
                        help="Stop the daemon listening on --port. "
                             "POSIX-only (uses ss/lsof + SIGTERM/SIGKILL).")
    action.add_argument("--status", action="store_true",
                        help="Curl /api/v1/status on --host:--port.")

    w.add_argument("--backend", choices=["ascend", "cuda", "cpu"],
                   help="Hardware backend (required for --start).")
    w.add_argument("--arch",
                   help="Arch string, e.g. ascend910b3 (required for --start).")
    w.add_argument("--devices",
                   help="Comma-separated device IDs, e.g. '2,5' "
                        "(required for --start).")

    w.add_argument("--host", default=None,
                   help="Bind / probe address. Default 0.0.0.0 for --start, "
                        "127.0.0.1 for --status / --stop.")
    w.add_argument("--port", type=int, default=9111,
                   help="TCP port (default: 9111).")
    w.add_argument("--bg", action="store_true",
                   help="Daemon mode for --start (detach, log to "
                        "/tmp/ar_worker_<port>.log). POSIX-only.")
    w.add_argument("--force", action="store_true",
                   help="For --stop: skip the worker-cmdline safety check.")

    w.add_argument("--remote-host", default=None,
                   help="Run the same worker command on a remote host "
                        "(SSH alias defined in autoresearch/config.yaml:"
                        "remote_worker.hosts). When --start, also opens "
                        "a local ssh -L tunnel so 127.0.0.1:<port> "
                        "forwards to the remote daemon.")
    w.set_defaults(func=_dispatch_worker)
    return ap


def _validate_worker_args(args) -> Optional[str]:
    if args.start and not (args.backend and args.arch and args.devices):
        return ("--start requires --backend, --arch, --devices "
                "(no auto-detect mode).")
    if args.host is None:
        args.host = "0.0.0.0" if args.start else "127.0.0.1"
    return None


def _dispatch_worker(args) -> int:
    err = _validate_worker_args(args)
    if err:
        print(f"[ar_cli] {err}", file=sys.stderr)
        return 2

    if args.remote_host:
        return _dispatch_remote_worker(args)

    if args.start:
        return cmd_worker_start(args)
    if args.stop:
        return cmd_worker_stop(args)
    if args.status:
        return cmd_worker_status(args)
    return 2


# ---------------------------------------------------------------------------
# Remote worker — SSH dispatch + ssh -L tunnel
# ---------------------------------------------------------------------------

def _load_remote_host_config(alias: str) -> Optional[dict]:
    """Look up alias under autoresearch/config.yaml :: remote_worker.hosts.
    Returns None if config or alias is missing — caller surfaces the error."""
    cfg_path = AR_ROOT / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ar_cli] failed to read {cfg_path}: {e}", file=sys.stderr)
        return None
    hosts = (((data.get("remote_worker") or {}).get("hosts") or {}))
    return hosts.get(alias)


def _build_remote_ar_cli_cmd(host_cfg: dict, ar_cli_args: list[str]) -> str:
    """Compose the bash command we send through ssh: source env, cd repo,
    invoke the remote ar_cli.py with the equivalent (non-remote) args.

    All values are shlex-quoted; the resulting string is passed to ssh
    AS A SINGLE ARG so the remote shell parses it as one command. The
    `--bg` flag is added unconditionally for --start so the daemon
    detaches from the ssh session before we tear it down.
    """
    python = host_cfg.get("python") or "python"
    repo_path = host_cfg["repo_path"]  # required; KeyError surfaces cleanly
    env_script = host_cfg.get("env_script")

    parts: list[str] = []
    if env_script:
        parts.append(f"source {shlex.quote(env_script)}")
    parts.append(f"cd {shlex.quote(repo_path)}")
    parts.append(
        f"{shlex.quote(python)} scripts/ar_cli.py "
        + " ".join(shlex.quote(a) for a in ar_cli_args)
    )
    return " && ".join(parts)


def _strip_remote_flags(args) -> list[str]:
    """Reconstruct the non-remote ar_cli worker args for the remote side.
    Mirrors the parser flags exactly so the remote ar_cli runs the same
    code path as if the user typed it directly there."""
    out = ["worker"]
    if args.start:
        out.append("--start")
    elif args.stop:
        out.append("--stop")
    elif args.status:
        out.append("--status")

    if args.backend:
        out += ["--backend", args.backend]
    if args.arch:
        out += ["--arch", args.arch]
    if args.devices:
        out += ["--devices", args.devices]
    # Always bind 0.0.0.0 on remote so the ssh -L on the loopback works
    # regardless of args.host (which on the dev side defaults to
    # 127.0.0.1 for status/stop — we don't want to push that to remote).
    if args.start:
        out += ["--host", "0.0.0.0", "--bg"]
    out += ["--port", str(args.port)]
    if args.force:
        out.append("--force")
    return out


def _tunnel_start(host: str, port: int) -> int:
    """Start `ssh -L <port>:127.0.0.1:<port> <host> -N -f`. The forked
    ssh writes its PID to <STATE_DIR>/tunnels/<port>.pid (via `-o
    PidFile=`-equivalent — actually ssh has no such option, we shell out
    and grep for the pid via pgrep after fork). Returns the pid on
    success, 0 on failure."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "tunnels").mkdir(exist_ok=True)
    pid_path = _tunnel_pid_path(port)

    # If a stale tunnel exists, tear it down first so the new one binds.
    _tunnel_stop_silent(port)

    # `-f` forks; `-N` no remote command; `-T` no pty; `ExitOnForwardFailure=yes`
    # so we fail fast if the LocalForward bind clashes.
    cmd = [
        "ssh", "-f", "-N", "-T",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=60",
        "-L", f"{port}:127.0.0.1:{port}",
        host,
    ]
    try:
        rc = subprocess.call(cmd)
    except Exception as e:
        print(f"[ar_cli] ssh tunnel launch failed: {e}", file=sys.stderr)
        return 0
    if rc != 0:
        print(f"[ar_cli] ssh tunnel exited rc={rc} (could not establish "
              f"LocalForward {port}:127.0.0.1:{port}).", file=sys.stderr)
        return 0

    # ssh -f forks itself; find the child by `pgrep -f "ssh.*-L <port>:.*<host>"`.
    # POSIX only; on Windows the ssh client doesn't fork — there `-f` is a no-op
    # and the call above would have blocked.
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"ssh.*-L {port}:.*{host}"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                pid = int(ln)
                pid_path.write_text(str(pid))
                return pid
    except Exception:
        pass

    # Couldn't find the pid but the forward did establish — print a hint.
    print(f"[ar_cli] tunnel established but pid not captured; manual "
          f"cleanup with `pkill -f 'ssh.*-L {port}:.*{host}'` if needed.",
          file=sys.stderr)
    return 0


def _tunnel_stop_silent(port: int) -> None:
    pid_path = _tunnel_pid_path(port)
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, ValueError, FileNotFoundError):
        pass
    except Exception as e:
        print(f"[ar_cli] tunnel stop failed: {e}", file=sys.stderr)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def _dispatch_remote_worker(args) -> int:
    host_cfg = _load_remote_host_config(args.remote_host)
    if host_cfg is None:
        print(f"[ar_cli] no remote_worker.hosts.{args.remote_host} entry "
              f"in autoresearch/config.yaml.", file=sys.stderr)
        return 2
    if "repo_path" not in host_cfg:
        print(f"[ar_cli] remote_worker.hosts.{args.remote_host} missing "
              f"`repo_path`.", file=sys.stderr)
        return 2

    remote_args = _strip_remote_flags(args)
    remote_cmd = _build_remote_ar_cli_cmd(host_cfg, remote_args)
    ssh_alias = host_cfg.get("ssh_alias") or args.remote_host

    print(f"[ar_cli] remote ({ssh_alias}): {remote_cmd}", file=sys.stderr)
    rc = subprocess.call(["ssh", ssh_alias, "bash", "-lc", remote_cmd])
    if rc != 0:
        print(f"[ar_cli] remote ar_cli exited rc={rc}", file=sys.stderr)
        return rc

    # Tunnel management is paired with daemon lifecycle.
    if args.start:
        pid = _tunnel_start(ssh_alias, args.port)
        if pid:
            print(f"[ar_cli] ssh -L 127.0.0.1:{args.port} -> "
                  f"{ssh_alias}:{args.port} (tunnel pid={pid})")
        # Verify the tunneled endpoint actually answers.
        st = _curl_status("127.0.0.1", args.port)
        if st is None:
            print(f"[ar_cli] tunneled status probe failed; remote daemon may "
                  f"not be ready or tunnel didn't bind.", file=sys.stderr)
            return 1
        print(json.dumps(st, indent=2))
        return 0

    if args.stop:
        _tunnel_stop_silent(args.port)
        print(f"[ar_cli] tore down local tunnel for :{args.port}")
        return 0

    if args.status:
        # Hit the local tunnel (assumes --start set it up).
        st = _curl_status("127.0.0.1", args.port)
        if st is None:
            print(f"Worker tunnel 127.0.0.1:{args.port} unreachable "
                  f"(--start may not have been called, or tunnel died).")
            return 1
        print(json.dumps(st, indent=2))
        return 0

    return 2


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
