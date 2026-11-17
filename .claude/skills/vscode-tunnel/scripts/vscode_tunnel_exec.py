#!/usr/bin/env python3
"""Connect to a remote VS Code Tunnel (`code tunnel`) over DevTunnels + MessagePack-RPC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import msgpack
import yaml

GLOBAL_TUNNELS_URL = "https://global.rel.tunnels.api.visualstudio.com/api/v1/tunnels?global=true"
CLUSTER_TUNNEL_URL_TEMPLATE = (
    "https://{cluster}.rel.tunnels.api.visualstudio.com/api/v1/tunnels/{tunnel_id}"
    "?includePorts=true&tokenScopes=connect"
)
DEVTUNNEL_DOWNLOAD_URL = "https://aka.ms/TunnelsCliDownload/linux-x64"
CONTROL_PORT = 31545
_FORWARD_RE = re.compile(rf"Forwarding from 127\.0\.0\.1:(\d+) to host port {CONTROL_PORT}\.")
_LISTENING_RE = re.compile(r"PortForwardingService listening on 127\.0\.0\.1:(\d+)\.")
SYSTEM_TAGS = frozenset({"protocolv6", "vscode-server-launcher", "_flag4"})


@dataclass(frozen=True)
class TunnelTarget:
    tunnel_id: str
    cluster_id: str
    name: str
    host_connections: int
    last_connected: str
    connect_token: str


def _candidate_github_tokens() -> list[str]:
    tokens: list[str] = []
    for env_key in ("VSCODE_TUNNEL_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val and val not in tokens:
            tokens.append(val)

    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        cli_tok = proc.stdout.strip()
        if proc.returncode == 0 and cli_tok and cli_tok not in tokens:
            tokens.append(cli_tok)
    except Exception:
        pass

    hosts_yml = Path.home() / ".config/gh/hosts.yml"
    if hosts_yml.is_file():
        try:
            data = yaml.safe_load(hosts_yml.read_text(encoding="utf-8")) or {}
            gh_cfg = data.get("github.com") or {}
            top_tok = (gh_cfg.get("oauth_token") or "").strip()
            if top_tok and top_tok not in tokens:
                tokens.append(top_tok)
            for user_info in (gh_cfg.get("users") or {}).values():
                if isinstance(user_info, dict):
                    u_tok = (user_info.get("oauth_token") or "").strip()
                    if u_tok and u_tok not in tokens:
                        tokens.append(u_tok)
        except Exception:
            pass

    return tokens


def _api_get_json(url: str, github_token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"github {github_token}",
            "Accept": "application/json",
            "User-Agent": "vscode-tunnel-skill/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def _discover_working_token_and_tunnels() -> tuple[str, list[dict[str, Any]]]:
    candidates = _candidate_github_tokens()
    if not candidates:
        raise RuntimeError(
            "No GitHub token found. Set VSCODE_TUNNEL_GITHUB_TOKEN / GH_TOKEN or run `gh auth login`."
        )
    last_err: Exception | None = None
    best_empty: tuple[str, list[dict[str, Any]]] | None = None
    for tok in candidates:
        try:
            data = _api_get_json(GLOBAL_TUNNELS_URL, tok)
            tunnels = data.get("value", data) if isinstance(data, dict) else data
            if isinstance(tunnels, list):
                if tunnels:
                    return tok, tunnels
                if best_empty is None:
                    best_empty = (tok, tunnels)
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (401, 403):
                continue
            raise
        except Exception as exc:
            last_err = exc
    if best_empty is not None:
        return best_empty
    raise RuntimeError(f"All discovered GitHub tokens failed DevTunnels authentication: {last_err}")


def _extract_friendly_name(detail: dict[str, Any]) -> str:
    explicit = (detail.get("name") or "").strip()
    if explicit:
        return explicit
    for tag in detail.get("tags") or []:
        if isinstance(tag, str) and tag not in SYSTEM_TAGS:
            return tag
    return ""


def _fetch_tunnel_target(github_token: str, tunnel_id: str, cluster_id: str) -> TunnelTarget:
    url = CLUSTER_TUNNEL_URL_TEMPLATE.format(cluster=cluster_id, tunnel_id=tunnel_id)
    detail = _api_get_json(url, github_token)
    status = detail.get("status") or {}
    host_conn = int(status.get("hostConnectionCount") or 0)
    last_conn = str(status.get("lastHostConnectionTime") or "")
    connect_token = str((detail.get("accessTokens") or {}).get("connect") or "")
    return TunnelTarget(
        tunnel_id=tunnel_id,
        cluster_id=cluster_id,
        name=_extract_friendly_name(detail),
        host_connections=host_conn,
        last_connected=last_conn,
        connect_token=connect_token,
    )


def _normalize_tunnel_spec(spec: str) -> str:
    cleaned = spec.strip()
    match = re.search(r"vscode\.dev/tunnel/([^/?#]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned.strip("/")


def _resolve_tunnel(spec: str) -> TunnelTarget:
    target_key = _normalize_tunnel_spec(spec)
    github_token, summaries = _discover_working_token_and_tunnels()

    if "." in target_key:
        tid, cid = target_key.split(".", 1)
        target = _fetch_tunnel_target(github_token, tid, cid)
        if not target.connect_token:
            raise RuntimeError(f"No connect token returned for tunnel {target_key}")
        return target

    ordered = sorted(
        summaries,
        key=lambda item: int((item.get("status") or {}).get("hostConnectionCount") or 0),
        reverse=True,
    )
    matches: list[TunnelTarget] = []
    for item in ordered:
        tid = str(item.get("tunnelId") or "")
        cid = str(item.get("clusterId") or "")
        if not tid or not cid:
            continue
        target = _fetch_tunnel_target(github_token, tid, cid)
        if target.tunnel_id == target_key or target.name.lower() == target_key.lower():
            matches.append(target)
            if target.host_connections > 0 and target.connect_token:
                return target

    if matches:
        best = matches[0]
        if not best.connect_token:
            raise RuntimeError(f"Tunnel '{target_key}' matched {best.tunnel_id}.{best.cluster_id} but has no connect token")
        return best

    raise RuntimeError(f"Could not find a VS Code tunnel matching '{spec}'")


def _ensure_devtunnel_binary() -> Path:
    found = shutil.which("devtunnel")
    if found:
        return Path(found)
    local_bin = Path.home() / ".local/bin/devtunnel"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return local_bin
    local_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = local_bin.with_suffix(".tmp")
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "-o", str(tmp_bin), DEVTUNNEL_DOWNLOAD_URL],
        check=True,
    )
    tmp_bin.chmod(0o755)
    tmp_bin.replace(local_bin)
    return local_bin


def _wait_for_forwarded_socket(proc: subprocess.Popen[str], timeout_sec: float = 30.0) -> socket.socket:
    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    output_lines: list[str] = []
    local_port: int | None = None
    port_ready = False
    deadline = time.monotonic() + timeout_sec
    try:
        while not port_ready and time.monotonic() < deadline:
            if proc.poll() is not None:
                rem = proc.stdout.read() or ""
                if rem:
                    output_lines.append(rem)
                raise RuntimeError(
                    f"devtunnel exited prematurely (code {proc.returncode}):\n{''.join(output_lines)}"
                )
            for key, _ in sel.select(timeout=0.5):
                line = key.fileobj.readline()  # type: ignore[union-attr]
                if not line:
                    continue
                output_lines.append(line)
                match = _FORWARD_RE.search(line)
                if match:
                    local_port = int(match.group(1))
                    continue
                if local_port is not None:
                    listen_match = _LISTENING_RE.search(line)
                    if listen_match and int(listen_match.group(1)) == local_port:
                        port_ready = True
                        break
    finally:
        sel.close()

    if local_port is None:
        raise TimeoutError(
            f"Timed out waiting for devtunnel port forwarding to host port {CONTROL_PORT}:\n"
            f"{''.join(output_lines)}"
        )

    def _drain_stdout(stream: Any) -> None:
        try:
            for _ in stream:
                pass
        except Exception:
            pass

    threading.Thread(target=_drain_stdout, args=(proc.stdout,), daemon=True).start()

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"devtunnel exited prematurely (code {proc.returncode})")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("127.0.0.1", local_port))
            return sock
        except OSError:
            sock.close()
            time.sleep(0.1)
    raise TimeoutError(f"Timed out connecting to forwarded port 127.0.0.1:{local_port}")


def _with_control_socket(target: TunnelTarget, timeout_sec: float):
    class _Session:
        def __init__(self) -> None:
            self._proc: subprocess.Popen[str] | None = None
            self.sock: socket.socket | None = None

        def __enter__(self) -> socket.socket:
            devtunnel_bin = _ensure_devtunnel_binary()
            self._proc = subprocess.Popen(
                [
                    str(devtunnel_bin),
                    "connect",
                    f"{target.tunnel_id}.{target.cluster_id}",
                    "-t",
                    "-",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdin is not None
            self._proc.stdin.write(target.connect_token + "\n")
            self._proc.stdin.close()
            self.sock = _wait_for_forwarded_socket(self._proc)
            self.sock.settimeout(timeout_sec)
            return self.sock

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()

    return _Session()


def cmd_list() -> int:
    github_token, summaries = _discover_working_token_and_tunnels()
    rows: list[TunnelTarget] = []
    for item in summaries:
        tid = str(item.get("tunnelId") or "")
        cid = str(item.get("clusterId") or "")
        if not tid or not cid:
            continue
        try:
            rows.append(_fetch_tunnel_target(github_token, tid, cid))
        except Exception:
            continue

    rows.sort(key=lambda r: (r.host_connections, r.last_connected), reverse=True)
    print(f"{'NAME':<24} {'TUNNEL_ID.CLUSTER':<28} {'ONLINE':<8} {'LAST_HOST_CONNECTED'}")
    for row in rows:
        label = row.name or "<unnamed>"
        fqid = f"{row.tunnel_id}.{row.cluster_id}"
        online = "yes" if row.host_connections > 0 else "no"
        print(f"{label:<24} {fqid:<28} {online:<8} {row.last_connected}")
    return 0


def cmd_exec(
    *,
    tunnel: str,
    command: str,
    cwd: str,
    login_shell: bool,
    timeout_sec: float,
) -> int:
    target = _resolve_tunnel(tunnel)
    shell_flag = "-lc" if login_shell else "-c"
    wrapped_cmd = f'export PATH="$HOME/.local/bin:$PATH"; {command}'
    payload = {
        "id": 1,
        "method": "spawn",
        "params": {
            "command": "/bin/bash",
            "args": [shell_flag, wrapped_cmd],
            "cwd": cwd,
            "env": {},
        },
    }
    with _with_control_socket(target, timeout_sec) as sock:
        sock.sendall(msgpack.packb(payload))
        unpacker = msgpack.Unpacker(raw=False)
        stdout_stream: int | None = None
        stderr_stream: int | None = None
        exit_code: int | None = None
        while exit_code is None:
            chunk = sock.recv(65536)
            if not chunk:
                break
            unpacker.feed(chunk)
            for msg in unpacker:
                if not isinstance(msg, dict):
                    continue
                method = msg.get("method")
                params = msg.get("params") or {}
                if method == "streams_started" and params.get("for_request_id") == 1:
                    sids = params.get("stream_ids") or []
                    if len(sids) >= 3:
                        stdout_stream, stderr_stream = sids[1], sids[2]
                elif method == "stream_data":
                    sid = params.get("stream")
                    seg = params.get("segment")
                    data = seg if isinstance(seg, bytes) else str(seg).encode("utf-8")
                    if sid == stderr_stream:
                        sys.stderr.buffer.write(data)
                        sys.stderr.buffer.flush()
                    elif sid == stdout_stream:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                elif msg.get("id") == 1:
                    if msg.get("error"):
                        sys.stderr.write(f"RPC error: {msg['error']}\n")
                        return 1
                    res = msg.get("result") or {}
                    exit_code = int(res.get("exit_code", 0))
        return 0 if exit_code is None else exit_code


def cmd_read(*, tunnel: str, remote_path: str, output: str | None, timeout_sec: float) -> int:
    target = _resolve_tunnel(tunnel)
    payload = {
        "id": 1,
        "method": "fs_read",
        "params": {"path": remote_path},
    }
    chunks: list[bytes] = []
    with _with_control_socket(target, timeout_sec) as sock:
        sock.sendall(msgpack.packb(payload))
        unpacker = msgpack.Unpacker(raw=False)
        read_stream: int | None = None
        stream_done = False
        rpc_done = False
        while not (stream_done and rpc_done):
            chunk = sock.recv(65536)
            if not chunk:
                break
            unpacker.feed(chunk)
            for msg in unpacker:
                if not isinstance(msg, dict):
                    continue
                method = msg.get("method")
                params = msg.get("params") or {}
                if method == "streams_started" and params.get("for_request_id") == 1:
                    sids = params.get("stream_ids") or []
                    if sids:
                        read_stream = sids[0]
                elif method == "stream_data" and params.get("stream") == read_stream:
                    seg = params.get("segment")
                    chunks.append(seg if isinstance(seg, bytes) else str(seg).encode("utf-8"))
                elif method == "stream_ended" and params.get("stream") == read_stream:
                    stream_done = True
                elif msg.get("id") == 1:
                    if msg.get("error"):
                        sys.stderr.write(f"RPC error: {msg['error']}\n")
                        return 1
                    rpc_done = True
                    if read_stream is None:
                        stream_done = True

    content = b"".join(chunks)
    if output:
        Path(output).write_bytes(content)
    else:
        sys.stdout.buffer.write(content)
        sys.stdout.buffer.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("list", help="List VS Code tunnels under the authenticated GitHub account")

    exec_p = sub.add_parser("exec", help="Execute a shell command on a remote VS Code tunnel")
    exec_p.add_argument("--tunnel", required=True, help="Tunnel name, vscode.dev URL, or <id>.<cluster>")
    exec_p.add_argument("--cmd", required=True, help="Command to execute (or '-' to read from stdin)")
    exec_p.add_argument("--cwd", default="/home/aiscuser", help="Remote working directory")
    exec_p.add_argument("--login", action="store_true", help="Use login shell (/bin/bash -lc)")
    exec_p.add_argument("--timeout", type=float, default=3600.0, help="Timeout in seconds")

    read_p = sub.add_parser("read", help="Read a file from a remote VS Code tunnel")
    read_p.add_argument("--tunnel", required=True, help="Tunnel name, vscode.dev URL, or <id>.<cluster>")
    read_p.add_argument("--path", required=True, help="Absolute path on the remote host")
    read_p.add_argument("--output", help="Optional local file path to write to")
    read_p.add_argument("--timeout", type=float, default=300.0, help="Timeout in seconds")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.subcommand == "list":
        return cmd_list()
    if args.subcommand == "exec":
        cmd_text = sys.stdin.read() if args.cmd == "-" else args.cmd
        return cmd_exec(
            tunnel=args.tunnel,
            command=cmd_text,
            cwd=args.cwd,
            login_shell=args.login,
            timeout_sec=args.timeout,
        )
    if args.subcommand == "read":
        return cmd_read(
            tunnel=args.tunnel,
            remote_path=args.path,
            output=args.output,
            timeout_sec=args.timeout,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
