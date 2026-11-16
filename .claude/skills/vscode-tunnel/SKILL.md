---
name: vscode-tunnel
description: Connect to a remote VS Code Tunnel (`code tunnel` / `https://vscode.dev/tunnel/<name>`) from the command line to list tunnels, execute shell commands with live stdout/stderr streaming, or read remote files over MessagePack-RPC. Triggers when asked to "access vscode tunnel", "connect to tunnel <name>", or run commands on another cluster via a `vscode.dev/tunnel/...` link. Contains zero hardcoded credentials, usernames, or hostnames.
user_invocable: true
---

# VS Code Tunnel CLI Connector

Access and run commands on a remote host or cluster exposed via `code tunnel`
(`https://vscode.dev/tunnel/<name>`) without opening a browser or storing
credentials in the repository.

## Security & Hygiene Guarantees

1. **Zero secrets in Git**: Neither this document nor
   [`scripts/vscode_tunnel_exec.py`](scripts/vscode_tunnel_exec.py) contains
   any tokens, GitHub usernames, tunnel IDs, cluster IDs, or private IPs.
2. **Runtime credential discovery**: The script discovers a valid GitHub token
   at runtime from environment variables (`VSCODE_TUNNEL_GITHUB_TOKEN`,
   `GH_TOKEN`, `GITHUB_TOKEN`), `gh auth token`, or `~/.config/gh/hosts.yml`
   (probing configured accounts against the DevTunnels API and skipping expired
   entries automatically).
3. **No token in process argv**: When invoking `devtunnel connect`, the
   short-lived tunnel `connect` token is piped via `stdin` (`-t -`), never
   passed on the command line where `ps` or `/proc/<pid>/cmdline` could expose
   it on a shared host.
4. **Clean working tree**: If the `devtunnel` CLI binary is not already
   installed, it is downloaded to `~/.local/bin/devtunnel` outside the Git
   repository so `git status` remains clean.

## Usage

### 1. List available tunnels

```bash
python3 .claude/skills/vscode-tunnel/scripts/vscode_tunnel_exec.py list
```

### 2. Run a command on a remote tunnel

Pass either the tunnel name (e.g. `online_rl_abl`), the full URL
(`https://vscode.dev/tunnel/online_rl_abl`), or an explicit
`<tunnelId>.<clusterId>`:

```bash
python3 .claude/skills/vscode-tunnel/scripts/vscode_tunnel_exec.py exec \
  --tunnel online_rl_abl \
  --cmd 'hostname && rocm-smi --showuse'
```

Options for `exec`:
- `--tunnel <name|url|id.cluster>` (required): Target tunnel identifier.
- `--cmd <shell_command>` (required, or pass `-` to read script from `stdin`).
- `--cwd <remote_dir>`: Working directory on the remote host (default: `/home/aiscuser`).
- `--login`: Run `/bin/bash -lc` instead of `/bin/bash -c` (useful when login profile variables are needed).
- `--timeout <seconds>`: Socket timeout in seconds (default: `3600`).

### 3. Read a file from a remote tunnel

```bash
python3 .claude/skills/vscode-tunnel/scripts/vscode_tunnel_exec.py read \
  --tunnel online_rl_abl \
  --path /etc/hosts
```

## How It Works

1. Queries `https://global.rel.tunnels.api.visualstudio.com/api/v1/tunnels?global=true`
   and resolves the tunnel whose `name` or `tags` matches `--tunnel` (preferring
   active tunnels with `hostConnectionCount > 0`).
2. Spawns an independent `~/.local/bin/devtunnel connect <tunnelId>.<clusterId> -t -`
   forwarder per invocation and parses the dynamically assigned local forwarding
   port (`Forwarding from 127.0.0.1:<local_port> to host port 31545.`), so
   multiple processes can access the same or different tunnels concurrently
   without lock contention or port collisions.
3. Speaks VS Code CLI `protocolv6` MessagePack-RPC (`spawn` / `fs_read`) over
   `127.0.0.1:<local_port>`, demultiplexing `stdout` and `stderr` streams in
   real time and propagating the remote exit code.
4. Terminates its own local `devtunnel` forwarder on exit.
