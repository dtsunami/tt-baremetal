"""
labkit — small shared helpers for the lab backends (nlab Kernel Lab, xlab Hart Lab, …).

Extracted from patterns that were duplicated across web/lab.py, web/l2lab.py and the
two telemetry paths in web/device.py. PURE (no device, no FastAPI) so it's unit-testable
in isolation; the labs adopt these incrementally behind the unchanged DeviceManager core.

  * parse_compiler_errors  — GNU-style `file:line:col: error: msg` → [{file,line,col,msg}]
                             (was lab._ERR_RE + l2lab._ERR_RE; serves gcc/rustc/ninja/JIT)
  * safe_path              — sandbox a workspace-relative path (was lab._safe + l2lab._safe)
  * Broadcaster            — fan-out telemetry channel (was _clients/_broadcast/subscribe ×2)
"""
import asyncio
import os
import re

# GNU-style diagnostics (gcc / rustc / ninja / tt-metal JIT all emit this shape)
_ERR_RE = re.compile(r"([\w./+\-]+):(\d+):(\d+):\s*(?:fatal\s+)?error:\s*(.*)", re.I)


def parse_compiler_errors(log):
    """Scrape `file:line:col: error: msg` lines into [{file, line, col, msg}]. Returns the
    file path exactly as matched; callers that want just the basename apply it themselves
    (nlab shows the tree-relative path; xlab basenames its TempDir file)."""
    return [{"file": m.group(1), "line": int(m.group(2)),
             "col": int(m.group(3)), "msg": m.group(4).strip()}
            for m in _ERR_RE.finditer(log or "")]


def safe_path(root, rel, exts):
    """Resolve a workspace-relative path under `root`, refusing directory traversal and
    non-allowlisted extensions. Raises ValueError (callers map to HTTP 400). Mirrors the
    old lab._safe / l2lab._safe (commonpath guard works for nested *and* flat workspaces)."""
    if not root:
        raise ValueError("workspace root not found")
    rroot = os.path.realpath(root)
    full = os.path.realpath(os.path.join(rroot, rel))
    if os.path.commonpath([full, rroot]) != rroot:
        raise ValueError(f"path escapes the workspace: {rel}")
    if os.path.splitext(full)[1] not in exts:
        raise ValueError(f"not an editable source file: {rel}")
    return full


class Broadcaster:
    """Fan-out channel for a telemetry stream: each subscriber gets an asyncio.Queue(maxsize=4);
    broadcast() does put_nowait and silently drops on a full queue (a slow client never blocks
    the producer). Replaces the duplicated _clients/_broadcast/subscribe in DeviceManager.

    `seed` (optional) is a 0-arg callable returning a snapshot frame handed to each NEW
    subscriber (the NoC stream seeds with the last frame; the L2 stream doesn't)."""

    def __init__(self, seed=None):
        self._clients = set()
        self._seed = seed

    async def subscribe(self):
        q = asyncio.Queue(maxsize=4)
        if self._seed is not None:
            snap = self._seed()
            if snap is not None:
                q.put_nowait(snap)
        self._clients.add(q)
        return q

    def unsubscribe(self, q):
        self._clients.discard(q)

    def broadcast(self, frame):
        for q in list(self._clients):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    def __len__(self):
        return len(self._clients)
