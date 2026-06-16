"""
tt-isa-documentation browser — pulls the Tenstorrent Blackhole ISA docs into the
Kernel Lab docs pane, live from GitHub, with assets left pointing at the repo.

The repo tree and individual markdown pages are fetched over HTTPS (stdlib urllib,
no extra deps) and cached on disk, so repeat views are instant and survive brief
network blips. Images and inter-doc links are NOT vendored: the page returns the
raw base URL and its repo directory, and the client resolves relative `![](...)`
and `[...](...md)` against them — so diagrams load straight from the ISA repo and
cross-links navigate within the pane.
"""
import json
import os
import time
import urllib.request

REPO = "tenstorrent/tt-isa-documentation"
REF = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{REF}/"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1"
GITHUB_BLOB = f"https://github.com/{REPO}/blob/{REF}/"
CACHE_DIR = os.path.expanduser("~/bhtop/.isa_cache")
TREE_TTL = 24 * 3600
DOC_TTL = 24 * 3600


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "bhtop-kernel-lab"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _cached(key, ttl, fetch):
    """Return (text, from_cache). Serve fresh cache; on fetch failure fall back to
    stale cache if any, else re-raise."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, key.replace("/", "__"))
    if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < ttl:
        with open(p, encoding="utf-8") as f:
            return f.read(), True
    try:
        data = fetch()
        with open(p, "w", encoding="utf-8") as f:
            f.write(data)
        return data, False
    except Exception:
        if os.path.exists(p):                       # network blip -> serve stale
            with open(p, encoding="utf-8") as f:
                return f.read(), True
        raise


def tree():
    """Nested tree of every .md doc in the repo: dirs (with children) then files."""
    raw, _ = _cached("__tree.json", TREE_TTL, lambda: _get(TREE_API))
    data = json.loads(raw)
    paths = sorted(e["path"] for e in data.get("tree", [])
                   if e.get("type") == "blob" and e["path"].endswith(".md"))
    return _nest(paths)


def _nest(paths):
    root = {}
    for p in paths:
        parts = p.split("/")
        node = root
        for d in parts[:-1]:
            node = node.setdefault(d, {})
        node.setdefault("__files__", []).append({"name": parts[-1], "path": p})
    return _to_list(root)


def _to_list(node):
    out = [{"type": "dir", "name": name, "children": _to_list(node[name])}
           for name in sorted(k for k in node if k != "__files__")]
    out += [{"type": "file", "name": f["name"], "path": f["path"]}
            for f in node.get("__files__", [])]
    return out


def doc(path):
    """Fetch one markdown page. Returns the body plus the bases the client needs to
    resolve its relative assets/links back to the repo."""
    if not path.endswith(".md") or ".." in path or path.startswith("/"):
        raise ValueError(f"bad isa doc path: {path}")
    md, from_cache = _cached(path, DOC_TTL, lambda: _get(RAW_BASE + path))
    return {"path": path, "markdown": md, "raw_base": RAW_BASE,
            "repo_dir": os.path.dirname(path), "cached": from_cache,
            "github": GITHUB_BLOB + path}
