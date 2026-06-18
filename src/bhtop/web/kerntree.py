"""
kerntree — shared, engine-agnostic kernel-tree helpers: a hierarchical (folder) listing that
replaces the old flat per-engine file list, plus new-folder / duplicate-folder / rename / delete.

PURE filesystem (no device, no FastAPI). Reuses labkit.safe_path for the traversal guard on
files; folders get an equivalent commonpath guard here (safe_path also enforces an editable
extension, which a directory has not). Hidden from the tree: kernel.json sidecars, *.orig
backups, and per-kernel build/ artifact dirs.
"""
import os
import shutil
from collections import Counter

from . import labkit
from . import kernmeta

HIDDEN_DIRS = {"build", "__pycache__", ".git"}


def gather_metal_sources(exdir, exts):
    """Collect a tt-metal example/project dir's device-kernel texts + concatenated host text:
      device = sources under a kernels/ dir, keyed by basename — or by path-relative-to-exdir when
               a basename collides across nested sub-examples, so NONE is dropped;
      host   = the remaining editable sources (the .cpp driver that sets the args).
    Returns (device:{key:text}, host_text:str). Pure filesystem; read errors are skipped."""
    seg = f"{os.sep}kernels{os.sep}"
    pend, host = [], []
    for dp, _, names in os.walk(exdir):
        for n in sorted(names):
            if os.path.splitext(n)[1] not in exts:
                continue
            full = os.path.join(dp, n)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if seg in full:
                pend.append((os.path.relpath(full, exdir), n, text))
            else:
                host.append(text)
    dup = Counter(base for _, base, _ in pend)
    device = {(rel if dup[base] > 1 else base): text for rel, base, text in pend}
    return device, "\n".join(host)


def _safe_dir(root, rel):
    """Resolve a workspace-relative DIRECTORY path under root, refusing traversal. Unlike
    labkit.safe_path this allows a path with no editable extension (a folder)."""
    if not root:
        raise ValueError("workspace root not found")
    rroot = os.path.realpath(root)
    full = os.path.realpath(os.path.join(rroot, rel or ""))
    if full != rroot and os.path.commonpath([full, rroot]) != rroot:
        raise ValueError(f"path escapes the workspace: {rel}")
    return full


def _node_file(root, full, exts, lang_of, role_of):
    rel = os.path.relpath(full, root)
    name = os.path.basename(full)
    lang = lang_of(name)
    return {"type": "file", "key": rel, "name": name,
            "lang": lang, "role": role_of(rel) if role_of else lang}


def _walk(root, cur, exts, lang_of, role_of, is_kernel):
    """Recursively build the tree under `cur` (absolute). Dirs first, then files, alpha."""
    dirs, files = [], []
    try:
        entries = sorted(os.listdir(cur))
    except OSError:
        return []
    for n in entries:
        full = os.path.join(cur, n)
        if os.path.isdir(full):
            if n in HIDDEN_DIRS:
                continue
            rel = os.path.relpath(full, root)
            kids = _walk(root, full, exts, lang_of, role_of, is_kernel)
            kern = is_kernel(rel) if is_kernel else os.path.isfile(os.path.join(full, kernmeta.META_NAME))
            dirs.append({"type": "dir", "key": rel, "name": n, "kernel": kern, "children": kids})
        elif os.path.isfile(full):
            if n == kernmeta.META_NAME or n.endswith(".orig"):
                continue
            if os.path.splitext(n)[1] in exts:
                files.append(_node_file(root, full, exts, lang_of, role_of))
    return dirs + files


def list_tree(root, exts, lang_of, role_of=None, is_kernel=None):
    """Nested listing: {available, root, tree:[ {type:'dir',name,key,kernel,children} |
    {type:'file',key,name,lang,role} ]}. `is_kernel(dir_rel)->bool` overrides the kernel flag
    (default = the dir holds a kernel.json). `available` is False when root is missing."""
    if not root or not os.path.isdir(root):
        return {"available": False, "root": root, "tree": []}
    return {"available": True, "root": root,
            "tree": _walk(root, root, exts, lang_of, role_of, is_kernel)}


# ---- folder operations (all guarded by _safe_dir) -----------------------------------
def folder_new(root, rel):
    full = _safe_dir(root, rel)
    if os.path.exists(full):
        raise ValueError(f"already exists: {rel}")
    os.makedirs(full)
    return os.path.relpath(full, root)


def folder_dup(root, src_rel, dst_rel):
    src = _safe_dir(root, src_rel)
    dst = _safe_dir(root, dst_rel)
    if not os.path.isdir(src):
        raise ValueError(f"not a folder: {src_rel}")
    if os.path.exists(dst):
        raise ValueError(f"already exists: {dst_rel}")
    # copy the kernel (sources + kernel.json) but not stale build artifacts / backups
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build", "*.orig", "__pycache__"))
    return os.path.relpath(dst, root)


def folder_rename(root, src_rel, dst_rel):
    src = _safe_dir(root, src_rel)
    dst = _safe_dir(root, dst_rel)
    if not os.path.isdir(src):
        raise ValueError(f"not a folder: {src_rel}")
    if os.path.exists(dst):
        raise ValueError(f"already exists: {dst_rel}")
    os.rename(src, dst)
    return os.path.relpath(dst, root)


def folder_delete(root, rel):
    full = _safe_dir(root, rel)
    if full == os.path.realpath(root):
        raise ValueError("refusing to delete the workspace root")
    if not os.path.isdir(full):
        raise ValueError(f"not a folder: {rel}")
    shutil.rmtree(full)
    return {"ok": True, "deleted": rel}


# ---- kernel resolution --------------------------------------------------------------
def kernel_dir_for(root, file_rel):
    """The kernel folder that governs a file: the nearest ancestor (within root) holding a
    kernel.json, else the file's own directory. Returns an ABSOLUTE path."""
    rroot = os.path.realpath(root)
    d = os.path.dirname(os.path.realpath(os.path.join(rroot, file_rel)))
    while True:
        if os.path.isfile(os.path.join(d, kernmeta.META_NAME)):
            return d
        if os.path.realpath(d) == rroot or os.path.commonpath([d, rroot]) != rroot:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.realpath(os.path.join(rroot, file_rel)))
