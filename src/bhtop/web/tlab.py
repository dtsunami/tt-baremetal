"""
tlab — filesystem half of the Tensix Compute Lab: edit the device kernels of the prebuilt
compute programming_examples. Like nlab's device kernels, the **compute + dataflow** kernels
are JIT-compiled by tt-metal at run time, so an edit takes effect on the next Run (no rebuild).
The **host** .cpp is a real binary and would need a rebuild — shown read-tagged.

FILESYSTEM ONLY (no device); runs off the device thread. Editing snapshots a one-time .orig.
"""
import os
import shutil

from . import labkit
from .. import metal

EDIT_EXT = {".cpp", ".hpp", ".h", ".cc"}


def examples_root():
    h = metal.metal_home()
    if not h:
        return None
    d = os.path.join(h, "tt_metal", "programming_examples")
    return d if os.path.isdir(d) else None


def _ex_dir(example):
    """Map a binary name ('metal_example_matmul_single_core') to its source dir. Most are
    top-level ('add_2_integers_in_compute/') but some are nested ('matmul/matmul_single_core/'),
    so fall back to searching for a same-named dir that has a kernels/ subdir."""
    root = examples_root()
    if not root:
        return None
    name = example.replace("metal_example_", "")
    top = os.path.join(root, name)
    if os.path.isdir(top):
        return top
    for dp, dirs, _ in os.walk(root):
        if os.path.basename(dp) == name and "kernels" in dirs:
            return dp
    return None


def _role(rel):
    """compute / dataflow device kernels (JIT — edit+Run) vs host program (needs rebuild)."""
    p = f"/{rel}"
    if "/kernels/compute/" in p:
        return "compute"
    if "/kernels/" in p:
        return "dataflow"
    return "host"


def files(example):
    """The editable sources of one compute example, compute kernels first."""
    root, d = examples_root(), _ex_dir(example)
    if not root or not d:
        return []
    out = []
    for dp, _, names in os.walk(d):
        for n in sorted(names):
            if os.path.splitext(n)[1] in EDIT_EXT:
                full = os.path.join(dp, n)
                rel = os.path.relpath(full, root)
                out.append({"path": rel, "name": n, "role": _role(rel), "bytes": os.path.getsize(full)})
    out.sort(key=lambda f: ({"compute": 0, "dataflow": 1, "host": 2}[f["role"]], f["path"]))
    return out


def read_file(rel):
    full = labkit.safe_path(examples_root(), rel, EDIT_EXT)
    with open(full, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    return {"path": rel, "role": _role(rel), "content": content,
            "has_backup": os.path.exists(full + ".orig")}


def write_file(rel, content):
    """Persist an edit; snapshot the shipped file to <file>.orig on first write."""
    full = labkit.safe_path(examples_root(), rel, EDIT_EXT)
    if not os.path.exists(full):
        raise ValueError(f"no such file: {rel}")
    if not os.path.exists(full + ".orig"):
        shutil.copy2(full, full + ".orig")
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    return {"ok": True, "path": rel, "role": _role(rel), "bytes": len(content.encode())}


def copy_file(src_rel, dst_name):
    """Duplicate a kernel into a new file in the same dir (a fresh editable variation).
    Note: the example's host loads fixed kernel paths, so the copy is a scratch variant —
    edit the original in place to change what Run executes (Revert restores it)."""
    root = examples_root()
    src = labkit.safe_path(root, src_rel, EDIT_EXT)
    dst_rel = os.path.join(os.path.dirname(src_rel), os.path.basename(dst_name))
    dst = labkit.safe_path(root, dst_rel, EDIT_EXT)
    if os.path.exists(dst):
        raise ValueError(f"{os.path.basename(dst_name)} already exists")
    shutil.copy2(src, dst)
    return read_file(dst_rel)


def revert_file(rel):
    full = labkit.safe_path(examples_root(), rel, EDIT_EXT)
    if not os.path.exists(full + ".orig"):
        return {"ok": False, "error": "no backup to revert to"}
    shutil.copy2(full + ".orig", full)
    return read_file(rel)
