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
from . import kerntree
from . import kernconf
from . import kernmeta
from . import kernparse
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


def tree():
    """Nested folder listing of programming_examples (each example as a folder, compute/dataflow
    kernels nested) for the device-browser folder view. Edits stay in place (JIT reads these
    exact paths on the next Run). `available:false` when tt-metal isn't present."""
    return kerntree.list_tree(examples_root(), EDIT_EXT, lambda n: "cpp", _role,
                              is_kernel=lambda rel: "/" not in rel)   # each example = a kernel


# ---- per-kernel config (kernel.json overlay) + restore --------------------------------
def _example_rel(key):
    """The example directory governing a selected file `key` (the nearest ancestor that owns a
    kernels/ subdir, within examples_root), relative to root — so nested programming_examples
    (matmul/matmul_single_core, profiler/...) each key their OWN overlay instead of sharing the
    top folder's. Falls back to the first path segment when none is found."""
    root = examples_root()
    top = (key or "").split("/")[0]
    if not root or not top:
        return top
    rroot = os.path.realpath(root)
    d = os.path.dirname(os.path.realpath(os.path.join(rroot, key or "")))
    while os.path.commonpath([d, rroot]) == rroot and d != rroot:
        if os.path.isdir(os.path.join(d, "kernels")):
            return os.path.relpath(d, rroot)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return top


def params(key):
    """The kernel.json (params + defaults) for the example the selected file belongs to. Stored
    as a bhtop overlay (no source copy); synthesizes an empty default if absent."""
    kdir, rel = kernconf.overlay_kdir_rel("tensix", _example_rel(key))
    meta = kernmeta.load(kdir, sources=[os.path.basename(key)] if key else [], lang="cpp", engine="tensix")
    return {"kernel": rel, "entry": key, "meta": meta}


def config_get(key):
    kdir, rel = kernconf.overlay_kdir_rel("tensix", _example_rel(key))
    return {"kernel": rel, **kernconf.raw_get(kdir, [os.path.basename(key)] if key else [], "cpp", "tensix")}


def config_put(key, text):
    kdir, _ = kernconf.overlay_kdir_rel("tensix", _example_rel(key))
    return kernconf.raw_put(kdir, text)


def merge_params(key, dry_run=False):
    """Parse the example's tt-metal sources (compute/dataflow kernels + host .cpp) and merge the
    discovered params (rtarg / ctarg / define) into the bhtop overlay kernel.json. Idempotent;
    preserves edits. Returns {kernel, added:[names], count}."""
    root = examples_root()
    if not root:
        raise ValueError("tt-metal programming_examples not found")
    rel = _example_rel(key)
    kdir, _ = kernconf.overlay_kdir_rel("tensix", rel)
    device, host = kerntree.gather_metal_sources(os.path.join(root, rel), EDIT_EXT)
    meta = kernmeta.load(kdir, sources=[os.path.basename(key)] if key else [], lang="cpp", engine="tensix")
    added = kernparse.merge(meta, kernparse.parse_metal(device, host))
    if not dry_run:
        os.makedirs(kdir, exist_ok=True)
        kernmeta.save(kdir, meta)
    return {"kernel": rel, "added": [p["name"] for p in added], "count": len(added)}


def merge_all(dry_run=False):
    """Merge every programming_example that ships device kernels — each example keyed to its OWN
    overlay (nested sub-examples are resolved separately). Returns a per-example summary."""
    root = examples_root()
    if not root:
        return {"available": False, "root": None, "results": []}
    results = []
    for exdir in _example_dirs(root):
        rel = os.path.relpath(exdir, root)
        results.append(merge_params(os.path.join(rel, "x.cpp"), dry_run=dry_run))
    return {"available": True, "root": root, "results": results}


def _example_dirs(root):
    """Every directory that directly owns a kernels/ subdir (= a runnable example), nested or flat.
    Does not descend into kernels/ itself."""
    out = []
    for dp, dirs, _ in os.walk(root):
        if dp != root and os.path.basename(dp) in kerntree.HIDDEN_DIRS:
            dirs[:] = []
            continue
        if "kernels" in dirs:
            out.append(dp)
            dirs[:] = [d for d in dirs if d != "kernels"]
    return sorted(out)


def restore():
    """Revert every edited example source to its shipped original (.orig). Destructive: discards
    your in-place TENSIX edits and restores the pristine programming_examples sources."""
    root = examples_root()
    if not root:
        return {"ok": False, "error": "tt-metal not found"}
    n = 0
    for dp, _, names in os.walk(root):
        for f in names:
            if f.endswith(".orig"):
                live = os.path.join(dp, f[:-5])
                if os.path.isfile(live):
                    shutil.copy2(os.path.join(dp, f), live)
                    n += 1
    return {"ok": True, "reverted": n}


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
