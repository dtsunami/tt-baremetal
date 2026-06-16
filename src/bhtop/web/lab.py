"""
Kernel Lab — author, build and debug Blackhole kernels next to their live NoC footprint.

This module is FILESYSTEM + SUBPROCESS only; it never touches the device (that stays
behind DeviceManager's single worker thread). It exposes:

  * the editable kernel "projects" under the tt-metal data_movement tree
    (each project = a host test_*.cpp + a kernels/ dir of device kernels),
  * read/write of those sources (allowlisted, with a one-time .orig backup),
  * an incremental `ninja` rebuild of unit_tests_data_movement (host edits only —
    device kernels are JIT-compiled by tt-metal at run time, no rebuild needed),
  * a small curated doc set (NoC counters, dataflow API, the 3-hop design, uarch
    diagrams) so the reference lives next to the editor.

The run + per-NoC footprint path is unchanged — it lives in metal.py / DeviceManager.
"""
import json
import os
import re
import shutil
import subprocess

from . import lab_docs
from . import labkit
from .. import metal

EDIT_EXT = {".cpp", ".hpp", ".h", ".cc"}
DESIGN_JSON = os.path.expanduser("~/bhtop/scripts/kernel_design.json")
UARCH_DIR = os.path.expanduser("~/blackhole/uarch")


# ---- paths ----------------------------------------------------------------
def dm_root():
    """tt-metal data_movement test tree, or None if tt-metal isn't present."""
    h = metal.metal_home()
    if not h:
        return None
    d = os.path.join(h, "tests/tt_metal/tt_metal/data_movement")
    return d if os.path.isdir(d) else None


def build_dir():
    """The configured ninja build dir (matches the binary metal.binary() runs)."""
    h = metal.metal_home()
    if not h:
        return None
    for sub in ("build_Release", "build"):
        if os.path.exists(os.path.join(h, sub, "build.ninja")):
            return sub
    return None


def _safe(rel):
    """Resolve a project-relative path inside dm_root with an editable extension (shared
    labkit.safe_path; raises ValueError on traversal / bad type)."""
    root = dm_root()
    if not root:
        raise ValueError("tt-metal data_movement tree not found")
    return labkit.safe_path(root, rel, EDIT_EXT)


def _role(rel):
    """device = JIT-compiled by tt-metal at run (no rebuild); host = needs ninja."""
    return "device" if "/kernels/" in f"/{rel}" else "host"


# ---- projects + files -----------------------------------------------------
def projects():
    """Discover editable kernel projects (a dir with a kernels/ subdir + a host test)."""
    root = dm_root()
    if not root:
        return {"available": False, "root": None, "default": None, "projects": []}
    found = []
    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)
        if os.path.isdir(os.path.join(pdir, "kernels")):
            host = [f for f in os.listdir(pdir) if f.startswith("test_") and f.endswith(".cpp")]
            found.append({"name": name, "host": host[0] if host else None})
    default = "gather_scatter_3hop" if any(p["name"] == "gather_scatter_3hop" for p in found) else (
        found[0]["name"] if found else None)
    return {"available": True, "root": root, "default": default, "projects": found}


def files(project):
    """All editable sources for one project, host first then device kernels."""
    root = dm_root()
    if not root:
        return []
    pdir = os.path.join(root, project)
    out = []
    for dirpath, _, names in os.walk(pdir):
        for n in sorted(names):
            if os.path.splitext(n)[1] not in EDIT_EXT:
                continue
            full = os.path.join(dirpath, n)
            rel = os.path.relpath(full, root)
            out.append({"path": rel, "name": n, "role": _role(rel),
                        "bytes": os.path.getsize(full)})
    out.sort(key=lambda f: (f["role"] != "host", f["path"]))
    return out


def read_file(rel):
    full = _safe(rel)
    with open(full, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    return {"path": rel, "role": _role(rel), "content": content,
            "has_backup": os.path.exists(full + ".orig")}


def write_file(rel, content):
    """Persist an edit. Snapshots the as-shipped file to <file>.orig on first write."""
    full = _safe(rel)
    if not os.path.exists(full):
        raise ValueError(f"no such file: {rel}")
    backup = full + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(full, backup)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    return {"ok": True, "path": rel, "role": _role(rel), "bytes": len(content.encode())}


def copy_file(src_rel, dst_name):
    """Duplicate a source into a new file in the same dir (a fresh editable variation)."""
    src = _safe(src_rel)
    dst_rel = os.path.join(os.path.dirname(src_rel), os.path.basename(dst_name))
    dst = _safe(dst_rel)
    if os.path.exists(dst):
        raise ValueError(f"{os.path.basename(dst_name)} already exists")
    shutil.copy2(src, dst)
    return read_file(dst_rel)


def revert_file(rel):
    """Restore the .orig snapshot if present."""
    full = _safe(rel)
    backup = full + ".orig"
    if not os.path.exists(backup):
        return {"ok": False, "error": "no backup to revert to"}
    shutil.copy2(backup, full)
    return read_file(rel)


# ---- incremental build ----------------------------------------------------
def build(target="unit_tests_data_movement", timeout=2400):
    """Incremental ninja rebuild of the host test binary. Device kernels do NOT need
    this (tt-metal JIT-compiles them at run time); only host-side edits do."""
    h = metal.metal_home()
    bd = build_dir()
    if not h or not bd:
        return {"ok": False, "error": "tt-metal build dir not found (set TT_METAL_HOME)"}
    cmd = ["ninja", "-C", bd, target]
    try:
        r = subprocess.run(cmd, cwd=h, env=metal._env(), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"build timed out after {timeout}s", "target": target}
    log = (r.stdout or "") + (r.stderr or "")
    errors = labkit.parse_compiler_errors(log)
    no_work = "ninja: no work to do." in log
    return {"ok": r.returncode == 0, "returncode": r.returncode, "target": target,
            "cmd": " ".join(cmd), "no_work": no_work, "errors": errors,
            "log_tail": "\n".join(log.splitlines()[-200:])}


# ---- docs -----------------------------------------------------------------
def docs_index():
    """List the curated reference docs available in the right-hand pane."""
    idx = [{"id": d["id"], "title": d["title"], "kind": "md"} for d in lab_docs.DOCS]
    if os.path.exists(DESIGN_JSON):
        idx.append({"id": "design", "title": "3-hop kernel — design notes", "kind": "md"})
    for img in _uarch_images():
        idx.append({"id": f"uarch/{img}", "title": f"uarch · {img}", "kind": "img"})
    return idx


def doc(doc_id):
    for d in lab_docs.DOCS:
        if d["id"] == doc_id:
            return {"id": doc_id, "title": d["title"], "kind": "md", "markdown": d["body"]}
    if doc_id == "design":
        return {"id": "design", "title": "3-hop kernel — design notes", "kind": "md",
                "markdown": _design_markdown()}
    raise ValueError(f"unknown doc: {doc_id}")


def _design_markdown():
    try:
        with open(DESIGN_JSON) as fh:
            data = json.load(fh)
        md = data.get("result", {}).get("kernel")
        return md or "_design notes present but empty_"
    except Exception as e:
        return f"_could not read design notes: {e}_"


def _uarch_images():
    if not os.path.isdir(UARCH_DIR):
        return []
    return sorted(f for f in os.listdir(UARCH_DIR)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg")))


def uarch_path(name):
    """Resolve a uarch image name to an absolute path (allowlisted to UARCH_DIR)."""
    if name not in _uarch_images():
        return None
    return os.path.join(UARCH_DIR, name)
