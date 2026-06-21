"""
l2lab — the FILESYSTEM + COMPILE half of the L2CPU cockpit (no device access).

Mirrors lab.py's split: anything that doesn't touch the chip lives here and runs off
the device thread, so editing/compiling never contends with live telemetry. Device ops
(bringup / load / telemetry / regs) live on DeviceManager's single worker thread.

A small **workspace** under ~/bhtop/l2cpu_kernels is seeded once from the bundled
examples, so you edit/create kernels freely without touching the shipped sources.
"""
import os
import re
import shutil
import struct
import tempfile

from . import labkit, kerntree, kernmeta, kernconf, kernparse
from ..l2cpu import toolchain, regmap

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../bhtop
EXAMPLES_DIR = os.path.join(PKG, "l2cpu", "examples")               # legacy flat sources (CLI/scripts)
KERN_CANON = os.path.join(PKG, "kernels", "x280")                   # canonical kernel folders (tracked)
WORKDIR = os.path.expanduser("~/bhtop/kernels/x280")                # working tree (gitignored, per-user)
EDIT_EXT = {".c", ".s", ".S", ".rs"}


# ---- workspace (a tree of per-kernel folders, seeded from the canonical kernels) -----
def _ensure_workspace():
    """Create the working tree and seed it from the canonical kernel folders. Additive:
    copies any canonical kernel folder not already present (so new bundled kernels show up)
    without touching the user's edits or user-created folders."""
    os.makedirs(WORKDIR, exist_ok=True)
    if os.path.isdir(KERN_CANON):
        for n in os.listdir(KERN_CANON):
            src = os.path.join(KERN_CANON, n)
            dst = os.path.join(WORKDIR, n)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    return WORKDIR


def tree():
    """Nested folder listing for the device-browser tree (replaces the flat files() list)."""
    return kerntree.list_tree(_ensure_workspace(), EDIT_EXT, toolchain.detect_lang)


def _safe(name):
    """Resolve a workspace-relative filename, refusing traversal + bad extensions
    (shared labkit.safe_path)."""
    return labkit.safe_path(_ensure_workspace(), name, EDIT_EXT)


def files():
    """Flat recursive list of workspace source files (key = workspace-relative path). Kept for
    back-compat / the /api/l2/files endpoint; the browser uses tree()."""
    root = _ensure_workspace()
    out = []
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in kerntree.HIDDEN_DIRS]
        for n in sorted(names):
            if os.path.splitext(n)[1] in EDIT_EXT:
                rel = os.path.relpath(os.path.join(dirpath, n), root)
                out.append({"name": rel, "key": rel, "lang": toolchain.detect_lang(n),
                            "bytes": os.path.getsize(os.path.join(dirpath, n))})
    out.sort(key=lambda f: f["key"])
    return out


# ---- per-kernel meta-params (kernel.json) + folder ops + regenerate ------------------
def _kernel_dir_and_srcs(key):
    root = _ensure_workspace()
    _safe(key)                                              # guard the key (traversal + ext)
    kdir = kerntree.kernel_dir_for(root, key)
    srcs = [n for n in sorted(os.listdir(kdir))
            if os.path.isfile(os.path.join(kdir, n)) and os.path.splitext(n)[1] in EDIT_EXT]
    return root, kdir, srcs


def kernel_meta(key):
    """The kernel.json (param schema + defaults) governing a selected file `key`, plus where
    its entry source lives. Synthesizes a default (deploy knobs only) when there's no sidecar."""
    root, kdir, srcs = _kernel_dir_and_srcs(key)
    lang = toolchain.detect_lang(srcs[0]) if srcs else "c"
    meta = kernmeta.load(kdir, sources=srcs, lang=lang)
    rel = os.path.relpath(kdir, root)
    entry = os.path.normpath(os.path.join(rel, (meta.get("sources") or srcs or [key])[0]))
    return {"kernel": rel, "entry": entry, "meta": meta}


def config_get(key):
    """Raw kernel.json text for the JSON editor (auto-creates a default if the folder has none)."""
    root, kdir, srcs = _kernel_dir_and_srcs(key)
    lang = toolchain.detect_lang(srcs[0]) if srcs else "c"
    return {"kernel": os.path.relpath(kdir, root), **kernconf.raw_get(kdir, srcs, lang, "x280")}


def config_put(key, text):
    """Write kernel.json from the editor (validates JSON + schema)."""
    root, kdir, srcs = _kernel_dir_and_srcs(key)
    return kernconf.raw_put(kdir, text)


def save_params(key, values):
    """Persist the user's chosen values as the kernel's new defaults (writes kernel.json in
    the working folder; per-user, the working tree is gitignored)."""
    root, kdir, srcs = _kernel_dir_and_srcs(key)
    if os.path.realpath(kdir) == os.path.realpath(root):
        raise ValueError("file is not inside a kernel folder; create one to save params")
    lang = toolchain.detect_lang(srcs[0]) if srcs else "c"
    meta = kernmeta.load(kdir, sources=srcs, lang=lang)
    for p in meta.get("params", []):
        if p["name"] in (values or {}):
            p["default"] = kernmeta.coerce(p, values[p["name"]])
    kernmeta.save(kdir, meta)
    return {"ok": True, "kernel": os.path.relpath(kdir, root)}


def _canon_defines():
    """Names never emitted as discovered `define` params: the canonical regmap (the harness map,
    injected from regmap.py) plus the reserved deploy-knob names (tile/hart/addr) — a `#define addr`
    would otherwise collide with the deploy 'addr' and shadow it in kernmeta.route()."""
    return set(regmap.harness_defines(regmap.CODE_ADDR)) | {p["name"] for p in kernmeta.DEFAULT_DEPLOY}


def _read_srcs(kdir, srcs):
    """{filename: text} for a kernel folder's source files (read errors skipped)."""
    out = {}
    for n in srcs:
        try:
            with open(os.path.join(kdir, n), encoding="utf-8", errors="replace") as fh:
                out[n] = fh.read()
        except OSError:
            pass
    return out


def merge_params(key, dry_run=False):
    """Parse the kernel's x280 source(s) and merge the discovered params (define + documented
    mailbox ops) into its kernel.json, populating the param schema without clobbering edits.
    Idempotent. Returns {kernel, added:[names], count}."""
    root, kdir, srcs = _kernel_dir_and_srcs(key)
    if os.path.realpath(kdir) == os.path.realpath(root):
        raise ValueError("file is not inside a kernel folder; create one to merge params")
    lang = toolchain.detect_lang(srcs[0]) if srcs else "c"
    meta = kernmeta.load(kdir, sources=srcs, lang=lang)
    discovered = kernparse.parse_x280(_read_srcs(kdir, srcs), skip_defines=_canon_defines())
    added = kernparse.merge(meta, discovered)
    if not dry_run:
        kernmeta.save(kdir, meta)
    return {"kernel": os.path.relpath(kdir, root), "added": [p["name"] for p in added],
            "count": len(added)}


def merge_all(root=None, dry_run=False):
    """Merge every kernel folder under `root` (default = the working tree). Pass the canonical
    KERN_CANON to populate the tracked, shipped sidecars. Returns a per-kernel summary list."""
    base = root or _ensure_workspace()
    skip = _canon_defines()
    results = []
    if not os.path.isdir(base):
        return {"available": False, "root": base, "results": []}
    for n in sorted(os.listdir(base)):
        kdir = os.path.join(base, n)
        if not os.path.isdir(kdir) or n in kerntree.HIDDEN_DIRS:
            continue
        srcs = [f for f in sorted(os.listdir(kdir))
                if os.path.isfile(os.path.join(kdir, f)) and os.path.splitext(f)[1] in EDIT_EXT]
        if not srcs:
            continue
        lang = toolchain.detect_lang(srcs[0])
        meta = kernmeta.load(kdir, sources=srcs, lang=lang)
        added = kernparse.merge(meta, kernparse.parse_x280(_read_srcs(kdir, srcs), skip_defines=skip))
        if added and not dry_run:
            kernmeta.save(kdir, meta)
        results.append({"kernel": n, "added": [p["name"] for p in added], "count": len(added)})
    return {"available": True, "root": base, "results": results}


def _sibling(src, name):
    """A new folder path: `name` as a sibling of `src` (a bare basename), or a full
    workspace-relative path if it contains a slash."""
    return name if "/" in name else os.path.join(os.path.dirname(src), name)


def folder_new(rel):
    return {"ok": True, "path": kerntree.folder_new(_ensure_workspace(), rel)}


def folder_dup(src, name):
    return {"ok": True, "path": kerntree.folder_dup(_ensure_workspace(), src, _sibling(src, name))}


def folder_rename(src, name):
    return {"ok": True, "path": kerntree.folder_rename(_ensure_workspace(), src, _sibling(src, name))}


def folder_delete(rel):
    return kerntree.folder_delete(_ensure_workspace(), rel)


def regenerate():
    """Re-seed bundled kernels: overwrite each canonical kernel folder into the working tree
    (restores pristine example sources + kernel.json), PRESERVING user-created folders."""
    root = _ensure_workspace()
    refreshed = []
    if os.path.isdir(KERN_CANON):
        for n in sorted(os.listdir(KERN_CANON)):
            src = os.path.join(KERN_CANON, n)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(root, n)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            refreshed.append(n)
    return {"ok": True, "refreshed": refreshed}


def read_file(name):
    full = _safe(name)
    with open(full, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    return {"name": name, "lang": toolchain.detect_lang(name), "content": content}


def write_file(name, content):
    full = _safe(name)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    return {"ok": True, "name": name, "bytes": len(content.encode())}


def new_file(name, lang="c"):
    """Create a starter kernel of the given language if it doesn't exist. `name` may be nested
    in a folder (e.g. 'myk/myk.c'); parent folders are created as needed."""
    if "." not in os.path.basename(name):
        name += {"c": ".c", "asm": ".s", "rust": ".rs"}.get(lang, ".c")
    full = _safe(name)
    if os.path.exists(full):
        raise ValueError(f"{name} already exists")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(_STARTERS.get(toolchain.detect_lang(name), _STARTERS["c"]))
    return read_file(name)


def delete_file(name):
    os.remove(_safe(name))
    return {"ok": True, "name": name}


def copy_file(src, dst_name):
    """Duplicate a workspace kernel into a new editable name (a fresh variation to deploy)."""
    s = _safe(src)
    if "." not in dst_name:
        dst_name += os.path.splitext(src)[1]
    d = _safe(dst_name)
    if os.path.exists(d):
        raise ValueError(f"{dst_name} already exists")
    shutil.copy2(s, d)
    return read_file(dst_name)


def rename_file(src, dst_name):
    """Rename a workspace kernel (private flat workspace, so this is always safe)."""
    s = _safe(src)
    if "." not in dst_name:
        dst_name += os.path.splitext(src)[1]
    d = _safe(dst_name)
    if os.path.exists(d):
        raise ValueError(f"{dst_name} already exists")
    os.rename(s, d)
    return read_file(dst_name)


# ---- compile (no device) ------------------------------------------------------------
def compile_kernel(content, lang, addr, defines=None):
    """Compile editor `content` to a flat image. Returns words/bytes + disasm on success,
    or parsed compiler errors on failure — never raises (so the UI can show them inline).
    `defines` (name->int|hex) are the kernel's define-kind params, injected over the map."""
    ext = {"c": ".c", "asm": ".s", "rust": ".rs"}.get(lang, ".c")
    # Kernels that use RVV intrinsics (riscv_vector.h, directly or via <rvv.h>) need the V ISA.
    march = "rv64gcv" if ("riscv_vector.h" in content or "<rvv.h>" in content) else "rv64gc"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "kernel" + ext)
        with open(src, "w") as fh:
            fh.write(content)
        try:
            words = toolchain.compile_source(src, base=addr, lang=lang, defines=defines, march=march)
        except toolchain.ToolError as e:
            msg = str(e)
            errs = [{**err, "file": os.path.basename(err["file"])}
                    for err in labkit.parse_compiler_errors(msg)]
            return {"ok": False, "error": msg, "errors": errs}
        try:
            dis = toolchain.disasm(src, base=addr, lang=lang, defines=defines, march=march)
        except Exception:
            dis = ""
        return {"ok": True, "words": words, "bytes": len(words) * 4, "addr": addr,
                "disasm": _trim_disasm(dis)}


def _trim_disasm(dis):
    """Keep just the instruction lines from objdump output (drop the file header)."""
    lines = [ln for ln in dis.splitlines()
             if re.match(r"\s+[0-9a-f]+:\s", ln) or re.match(r"[0-9a-f]+ <", ln)]
    return "\n".join(lines[:400])


def have_rust():
    return toolchain.have_rust()


# ---- static map (no device) ---------------------------------------------------------
def map_text():
    return regmap.render_map()


def examples():
    """The bundled (read-only) example sources, for reference."""
    out = []
    for n in sorted(os.listdir(EXAMPLES_DIR)):
        if os.path.splitext(n)[1] in EDIT_EXT:
            out.append({"name": n, "lang": toolchain.detect_lang(n)})
    return out


# ---- docs pane (same idea as the tt-metal Kernel Lab) -------------------------------
L2CPU_DIR = os.path.join(PKG, "l2cpu")
# id -> (title, kind, source). kind "md" = markdown file; "map" = the live register map;
# "code" = a harness source shown fenced so it renders highlighted next to the editor.
# Harness sources are shown fenced so they render highlighted next to the editor — these
# are exactly the includes a kernel builds against (the `include/` headers + the `rt/` runtime).
_DOCS = [
    ("hardware", "Hardware guide", "md", "HARDWARE.md"),
    ("readme", "Loader README", "md", "README.md"),
    ("linux", "Linux + SSH", "md", "LINUX.md"),
    ("map", "Register map", "map", None),
    ("bh.h", "include · bh.h", "code", "include/bh.h"),
    ("tele.h", "include · tele.h", "code", "include/tele.h"),
    ("bh.inc", "include · bh.inc", "code", "include/bh.inc"),
    ("bh.rs", "runtime · bh.rs", "code", "rt/bh.rs"),
    ("crt0.s", "runtime · crt0.s", "code", "rt/crt0.s"),
    ("link.ld", "runtime · link.ld", "code", "rt/link.ld"),
    ("regmap.py", "canonical map · regmap.py", "code", "regmap.py"),
]

# markdown fence language per source extension (drives syntax highlighting in the Docs pane)
_FENCE = {".h": "c", ".inc": "asm", ".rs": "rust", ".s": "asm", ".S": "asm",
          ".ld": "text", ".py": "python"}


def docs_index():
    return [{"id": d[0], "title": d[1], "kind": "md"} for d in _DOCS]


def doc(doc_id):
    for did, title, kind, src in _DOCS:
        if did != doc_id:
            continue
        if kind == "map":
            return {"id": did, "title": title, "markdown": "# Register map\n\n```\n" + map_text() + "\n```\n"}
        full = os.path.join(L2CPU_DIR, src)
        with open(full, encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        if kind == "code":
            fence = _FENCE.get(os.path.splitext(src)[1], "")
            body = f"# {os.path.basename(src)}\n\n```{fence}\n{body}\n```\n"
        return {"id": did, "title": title, "markdown": body}
    raise ValueError(f"unknown doc: {doc_id}")


_STARTERS = {
    "c": '#include <bh.h>\n\nint main(void) {\n    unsigned hb = 0;\n'
         '    TELE[1] = bh_hartid();\n    for (;;) {\n        hb++;\n'
         '        TELE[0] = hb;            /* slot 0 = heartbeat */\n'
         '        TELE[2] = (unsigned)bh_cycles();\n    }\n}\n',
    "asm": '    .include "bh.inc"\n    .option norvc\n'
           '    .section .text._start,"ax"; .globl _start\n_start:\n'
           '    lui   t0, %hi(BH_TELE_BASE)\n    li    t1, 0\n'
           '1:  addi  t1, t1, 1\n    sw    t1, %lo(BH_TELE_BASE)(t0)\n    j 1b\n',
    "rust": '#![no_std]\n#![no_main]\ninclude!(concat!(env!("BH_RT"), "/bh.rs"));\n\n'
            '#[no_mangle]\nextern "C" fn kmain() -> ! {\n    let mut hb: u32 = 0;\n'
            '    unsafe { bh_tele(1, bh_hartid()); }\n    loop {\n'
            '        hb = hb.wrapping_add(1);\n        unsafe { bh_tele(0, hb); }\n    }\n}\n',
}
