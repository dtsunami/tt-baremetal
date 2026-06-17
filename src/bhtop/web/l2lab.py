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

from . import labkit
from ..l2cpu import toolchain, regmap

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../bhtop
EXAMPLES_DIR = os.path.join(PKG, "l2cpu", "examples")
WORKDIR = os.path.expanduser("~/bhtop/l2cpu_kernels")
EDIT_EXT = {".c", ".s", ".S", ".rs"}


# ---- workspace ----------------------------------------------------------------------
def _ensure_workspace():
    """Create the workspace and seed it from the bundled examples. Additive: copies any
    bundled example not already present (so new examples show up) without touching edits."""
    os.makedirs(WORKDIR, exist_ok=True)
    for n in os.listdir(EXAMPLES_DIR):
        if os.path.splitext(n)[1] in EDIT_EXT:
            dst = os.path.join(WORKDIR, n)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(EXAMPLES_DIR, n), dst)
    return WORKDIR


def _safe(name):
    """Resolve a workspace-relative filename, refusing traversal + bad extensions
    (shared labkit.safe_path)."""
    return labkit.safe_path(_ensure_workspace(), name, EDIT_EXT)


def files():
    """List workspace kernels (name, lang, bytes), C first then asm/Rust."""
    root = _ensure_workspace()
    out = []
    for n in sorted(os.listdir(root)):
        full = os.path.join(root, n)
        if os.path.isfile(full) and os.path.splitext(n)[1] in EDIT_EXT:
            out.append({"name": n, "lang": toolchain.detect_lang(n),
                        "bytes": os.path.getsize(full)})
    out.sort(key=lambda f: ({"c": 0, "asm": 1, "rust": 2}.get(f["lang"], 3), f["name"]))
    return out


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
    """Create a starter kernel of the given language if it doesn't exist."""
    if "." not in name:
        name += {"c": ".c", "asm": ".s", "rust": ".rs"}.get(lang, ".c")
    full = _safe(name)
    if os.path.exists(full):
        raise ValueError(f"{name} already exists")
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
def compile_kernel(content, lang, addr):
    """Compile editor `content` to a flat image. Returns words/bytes + disasm on success,
    or parsed compiler errors on failure — never raises (so the UI can show them inline)."""
    ext = {"c": ".c", "asm": ".s", "rust": ".rs"}.get(lang, ".c")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "kernel" + ext)
        with open(src, "w") as fh:
            fh.write(content)
        try:
            words = toolchain.compile_source(src, base=addr, lang=lang)
        except toolchain.ToolError as e:
            msg = str(e)
            errs = [{**err, "file": os.path.basename(err["file"])}
                    for err in labkit.parse_compiler_errors(msg)]
            return {"ok": False, "error": msg, "errors": errs}
        try:
            dis = toolchain.disasm(src, base=addr, lang=lang)
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
