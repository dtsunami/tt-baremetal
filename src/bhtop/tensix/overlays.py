"""
tensix.overlays — the registry of bhtop bootloader code overlays + their telemetry/param SCHEMAS.

Each overlay is a freestanding BRISC blob (kernels/tensix/<name>/<name>.c, built to the working
tree's _build/<name>.bin) that the
resident bootloader hot-swaps into a code slot and runs. This module is the single source of truth
the web cockpit and CLI use to: list available overlays, load their bytes to stage, compute the
"loaded kernel hash" (sha256 of the .bin — the identity shown per core), and render LABELED
telemetry + param controls instead of raw hex ("configure telemetry").

Telemetry schema kinds: counter | cycles | rate(work/cycles) | hex | marker | int.
The standard header every overlay fills: telem[0]=work, [1]=cycles, [2]=result, [3]=0xC0FFEE done.
"""
import hashlib
import json
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../bhtop
# Tracked, shipped overlay kernels — folder-per-kernel ({name}/{name}.c + kernel.json) plus the
# shared overlay.h / overlay.ld, mirroring src/bhtop/kernels/x280 (the x280 canon). This is the
# single source of truth: metadata (telemetry/param schemas) lives in each kernel.json, not here.
CANON_DIR = os.path.join(PKG, "kernels", "tensix", "overlays")
# gitignored per-user working area (built .bin + user-compiled custom overlays), under the same
# ~/bhtop/kernels working tree the x280 lab uses.
WORKDIR = os.path.expanduser("~/bhtop/kernels/tensix/overlays")
BUILD_DIR = os.path.join(WORKDIR, "_build")
CUSTOM_DIR = os.path.join(WORKDIR, "_custom")
OVERLAY_DIR = CANON_DIR                                                # -I include + overlay.ld root

# Standard telemetry header shared by every overlay (slots 0..3).
_STD_TELEM = [
    {"slot": 0, "name": "work", "kind": "counter", "desc": "work units done this burst"},
    {"slot": 1, "name": "cycles", "kind": "cycles", "desc": "wall-clock cycles for the burst"},
    {"slot": 2, "name": "result", "kind": "hex", "desc": "checksum / instruction word"},
    {"slot": 3, "name": "done", "kind": "marker", "desc": "0xC0FFEE when complete"},
]


def _load_canon():
    """Read the tracked overlay registry from the canon kernel.json files. Each folder under
    CANON_DIR with a kernel.json {"kind":"overlay", ...} is one overlay; its telemetry is the
    standard header plus any per-kernel `telemetry_extra`. Returns {name: entry}."""
    reg = {}
    if not os.path.isdir(CANON_DIR):
        return reg
    for name in sorted(os.listdir(CANON_DIR)):
        kj = os.path.join(CANON_DIR, name, "kernel.json")
        if not os.path.isfile(kj):
            continue
        try:
            with open(kj) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        if m.get("kind") != "overlay":
            continue
        reg[name] = {
            "title": m.get("title", name),
            "engine": m.get("engine", ""),
            "experimental": bool(m.get("experimental", False)),
            "verified": m.get("verified", "untested"),
            "desc": m.get("desc", ""),
            "params": m.get("params", []),
            "telemetry": _STD_TELEM + (m.get("telemetry_extra") or []),
            "derived": m.get("derived", []),
        }
    return reg


# Builtin registry, cached (re-read if any kernel.json changes on disk).
_BUILTIN = None
_BUILTIN_KEY = None


def _builtin():
    global _BUILTIN, _BUILTIN_KEY
    key = tuple(sorted(
        (n, os.path.getmtime(os.path.join(CANON_DIR, n, "kernel.json")))
        for n in (os.listdir(CANON_DIR) if os.path.isdir(CANON_DIR) else [])
        if os.path.isfile(os.path.join(CANON_DIR, n, "kernel.json"))))
    if _BUILTIN is None or _BUILTIN_KEY != key:
        _BUILTIN = _load_canon()
        _BUILTIN_KEY = key
    return _BUILTIN


# Runtime-registered, user-compiled overlays (the Develop flow). Same shape as builtin entries.
CUSTOM = {}

# Editable starting point shown in the Develop editor.
TEMPLATE = '''// custom overlay — runs on BRISC. Fill telem[0..3]; return a u32 (-> OVL_RET).
#include "overlay.h"

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t n = ovl_param(ctrl, 0);
    if (n == 0) n = 1000000;
    uint32_t c0 = ovl_cycles();
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; i++) { acc += i * 2654435761u; asm volatile("" : "+r"(acc)); }
    uint32_t c1 = ovl_cycles();
    ovl_publish(n, c1 - c0, acc);          // telem[0]=work [1]=cycles [2]=result [3]=done
    return n;
}
'''


def _reg():
    """Merged view of built-in (from canon kernel.json) + custom overlays."""
    return {**_builtin(), **CUSTOM}


def compile(name, source):
    """Compile a user overlay source -> .bin with the SFPI toolchain + overlay.ld, and register it
    so it can be staged. Returns {ok, name, hash, bytes} or {ok:False, log}. Pure host (no device)."""
    import re
    import subprocess
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "custom"
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    src = os.path.join(CUSTOM_DIR, safe + ".c")
    with open(src, "w") as f:
        f.write(source)
    sfpi = os.path.expanduser("~/tt-metal/runtime/sfpi")
    gpp = os.path.join(sfpi, "compiler/bin/riscv-tt-elf-g++")
    oc = os.path.join(sfpi, "compiler/bin/riscv-tt-elf-objcopy")
    ld = os.path.join(OVERLAY_DIR, "overlay.ld")
    elf = os.path.join(BUILD_DIR, safe + ".elf")
    binf = bin_path(safe)
    cflags = ["-Os", "-march=rv32im", "-mabi=ilp32", "-nostdlib", "-ffreestanding",
              "-fno-exceptions", "-fno-rtti", f"-I{OVERLAY_DIR}"]
    c = subprocess.run([gpp, *cflags, "-T", ld, src, "-o", elf], capture_output=True, text=True)
    if c.returncode != 0:
        return {"ok": False, "log": (c.stderr or c.stdout)[-4000:]}
    o = subprocess.run([oc, "-O", "binary", "-j", ".text", "-j", ".rodata", elf, binf],
                       capture_output=True, text=True)
    if o.returncode != 0:
        return {"ok": False, "log": o.stderr[-4000:]}
    data = open(binf, "rb").read()
    CUSTOM[safe] = {
        "title": name, "engine": "custom (BRISC)", "experimental": False, "verified": "custom",
        "desc": "user-compiled overlay", "custom": True, "source": source,
        "params": [{"i": i, "name": f"param{i}", "default": 0, "desc": ""} for i in range(4)],
        "telemetry": _STD_TELEM, "derived": [{"name": "work/cyc", "expr": "work/cycles"}],
    }
    return {"ok": True, "name": safe, "hash": bin_hash(safe), "bytes": len(data),
            "log": (c.stderr or "").strip()}


def _canon_src(name):
    return os.path.join(CANON_DIR, name, name + ".c")


def _work_src(name):
    """Per-user working copy of a builtin overlay source (gitignored), seeded from canon on edit —
    so editing in the cockpit never mutates the tracked canonical source (x280 parity)."""
    return os.path.join(WORKDIR, name, name + ".c")


def source(name):
    """The .c source of an overlay (for the editor). Custom overlays keep their source in-memory;
    builtins prefer an edited working copy, falling back to the tracked canon source."""
    if name in CUSTOM and CUSTOM[name].get("source") is not None:
        return CUSTOM[name]["source"]
    for p in (_work_src(name), _canon_src(name), os.path.join(CUSTOM_DIR, name + ".c")):
        if os.path.exists(p):
            with open(p) as f:
                return f.read()
    raise FileNotFoundError(f"no source for overlay {name!r}")


def save_source(name, src):
    """Persist edited overlay source. Builtin edits go to the gitignored working tree (canon stays
    pristine); custom overlays write to the custom scratch dir."""
    path = _work_src(name) if os.path.exists(_canon_src(name)) else os.path.join(CUSTOM_DIR, name + ".c")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(src)
    if name in CUSTOM:
        CUSTOM[name]["source"] = src
    return {"ok": True, "path": path}


def names():
    return list(_reg().keys())


def bin_path(name):
    return os.path.join(BUILD_DIR, f"{name}.bin")


def bin_bytes(name):
    """Raw overlay bytes to stage. Raises FileNotFoundError if not built (run kernels/tensix/build.sh)."""
    if name not in _reg():
        raise ValueError(f"unknown overlay {name!r}; have {names()}")
    with open(bin_path(name), "rb") as f:
        return f.read()


def disasm(name):
    """objdump -d the built overlay ELF (for the Disasm tab). Returns {ok, text} or {ok:False}."""
    import subprocess
    elf = os.path.join(BUILD_DIR, name + ".elf")
    if not os.path.isfile(elf):
        return {"ok": False, "error": f"overlay {name!r} not built (run kernels/tensix/overlays/build.sh)"}
    objdump = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
    p = subprocess.run([objdump, "-d", "-C", elf], capture_output=True, text=True)
    return {"ok": p.returncode == 0, "name": name,
            "text": p.stdout if p.returncode == 0 else (p.stderr or "objdump failed")}


def bin_hash(name):
    """sha256[:12] of the built .bin — the loaded-kernel identity shown per core. None if unbuilt."""
    try:
        return hashlib.sha256(bin_bytes(name)).hexdigest()[:12]
    except FileNotFoundError:
        return None


def _safe_expr(expr, names):
    """Evaluate a derived-metric expression over the telemetry field names (author-controlled,
    not user input). e.g. 'work*8/cycles'. No builtins; returns float."""
    return eval(expr, {"__builtins__": {}}, {k: float(v) for k, v in names.items()})


def decode_telemetry(name, words):
    """Map raw telem words -> labeled fields + derived metrics, per the overlay's schema.
    Falls back to raw hex for an unknown overlay. `words` is the telem block read from L1."""
    m = _reg().get(name)
    if not m:
        return {"fields": [{"name": f"telem[{i}]", "kind": "hex", "value": int(w)}
                           for i, w in enumerate(words)], "derived": []}
    fields = []
    for f in m["telemetry"]:
        s = f["slot"]
        fields.append({"name": f["name"], "kind": f["kind"], "desc": f.get("desc", ""),
                       "value": int(words[s]) if s < len(words) else 0})
    by = {f["name"]: (words[f["slot"]] if f["slot"] < len(words) else 0) for f in m["telemetry"]}
    derived = []
    for d in m.get("derived", []):
        try:
            v = _safe_expr(d["expr"], by)
        except Exception:
            v = None
        derived.append({"name": d["name"], "value": v})
    return {"fields": fields, "derived": derived}


def manifest():
    """JSON-able list of overlays (metadata + schemas + hash + built?) for the cockpit."""
    out = []
    for name, m in _reg().items():
        b = None
        try:
            b = len(bin_bytes(name))
        except FileNotFoundError:
            pass
        out.append({"name": name, **m, "hash": bin_hash(name), "built": b is not None, "bytes": b})
    return out
