"""
tlab_build — standalone (no-device) build of a Tensix kernel from a bhtop-owned copy, by REPLAYING
tt-metal's own JIT compile/link commands. The x280 lab owns its toolchain; tt-metal kernels can't
be that simple (the recipe is ~50 includes + defines + a firmware wrapper + a linker script), so we
don't reconstruct it — we capture tt-metal's EXACT command and rerun it against your extracted source.

Flow (the user does one Run first, then builds offline as many times as they like):
  1. CAPTURE  — bhtop sets TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 (see metal._env), so a Run logs
                "g++ compile cmd: …" / "g++ link cmd: …" per kernel. parse_recipe() pulls those out
                of the captured Run output; save_recipe() persists them per example.
  2. EXTRACT  — copy the kernel .cpp(s) (the files tt-metal #includes via kernel_includes.hpp) into
                ~/bhtop/kernels/tensix/<example>/src/, leaving the tt-metal tree untouched.
  3. BUILD    — stage tt-metal's generated headers, REPOINT kernel_includes.hpp at your extracted
                .cpp, relocate the command's paths into a bhtop build dir, run compile+link (no
                device). Emit: the exact command, the build log, the .elf (already has -g symbols),
                objdump disassembly, and the symbol table.

PURE filesystem + subprocess (no device). Transparent: every result carries the command it ran.
"""
import json
import os
import re
import shutil
import subprocess

from . import inspector, tlab

OVERLAY = os.path.expanduser("~/bhtop/kernels/tensix")
SFPI = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin")
OBJDUMP = os.path.join(SFPI, "riscv-tt-elf-objdump")
NM = os.path.join(SFPI, "riscv-tt-elf-nm")

# "… g++ compile cmd: cd <out> && <gpp> … -c -o <obj> <src.cc> …"  /  "g++ link cmd: …"
_COMPILE_RE = re.compile(r"g\+\+ compile cmd:\s*(?P<cmd>cd\s+.+)$")
_LINK_RE = re.compile(r"g\+\+ link cmd:\s*(?P<cmd>cd\s+.+)$")
_CD_RE = re.compile(r"^cd\s+(?P<dir>\S+)\s+&&")
_ELF_RE = re.compile(r"-o\s+(?P<elf>\S+\.elf)\b")


def _build_dir(example):
    return os.path.join(OVERLAY, example, ".build")


def _recipe_path(example):
    return os.path.join(_build_dir(example), "recipe.json")


# ---- 1. capture ---------------------------------------------------------------------
def parse_recipe(output):
    """Pull per-target {compile, link, out_dir, elf, target} from a Run's captured output. Targets
    are keyed by their ELF path (unique per kernel-engine: …/<name>/<hash>/<risc>/<risc>.elf)."""
    units = {}
    for line in output.splitlines():
        for kind, rx in (("compile", _COMPILE_RE), ("link", _LINK_RE)):
            m = rx.search(line)
            if not m:
                continue
            cmd = m.group("cmd").strip()
            cd = _CD_RE.match(cmd)
            out_dir = cd.group("dir") if cd else None
            elf = _ELF_RE.search(cmd)
            # the compile and link of one target share the same out_dir → group on it
            key = (elf.group("elf") if elf else out_dir) or cmd[:40]
            if kind == "link" and elf:
                key = elf.group("elf")
            u = units.setdefault(out_dir or key, {"out_dir": out_dir})
            u[kind] = cmd
            if elf:
                u["elf"] = elf.group("elf")
    # keep only complete units (have both a compile and a link)
    return {k: v for k, v in units.items() if "compile" in v}


def save_recipe(example, output):
    """Parse a Run's output and persist the recipe for `example`. Returns the count captured."""
    units = parse_recipe(output)
    if not units:
        return {"ok": False, "captured": 0}
    os.makedirs(_build_dir(example), exist_ok=True)
    with open(_recipe_path(example), "w") as fh:
        json.dump({"example": example, "units": list(units.values())}, fh, indent=2)
    return {"ok": True, "captured": len(units), "path": _recipe_path(example)}


def load_recipe(example):
    p = _recipe_path(example)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# ---- 2. extract ---------------------------------------------------------------------
def _kernel_includes(out_dir):
    """The hash dir holding kernel_includes.hpp at/above out_dir, or None."""
    d = out_dir
    for _ in range(3):
        if d and os.path.isfile(os.path.join(d, "kernel_includes.hpp")):
            return d
        d = os.path.dirname(d.rstrip("/")) if d else None
    return None


def _user_source(hashdir):
    """The user kernel .cpp path that kernel_includes.hpp pulls in (the last #include of a real path)."""
    try:
        txt = open(os.path.join(hashdir, "kernel_includes.hpp")).read()
    except OSError:
        return None
    paths = re.findall(r'#include\s+"(/[^"]+\.cpp)"', txt)
    return paths[-1] if paths else None


def extract(example):
    """Copy the example's kernel .cpp sources into ~/bhtop/kernels/tensix/<example>/src/ (mirroring
    their programming_examples-relative path), leaving the tt-metal tree untouched. Returns the
    files copied. Uses the recipe's kernel_includes when available, else the example's kernels/."""
    root = tlab.examples_root()
    dst_root = os.path.join(OVERLAY, example, "src")
    copied = []
    srcs = set()
    rec = load_recipe(example)
    if rec:
        for u in rec["units"]:
            hd = _kernel_includes(u.get("out_dir") or "")
            s = _user_source(hd) if hd else None
            if s and os.path.isfile(s):
                srcs.add(s)
    if not srcs and root:                     # fallback: every .cpp under the example's kernels/
        exdir = os.path.join(root, example)
        for dp, _, names in os.walk(exdir):
            if f"{os.sep}kernels{os.sep}" in dp + os.sep:
                srcs.update(os.path.join(dp, n) for n in names if n.endswith((".cpp", ".hpp", ".h")))
    for s in sorted(srcs):
        rel = os.path.relpath(s, root) if root and s.startswith(os.path.realpath(root)) else os.path.basename(s)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(s, dst)
        copied.append({"src": s, "dst": dst, "rel": rel})
    return {"ok": bool(copied), "dir": dst_root, "files": copied}


def extracted_path(example, user_src):
    """Where an extracted copy of `user_src` lives (mirrors its programming_examples-relative path)."""
    root = tlab.examples_root()
    rel = os.path.relpath(user_src, root) if root and user_src.startswith(os.path.realpath(root)) else os.path.basename(user_src)
    return os.path.join(OVERLAY, example, "src", rel)


# ---- 3. build (replay, no device) ---------------------------------------------------
def _run(cmd, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _objdump(elf):
    if not os.path.exists(OBJDUMP) or not os.path.exists(elf):
        return ""
    r = subprocess.run([OBJDUMP, "-d", elf], capture_output=True, text=True, timeout=20)
    return r.stdout or r.stderr


def _symbols(elf):
    if not os.path.exists(NM) or not os.path.exists(elf):
        return ""
    r = subprocess.run([NM, "-n", "-S", elf], capture_output=True, text=True, timeout=20)
    return r.stdout or r.stderr


def build(example):
    """Replay tt-metal's compile+link for each captured kernel-engine, against the EXTRACTED source,
    in a bhtop build dir (no device). Returns per-unit {target, ok, cmd, log, elf, disasm, symbols}."""
    rec = load_recipe(example)
    if not rec:
        return {"ok": False, "error": "no build recipe yet — Run this example once "
                "(bhtop logs the compile commands), then Build."}
    results = []
    for u in rec["units"]:
        out_dir = (u.get("out_dir") or "").rstrip("/")
        hashdir = _kernel_includes(out_dir)
        target = os.path.basename(u.get("elf", out_dir))
        if not hashdir:
            results.append({"target": target, "ok": False, "log": "kernel_includes.hpp not found "
                            "(cache evicted?) — re-Run the example to regenerate."})
            continue
        stage = os.path.join(_build_dir(example), os.path.relpath(hashdir, os.path.dirname(hashdir)))
        try:
            res = _build_unit(example, u, hashdir, stage, target)
        except Exception as e:                       # transparent: never sink the whole build
            res = {"target": target, "ok": False, "log": f"build error: {type(e).__name__}: {e}"}
        results.append(res)
    return {"ok": all(r["ok"] for r in results) if results else False,
            "example": example, "units": results}


def _build_unit(example, u, hashdir, stage, target):
    # stage tt-metal's generated headers into a bhtop build dir
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    shutil.copytree(hashdir, stage, ignore=shutil.ignore_patterns("*.elf", "*.o", "*.log", "*.d", "*.xip.elf"))
    # repoint kernel_includes.hpp at the EXTRACTED copy of the user kernel
    user_src = _user_source(hashdir)
    repoint = None
    if user_src:
        ext = extracted_path(example, user_src)
        if os.path.isfile(ext):
            ki = os.path.join(stage, "kernel_includes.hpp")
            open(ki, "w").write(open(os.path.join(hashdir, "kernel_includes.hpp")).read().replace(user_src, ext))
            repoint = ext
    # relocate every reference to the original hash dir into our staging dir, then run
    log, elf_out = "", None
    for step in ("compile", "link"):
        if step not in u:
            continue
        cmd = u[step].replace(hashdir, stage)
        log += f"$ {cmd}\n"
        rc, out = _run(cmd)
        log += out + (f"\n[exit {rc}]\n\n" if rc else "\n")
        if rc != 0:
            return {"target": target, "ok": False, "source": repoint, "cmd": cmd, "log": log}
    elf_out = (u.get("elf") or "").replace(hashdir, stage)
    ok = bool(elf_out) and os.path.isfile(elf_out)
    return {"target": target, "ok": ok, "source": repoint, "elf": elf_out if ok else None,
            "log": log.strip() or "(compiled clean)",
            "disasm": _objdump(elf_out) if ok else "", "symbols": _symbols(elf_out) if ok else ""}
