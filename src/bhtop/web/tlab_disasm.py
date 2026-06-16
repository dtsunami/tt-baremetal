"""
tlab_disasm — per-engine RISC-V disassembly of the LAST compute run's JIT-compiled kernels.

tt-metal JIT-compiles kernels to ~/.cache/tt-metal-cache/<build_key>/kernels/<name>/<hash>/
{brisc,ncrisc,trisc0,trisc1,trisc2}/*.elf. We locate the right ELFs by **build hash** via the
Inspector dump (web/inspector.py) — the latest user program's kernels, assembled across their
hash dirs by engine role — and objdump each with the sfpi toolchain. Hash-addressed (not
mtime-guessed), so it's multi-kernel correct and finally surfaces the reader/writer DM kernels
too (they live in their own ncrisc/brisc hash dirs, which the old single-dir scan missed).

Falls back to the newest-by-mtime cache dir when no Inspector dump exists. File-based +
host-side — no device ownership, doesn't fight the poller.
"""
import os
import subprocess

from . import inspector

OBJDUMP = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
CACHE = os.path.expanduser("~/.cache/tt-metal-cache")

# engine role -> (ELF subdir, label). Matches metal._TRISC_ROLE + inspector._ROLE_SUBDIR.
ENGINES = [
    ("reader", "ncrisc", "NCRISC · reader (DM)"),
    ("unpack", "trisc0", "TRISC_0 · UNPACK"),
    ("math",   "trisc1", "TRISC_1 · MATH"),
    ("pack",   "trisc2", "TRISC_2 · PACK"),
    ("writer", "brisc",  "BRISC · writer (DM)"),
]


def _newest_subdir(path):
    if not os.path.isdir(path):
        return None
    subs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return max(subs, key=os.path.getmtime) if subs else None


def _fallback_elfs():
    """role -> elf via the old newest-by-mtime heuristic, when no Inspector dump exists.
    One dir only holds some engines, so this is best-effort (returns what's present)."""
    bk = _newest_subdir(CACHE)
    kroot = os.path.join(bk, "kernels") if bk else None
    if not kroot or not os.path.isdir(kroot):
        return {}, None
    best, best_mt, best_name = None, -1, None
    for name in os.listdir(kroot):
        h = _newest_subdir(os.path.join(kroot, name))
        if h and os.path.getmtime(h) > best_mt:
            best, best_mt, best_name = h, os.path.getmtime(h), name
    if not best:
        return {}, None
    elfs = {}
    for role, sub, _ in ENGINES:
        elf = os.path.join(best, sub, f"{sub}.elf")
        if os.path.exists(elf):
            elfs[role] = elf
    return elfs, best_name


def _objdump(elf, limit=600):
    if not os.path.exists(OBJDUMP):
        return "(sfpi objdump not found)"
    try:
        r = subprocess.run([OBJDUMP, "-d", elf], capture_output=True, text=True, timeout=15)
    except Exception as e:                       # pragma: no cover
        return f"(objdump failed: {e})"
    lines = (r.stdout or r.stderr).splitlines()
    # keep instruction + label lines (drop the elf header preamble)
    keep = [ln for ln in lines if ln[:1].isspace() and ":\t" in ln or "<" in ln and ">:" in ln]
    return "\n".join((keep or lines)[:limit])


def fetch_last():
    """Disassembly for each engine of the last compute run, or an explanatory error.

    Hash-addressed via the Inspector (latest user program's kernels assembled by role), with a
    newest-by-mtime fallback when no dump exists yet."""
    pid, kernels = inspector.latest_user_program()
    if kernels:
        elfs = inspector.engine_elfs(kernels)
        kname = " + ".join(sorted({k["name"] for k in kernels}))
    else:
        elfs, kname = _fallback_elfs()
    if not elfs:
        return {"ok": False, "error": "no JIT-compiled kernels yet — Run a compute example first."}
    engines = [{"role": role, "label": label, "present": role in elfs,
                "disasm": _objdump(elfs[role]) if role in elfs else ""}
               for role, _, label in ENGINES]
    return {"ok": True, "kernel": kname, "program_id": pid, "engines": engines}
