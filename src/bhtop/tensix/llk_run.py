"""
tensix.llk_run — build + load + run an LLK perf kernel on a Tensix core over tt-exalens.

This is the on-device half of the LLK lane ([[tensix-llk]] / kernels/tensix/llk). It ports tt-llk's
own TRISC-boot sequence (tests/python_tests/helpers/test_config.run_elf_files, BootMode.TRISC) onto
bhtop's exalens context — no BRISC, no metal. The kernels are built with -DLLK_BOOT_MODE_TRISC, so
T0 (unpack) self-boots: it runs device_setup() + clear_trisc_soft_reset(), releasing T1/T2. The
cockpit therefore only has to load the three ELFs and deassert TRISC0.

Sequence (mirrors run_elf_files BootMode.TRISC):
  1. assert all three TRISC soft-resets
  2. load_elf trisc0/1/2  (writes code/data, sets each reset-PC; cores stay in reset)
  3. write RuntimeParams (FormatConfig + TILE_CNT) to L1 __runtime_args_start
  4. reset the three mailboxes (0xA3)
  5. deassert TRISC0 only -> T0 boots, device_setup()s, releases T1/T2
  6. poll the mailboxes for KERNEL_COMPLETE (0xFF), per thread

Device access only — call it on the DeviceManager worker thread, sharing dm.ctx so the chip never
sees a second owner. Perf-counter readback (the 5 banks / zoned cycles) is a later telemetry layer;
v1 reports per-thread completion + the raw mailbox + a peek at the perf-counter L1 region.
"""
import os
import struct
import subprocess
import time

# Fixed L1 layout (from tt-llk: ckernel.h, llk_params.Mailboxes, sections.ld, perf.h).
RUNTIME_ARGS_START = 0x20000          # __runtime_args_start (NOLOAD .runtime_args)
MAILBOX_UNPACK     = 0x1FFB8          # +0/+4/+8 = unpack/math/pack thread mailboxes
KERNEL_COMPLETE    = 0xFF
RESET_MAILBOX_VAL  = 0xA3
PERF_COUNTERS_BASE = 0x169000

_COMPONENTS = ["UNPACK", "MATH", "PACK"]          # canon thread names -> trisc0/1/2
_THREADS = {"UNPACK": "trisc0", "MATH": "trisc1", "PACK": "trisc2"}
_MAILBOX = {"UNPACK": MAILBOX_UNPACK, "MATH": MAILBOX_UNPACK + 4, "PACK": MAILBOX_UNPACK + 8}

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../bhtop
CANON_DIR = os.path.join(PKG, "kernels", "tensix", "llk")
BUILD_DIR = os.path.expanduser("~/bhtop/kernels/tensix/llk/_build")


def build(name, run_type=None, timeout=180):
    """Compile+link the kernel's ELFs via kernels/tensix/llk/build.sh, generating a build.h for the
    chosen run type (PERF_RUN_TYPE). Returns {ok, log, elfs, run_type, fields}."""
    from . import llk
    out = os.path.join(BUILD_DIR, name)
    os.makedirs(out, exist_ok=True)
    text, rt, fields = llk.gen_build_h(name, run_type)         # variant build.h for this run type
    bh = os.path.join(out, "build.h")
    with open(bh, "w") as f:
        f.write(text)
    sh = os.path.join(CANON_DIR, "build.sh")
    p = subprocess.run(["bash", sh, name, bh], capture_output=True, text=True, timeout=timeout)
    elfs = {c: os.path.join(out, c + ".elf") for c in _COMPONENTS
            if os.path.isfile(os.path.join(out, c + ".elf"))}
    import hashlib
    artifacts = []
    for c, p_elf in elfs.items():
        data = open(p_elf, "rb").read()
        artifacts.append({"thread": c, "file": c + ".elf", "bytes": len(data),
                          "sha": hashlib.sha256(data).hexdigest()[:12]})
    return {"ok": p.returncode == 0, "log": (p.stdout + p.stderr).strip(), "elfs": elfs,
            "dir": out, "run_type": rt, "fields": fields, "artifacts": artifacts,
            "flags": "-mcpu=tt-bh-tensix -O3 -std=c++17 -DARCH_BLACKHOLE -DLLK_TRISC_{UNPACK,MATH,PACK} "
                     "-DLLK_BOOT_MODE_TRISC -DRUNTIME_FORMATS  (full recipe: kernels/tensix/llk/build.sh)"}


def disasm(name):
    """objdump -d each built TRISC ELF (for the Disasm tab). Returns {thread: text} or {error}."""
    out = os.path.join(BUILD_DIR, name)
    objdump = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
    res = {}
    for c in _COMPONENTS:
        elf = os.path.join(out, c + ".elf")
        if not os.path.isfile(elf):
            continue
        p = subprocess.run([objdump, "-d", "-C", elf], capture_output=True, text=True)
        res[c] = p.stdout if p.returncode == 0 else (p.stderr or "objdump failed")
    if not res:
        return {"ok": False, "error": f"{name} not built — Build it first"}
    return {"ok": True, "name": name, "threads": res}


def _present(name):
    """Which TRISC components this kernel implements (by built ELF)."""
    out = os.path.join(BUILD_DIR, name)
    return [c for c in _COMPONENTS if os.path.isfile(os.path.join(out, c + ".elf"))]


def _runtime_params(formats, tile_cnt):
    """RuntimeParams = FormatConfig (12 u32) + TILE_CNT (u32). `formats` is a 12-int list (the
    DataFormat enum values for each FormatConfig field) or None for all-zero (activity/cycle run)."""
    f = list(formats or [0] * 12)[:12]
    f += [0] * (12 - len(f))
    return f + [int(tile_cnt)]


def run(name, coord, *, ctx, device_id=0, tile_cnt=16, formats=None, timeout=5.0, poll=0.02):
    """Load the kernel onto the Tensix core at `coord` and run it (TRISC boot). Returns a dict with
    per-thread completion + the perf-counter region peek. `coord` is anything exalens accepts
    (OnChipCoordinate or 'x,y'); `ctx` MUST be the DeviceManager's shared context."""
    from ttexalens.tt_exalens_lib import load_elf, write_words_to_device, read_word_from_device

    comps = _present(name)
    if not comps:
        return {"ok": False, "error": f"{name} not built — run build() first"}
    dev = ctx.devices[device_id]
    block = dev.get_block(coord)
    rdbg = {c: block.get_risc_debug(_THREADS[c]) for c in comps}

    # 1. assert all present TRISCs into soft reset
    for c in comps:
        rdbg[c].set_reset_signal(True)

    # 2. load each ELF onto its TRISC (code/data + reset-PC; stays in reset)
    out = os.path.join(BUILD_DIR, name)
    for c in comps:
        load_elf(elf_file=os.path.join(out, c + ".elf"), location=coord,
                 risc_name=_THREADS[c], device_id=device_id, context=ctx, verify_write=False)

    # 3. RuntimeParams -> L1 (zero the region first so unwritten fields read 0, not stale garbage),
    #    then 4. reset mailboxes
    write_words_to_device(coord, RUNTIME_ARGS_START, [0] * 64, device_id=device_id, context=ctx)
    write_words_to_device(coord, RUNTIME_ARGS_START, _runtime_params(formats, tile_cnt),
                          device_id=device_id, context=ctx)
    write_words_to_device(coord, MAILBOX_UNPACK, [RESET_MAILBOX_VAL] * 3,
                          device_id=device_id, context=ctx)

    # 5. deassert TRISC0 only — T0 boots, device_setup()s, releases T1/T2 (LLK_BOOT_MODE_TRISC)
    t0 = "UNPACK" if "UNPACK" in rdbg else comps[0]
    rdbg[t0].set_reset_signal(False)

    # 6. poll mailboxes for KERNEL_COMPLETE
    threads = {c: {"mailbox": _MAILBOX[c], "done": False, "value": None} for c in comps}
    deadline = time.time() + timeout
    while time.time() < deadline and not all(t["done"] for t in threads.values()):
        for c, t in threads.items():
            if not t["done"]:
                t["value"] = read_word_from_device(coord, t["mailbox"], device_id=device_id, context=ctx)
                t["done"] = (t["value"] == KERNEL_COMPLETE)
        if not all(t["done"] for t in threads.values()):
            time.sleep(poll)

    # perf-counter region peek (raw; full bank decode is a later telemetry layer)
    perf = read_word_from_device(coord, PERF_COUNTERS_BASE, device_id=device_id, context=ctx)
    ok = all(t["done"] for t in threads.values())
    return {"ok": ok, "name": name, "coord": str(coord), "tile_cnt": tile_cnt,
            "threads": {c: {"done": t["done"], "mailbox_hex": hex(t["value"] or 0)}
                        for c, t in threads.items()},
            "perf_base_hex": hex(perf or 0),
            "status": "complete" if ok else "timeout — some thread did not reach KERNEL_COMPLETE"}


def build_and_run(name, coord, *, ctx, device_id=0, tile_cnt=16, timeout=5.0):
    b = build(name)
    if not b["ok"]:
        return {"ok": False, "stage": "build", "log": b["log"]}
    r = run(name, coord, ctx=ctx, device_id=device_id, tile_cnt=tile_cnt, timeout=timeout)
    r["build_log"] = b["log"]
    return r
