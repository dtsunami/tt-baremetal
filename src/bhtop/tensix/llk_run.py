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
PERF_COUNTERS_BASE = 0x169000        # L1 software perf region (tt-llk perf.h; ends 0x16AFF4).
                                     # NOT the 0xFFB12000 HW debug perf MMIO — distinct source.

_COMPONENTS = ["UNPACK", "MATH", "PACK"]          # canon thread names -> trisc0/1/2
_THREADS = {"UNPACK": "trisc0", "MATH": "trisc1", "PACK": "trisc2"}
_MAILBOX = {"UNPACK": MAILBOX_UNPACK, "MATH": MAILBOX_UNPACK + 4, "PACK": MAILBOX_UNPACK + 8}

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../bhtop
CANON_DIR = os.path.join(PKG, "kernels", "tensix", "llk")
BUILD_DIR = os.path.expanduser("~/bhtop/kernels/tensix/llk/_build")


def _variant_dir(name, variant):
    return os.path.join(BUILD_DIR, f"{name}__{variant}" if variant else name)


def build(name, run_type=None, timeout=180, fidelity=None, fp32_acc=None, formats=None, overrides=None,
          variant=None, cache=False):
    """Compile+link the kernel's ELFs via kernels/tensix/llk/build.sh, generating a build.h for the
    chosen run type (PERF_RUN_TYPE). Returns {ok, log, elfs, run_type, fields}.

    fidelity/fp32_acc/formats are the precision knobs (see llk.gen_build_h): they flow into the
    generated build.h so a bit-exact matmul RUN can select HiFi4 + fp32 dest-acc + an explicit
    FormatConfig without hand-editing a header. variant: build into a per-variant ELF dir (so distinct
    configs — e.g. one SFPU op each — don't clobber each other). cache=True: skip the compile if that
    variant's ELFs already exist (amortizes builds across a training loop)."""
    from . import llk
    out = _variant_dir(name, variant)
    os.makedirs(out, exist_ok=True)
    if cache and all(os.path.isfile(os.path.join(out, c + ".elf")) for c in _COMPONENTS):
        elfs = {c: os.path.join(out, c + ".elf") for c in _COMPONENTS}
        return {"ok": True, "log": "(cached)", "elfs": elfs, "dir": out, "run_type": run_type,
                "fields": None, "artifacts": [], "cached": True}
    text, rt, fields = llk.gen_build_h(name, run_type, fidelity=fidelity,
                                       fp32_acc=fp32_acc, formats=formats, overrides=overrides)
    bh = os.path.join(out, "build.h")
    with open(bh, "w") as f:
        f.write(text)
    sh = os.path.join(CANON_DIR, "build.sh")
    env = dict(os.environ, OUT=out)
    p = subprocess.run(["bash", sh, name, bh], capture_output=True, text=True, timeout=timeout, env=env)
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
    """RuntimeParams runtime image. Formats are now COMPILE-TIME (static constexpr FormatConfig in
    build.h, the SPEED_OF_LIGHT/compile_time_formats path), so they are NOT a runtime instance member
    and are no longer written here. gen_build_h ALWAYS emits TILE_CNT as the first runtime field, so
    offset 0 == TILE_CNT for every kernel; we write tile_cnt there. The region is pre-zeroed so the
    remaining scalar fields (and any Operand stimuli buffers) default to 0. `formats` is accepted for
    call-site compatibility but ignored. Kernels that take other runtime dims
    (CT_DIM/KT_DIM/RT_DIM/LOOP_FACTOR/TILE_SIZE_UNPACK_* — e.g. the matmul family) supply their full
    param vector via run(runtime_words=[...]) in RuntimeParams struct order; see tensix.matmul which
    builds that vector for a bit-exact A@B."""
    return [int(tile_cnt)]


def run(name, coord, *, ctx, device_id=0, tile_cnt=16, formats=None, timeout=5.0, poll=0.02,
        runtime_words=None, variant=None):
    """Load the kernel onto the Tensix core at `coord` and run it (TRISC boot). Returns a dict with
    per-thread completion + the perf-counter region peek. `coord` is anything exalens accepts
    (OnChipCoordinate or 'x,y'); `ctx` MUST be the DeviceManager's shared context.

    runtime_words — if given, this exact list of u32 is written verbatim to RUNTIME_ARGS_START
    (after zeroing the region), overriding the single-tile _runtime_params default. The matmul RUN
    uses it to supply all 10 dims in RuntimeParams struct order (see tensix.matmul).
    variant — load the ELFs from the per-variant build dir (see build(variant=...))."""
    from ttexalens.tt_exalens_lib import load_elf, write_words_to_device, read_word_from_device

    out = _variant_dir(name, variant)
    comps = [c for c in _COMPONENTS if os.path.isfile(os.path.join(out, c + ".elf"))]
    if not comps:
        return {"ok": False, "error": f"{name} (variant={variant}) not built — run build() first"}
    dev = ctx.devices[device_id]
    block = dev.get_block(coord)
    rdbg = {c: block.get_risc_debug(_THREADS[c]) for c in comps}

    # 1. assert all present TRISCs into soft reset
    for c in comps:
        rdbg[c].set_reset_signal(True)

    # 2. load each ELF onto its TRISC (code/data + reset-PC; stays in reset)
    for c in comps:
        load_elf(elf_file=os.path.join(out, c + ".elf"), location=coord,
                 risc_name=_THREADS[c], device_id=device_id, context=ctx, verify_write=False)

    # 3. RuntimeParams -> L1 (zero the region first so unwritten fields read 0, not stale garbage),
    #    then 4. reset mailboxes
    write_words_to_device(coord, RUNTIME_ARGS_START, [0] * 64, device_id=device_id, context=ctx)
    words = runtime_words if runtime_words is not None else _runtime_params(formats, tile_cnt)
    write_words_to_device(coord, RUNTIME_ARGS_START, words,
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
