"""
tensix.resident — drive a RESIDENT, doorbell-driven Tensix kernel over tt-exalens.

The stock LLK perf life-cycle is one host dispatch = one reboot: load 3 ELFs, boot T0, poll the
mailboxes for KERNEL_COMPLETE (0xFF). The render fires ~190 such dispatches per tile, and the host
round-trip (load+boot+poll over exalens) dominates — not the on-device compute. A RESIDENT kernel
(kernels/tensix/llk/resident_mm_perf) breaks that: all three compute threads run INIT once then spin in
a `for(;;)` doorbell loop, so the host loads ONCE and drives N tiles by (re)staging operands and
ringing a doorbell word — the same residency the x280 opt_step.c proved, now across the 3 Tensix
threads (the keystone the 120-worker resident grid needs).

This module is the host half: `ResidentMatmul` boots the kernel (WITHOUT the KERNEL_COMPLETE poll —
the threads never complete) and exposes `.ring(a, b)` = stage operands -> bump doorbell -> poll DONE ->
read + untilize the output. `.close()` parks the cores (assert soft-reset).

Doorbell mailbox (free L1 gap, must match resident_mm_perf.cpp):
  DB=0x16000 (host->kernel ring), DONE=0x16010 (kernel->host completed), HB=0x16020 (heartbeat).
"""
import os
import time

from . import llk_run
from . import matmul as MM

DB_RING = 0x16000
DB_DONE = 0x16010
DB_HB   = 0x16020

_THREADS = {"UNPACK": "trisc0", "MATH": "trisc1", "PACK": "trisc2"}
_COMPONENTS = ["UNPACK", "MATH", "PACK"]


def boot_resident(name, coord, *, ctx, device_id=0, runtime_words, clear_words=32, variant=None):
    """Load a resident LLK kernel's 3 ELFs onto `coord` and boot T0 WITHOUT polling KERNEL_COMPLETE
    (a resident kernel loops forever and never completes). Generic for any doorbell kernel — the fused
    render reuses this. Sequence mirrors llk_run.run's boot but skips the completion poll and zeroes the
    doorbell region (0x16000.. `clear_words`) before deasserting T0 so no thread reads a stale ring/flag.

    runtime_words: the RuntimeParams image (struct order, TILE_CNT first). Returns the per-thread
    RiscDebug handles (so the caller can park the cores with set_reset_signal(True))."""
    from ttexalens.tt_exalens_lib import load_elf, write_words_to_device
    out = llk_run._variant_dir(name, variant)
    comps = [c for c in _COMPONENTS if os.path.isfile(os.path.join(out, c + ".elf"))]
    if len(comps) != 3:
        raise RuntimeError(f"{name} (variant={variant}): expected 3 ELFs, got {comps}")
    block = ctx.devices[device_id].get_block(coord)
    rdbg = {c: block.get_risc_debug(_THREADS[c]) for c in comps}
    for c in comps:
        rdbg[c].set_reset_signal(True)
    for c in comps:
        load_elf(elf_file=os.path.join(out, c + ".elf"), location=coord, risc_name=_THREADS[c],
                 device_id=device_id, context=ctx, verify_write=False)
    write_words_to_device(coord, llk_run.RUNTIME_ARGS_START, [0] * 64, device_id=device_id, context=ctx)
    write_words_to_device(coord, llk_run.RUNTIME_ARGS_START, runtime_words, device_id=device_id, context=ctx)
    write_words_to_device(coord, DB_RING, [0] * clear_words, device_id=device_id, context=ctx)
    write_words_to_device(coord, llk_run.MAILBOX_UNPACK, [llk_run.RESET_MAILBOX_VAL] * 3,
                          device_id=device_id, context=ctx)
    rdbg["UNPACK"].set_reset_signal(False)
    return rdbg


class ResidentMatmul:
    """A resident bare-metal Tensix matmul: boot once, then `ring(a,b)` per tile with no reload.

    out_format: "bf16" | "fp32" | "int32" (same _MODES as tensix.matmul). A single 32x32 @ 32x32 tile.
    """

    def __init__(self, coord, *, ctx, device_id=0, out_format="fp32", name="resident_mm_perf"):
        self.coord = coord
        self.ctx = ctx
        self.device_id = device_id
        self.out_format = out_format
        self.name = name
        self.mode = MM._MODES[out_format]
        self._ring = 0
        self._booted = False
        self._encode = ((lambda m: MM.pack_int8_words(MM.tilize32(m))) if self.mode["in_df"] == MM.DF_INT8
                        else (lambda m: MM.pack_bf16_words([float(x) for x in MM.tilize32(m)])))
        self._decode = (MM.unpack_int32_words if out_format == "int32"
                        else MM.unpack_fp32_words if out_format == "fp32"
                        else MM.unpack_bf16_words)
        self._out_words = self.mode["out_words"]

    # ---- build + boot -----------------------------------------------------------------------------
    def build(self):
        b = llk_run.build(self.name, run_type="L1_TO_L1", fidelity=self.mode["fidelity"],
                          fp32_acc=self.mode["fp32_acc"], formats=self.mode["formats"])
        if not b["ok"]:
            raise RuntimeError(f"resident_mm_perf build failed:\n{b['log']}")
        return b

    def boot(self, *, build=True):
        """Load the 3 ELFs and boot T0 (which releases T1/T2). Do NOT poll KERNEL_COMPLETE — the
        threads spin in the doorbell loop forever. Initializes the doorbell region before deasserting T0."""
        if build:
            self.build()
        tsize = MM.TILE_SIZE_WORDS[self.mode["in_df"]]      # single 32x32 tile, stock matmul ABI
        runtime_words = [1, 1, 1, 1, 1, tsize, tsize, 0, 4, 4]
        self._rdbg = boot_resident(self.name, self.coord, ctx=self.ctx, device_id=self.device_id,
                                   runtime_words=runtime_words)
        self._booted = True
        self._ring = 0
        return self

    # ---- drive ------------------------------------------------------------------------------------
    def ring_async(self, a, b, *, b_prestaged=False):
        """Stage A,B at PERF_INPUT_A/B, poison the output, and bump the doorbell — but DON'T wait.
        Returns the ring id. Pair with collect(). This split lets a host drive MANY resident workers
        concurrently (ring them all, then collect them all) — the grid-parallel path."""
        from ttexalens.tt_exalens_lib import write_words_to_device
        if not self._booted:
            raise RuntimeError("boot() first")
        write_words_to_device(self.coord, MM.PERF_INPUT_A, self._encode(a),
                              device_id=self.device_id, context=self.ctx)
        if not b_prestaged:
            write_words_to_device(self.coord, MM.PERF_INPUT_B, self._encode(b),
                                  device_id=self.device_id, context=self.ctx)
        write_words_to_device(self.coord, MM.PERF_OUTPUT, [0xBADF00D5] * self._out_words,
                              device_id=self.device_id, context=self.ctx)
        self._ring += 1
        write_words_to_device(self.coord, DB_RING, [self._ring], device_id=self.device_id, context=self.ctx)
        self._pending = (a, b, self._ring)
        return self._ring

    def collect(self, *, timeout=4.0, poll=0.005, verify_golden=True):
        """Poll DONE for the last ring_async, then read + untilize PERF_OUTPUT and (optionally) verify."""
        from ttexalens.tt_exalens_lib import read_word_from_device, read_words_from_device
        a, b, r = self._pending
        t0 = time.time()
        done = None
        while time.time() - t0 < timeout:
            done = read_word_from_device(self.coord, DB_DONE, device_id=self.device_id, context=self.ctx)
            if done == r:
                break
            time.sleep(poll)
        elapsed = (time.time() - t0) * 1e3
        hb = read_word_from_device(self.coord, DB_HB, device_id=self.device_id, context=self.ctx)
        out_words = read_words_from_device(self.coord, MM.PERF_OUTPUT, device_id=self.device_id,
                                           word_count=self._out_words, context=self.ctx)
        c_dev = MM.untilize32(self._decode(out_words))
        res = {"ring": r, "done": done, "done_ok": (done == r), "elapsed_ms": elapsed, "hb": hb,
               "c_dev": c_dev, "coord": str(self.coord)}
        if verify_golden:
            c_gold = MM.matmul_golden(a, b)
            mism = [(i, c_gold[i], c_dev[i]) for i in range(MM.TILE_ELEMS)
                    if float(c_gold[i]) != c_dev[i]]
            res["bit_exact"] = (len(mism) == 0)
            res["mismatches"] = len(mism)
            res["sample"] = mism[:6]
            res["corner_gold"] = c_gold[0]
            res["corner_dev"] = c_dev[0]
        return res

    def ring(self, a, b, *, timeout=4.0, poll=0.005, b_prestaged=False, verify_golden=True):
        """Stage A,B, ring, poll DONE, read + untilize PERF_OUTPUT (serial convenience wrapper)."""
        self.ring_async(a, b, b_prestaged=b_prestaged)
        return self.collect(timeout=timeout, poll=poll, verify_golden=verify_golden)

    def close(self):
        """Park the cores (assert soft-reset). Safe to call even if boot failed."""
        if not getattr(self, "_rdbg", None):
            return
        for c in _COMPONENTS:
            if c in self._rdbg:
                try:
                    self._rdbg[c].set_reset_signal(True)
                except Exception:
                    pass
        self._booted = False
