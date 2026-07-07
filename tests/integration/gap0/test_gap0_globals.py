"""GAP-0 C-GLOBAL + REBOOT-TRIAGE tracer. Config + full debug bus are identical entering ring 1 vs ring 5,
so the accumulator must be RISC software state the ELF reload resets: the per-thread C-globals
(dest_offset_id / cfg_state_id / unp_cfg_context / math_sync_tile_dst_index / pack_sync_tile_dst_ptr) in
each RISC's data RAM. Read them per-RISC across rings 1..4, then REBOOT and re-read: whatever drifts across
rings and the reboot resets is the culprit (reboot is the known fix, so its delta IS the accumulator).

Run: /home/starboy/bhtop/.venv/bin/python scratchpad/test_gap0_globals.py
"""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident

DB, DONE = 0x16000, 0x16010
A_ADDR, B_ADDR, D_ADDR = 0x21000, 0x31000, 0x61000
enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
# per-RISC C-global addresses (base 0xffb00000), from nm of the built ELFs
G = {
    "trisc0": {"dest_off": 0xffb0001c, "cfg_sid": 0xffb00020, "math_dst": 0xffb00024, "pack_ptr": 0xffb00028, "unp_ctx": 0xffb0002c},
    "trisc1": {"dest_off": 0xffb00024, "cfg_sid": 0xffb00028, "math_dst": 0xffb00030, "pack_ptr": 0xffb00034, "unp_ctx": 0xffb00038},
    "trisc2": {"dest_off": 0xffb0001c, "cfg_sid": 0xffb00020, "math_dst": 0xffb00024, "pack_ptr": 0xffb00028, "unp_ctx": 0xffb0002c},
}
RUNTIME = [1, 1, 1, 1, 1, 128, 128, 0, 4, 4]


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    blk = ctx.devices[0].get_block(coord)
    rdbg = {t: blk.get_risc_debug(t) for t in ("trisc0", "trisc1", "trisc2")}

    A = [((i * 7 + k * 3) % 13) * 0.1 for i in range(32) for k in range(32)]
    B = [((k * 5 + j * 2) % 11) * 0.1 for k in range(32) for j in range(32)]
    D = [((i + j) % 7) * 0.1 + 0.05 for i in range(32) for j in range(32)]
    ov = {"ELTWISE_BINARY_OP": "constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
    b = llk_run.build("resident_mm_elw_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False, overrides=ov)
    assert b["ok"], b["log"][-2000:]

    def stage():
        wr(coord, A_ADDR, enc(A), context=ctx); wr(coord, B_ADDR, enc(B), context=ctx); wr(coord, D_ADDR, enc(D), context=ctx)

    def read_globals():
        out = {}
        for t, regs in G.items():
            for nm, addr in regs.items():
                try: out[f"{t}.{nm}"] = rdbg[t].read_memory(addr)
                except Exception as e: out[f"{t}.{nm}"] = f"err:{e}"
        return out

    def ring(r):
        wr(coord, DB, [r], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != r:
            time.sleep(0.004)
        return rd(coord, DONE, context=ctx) == r

    boot_resident("resident_mm_elw_perf", coord, ctx=ctx, runtime_words=RUNTIME, clear_words=48)
    time.sleep(0.3); stage()
    pr("[glob] booted + staged\n")

    snaps = {0: read_globals()}
    for r in range(1, 5):
        assert ring(r), f"ring {r} unexpectedly stalled"
        time.sleep(0.02); snaps[r] = read_globals()
        pr(f"[glob] ring {r} done")

    keys = sorted(snaps[0].keys())
    pr("\n[glob] === C-globals across clean(0) + rings 1..4 ===")
    for k in keys:
        vals = [snaps[r].get(k) for r in (0, 1, 2, 3, 4)]
        iv = [v for v in vals if isinstance(v, int)]
        drift = len(set(iv)) > 1
        pr(f"    {k:22s} {vals}{'   <<< CHANGES' if drift else ''}")

    # ---- reboot-triage: reboot (the known fix), re-read; the delta from ring-4 = what reboot resets ----
    pr("\n[glob] REBOOT (the known fix) — re-reading globals post-reboot:")
    boot_resident("resident_mm_elw_perf", coord, ctx=ctx, runtime_words=RUNTIME, clear_words=48)
    time.sleep(0.3)
    post = read_globals()
    for k in keys:
        pr(f"    {k:22s} ring4={snaps[4].get(k)}  post-reboot={post.get(k)}"
           f"{'   <<< REBOOT RESET' if snaps[4].get(k) != post.get(k) else ''}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
