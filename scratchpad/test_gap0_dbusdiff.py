"""GAP-0 DEBUG-BUS DRIFT TRACER — config regs were all constant, so the accumulator is non-config backend
state (src bank pointers / dvalid / RWC / dest scoreboard). Snapshot the backend debug-bus signals at the
quiescent point after clean boot + rings 1..4 and diff for MONOTONIC drift = the accumulator.

Run: /home/starboy/bhtop/.venv/bin/python scratchpad/test_gap0_dbusdiff.py
"""
import sys, time, re
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident

DB, DONE = 0x16000, 0x16010
A_ADDR, B_ADDR, D_ADDR = 0x21000, 0x31000, 0x61000
enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
# backend-state signals; exclude per-instruction bits (payload/opcode/pc) that vary meaninglessly
KEEP = re.compile(r"(^rwc|^adcs|srca|srcb|dest_reg_deps|_vld$|dvalid|bank|reg_addr|_cr$|winner|data_ready)", re.I)
DROP = re.compile(r"(payload|opcode|_pc$|instrn_vld|icache|noc_ctrl|mailbox|dbg_obs|perf_cnt|ibuffer|lsq_head|rq_head)", re.I)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    db = ctx.devices[0].get_block(coord).get_debug_bus()
    sigs = list(db.signal_names)   # ALL signals — the accumulator wasn't in the filtered subset
    pr(f"[dbus] {len(sigs)} signals (full sweep)")

    A = [((i * 7 + k * 3) % 13) * 0.1 for i in range(32) for k in range(32)]
    B = [((k * 5 + j * 2) % 11) * 0.1 for k in range(32) for j in range(32)]
    D = [((i + j) % 7) * 0.1 + 0.05 for i in range(32) for j in range(32)]
    ov = {"ELTWISE_BINARY_OP": "constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
    b = llk_run.build("resident_mm_elw_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False, overrides=ov)
    assert b["ok"], b["log"][-2000:]
    boot_resident("resident_mm_elw_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=48)
    time.sleep(0.3)
    wr(coord, A_ADDR, enc(A), context=ctx); wr(coord, B_ADDR, enc(B), context=ctx); wr(coord, D_ADDR, enc(D), context=ctx)
    pr("[dbus] booted + staged\n")

    def snap():
        s = {}
        for n in sigs:
            try: s[n] = int(db.read_signal(n))
            except Exception: pass
        return s

    snaps = {0: snap()}
    for r in range(1, 5):
        wr(coord, DB, [r], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != r:
            time.sleep(0.004)
        assert rd(coord, DONE, context=ctx) == r, f"ring {r} did not complete"
        time.sleep(0.02)
        snaps[r] = snap()
        pr(f"[dbus] ring {r} done + snapshot")

    common = set.intersection(*[set(snaps[r]) for r in range(5)])
    pr(f"\n[dbus] diffing {len(common)} readable signals across clean(0)+rings 1..4")
    # SHARPEST detector: ring 1 & 3 & 5 complete; ring 5 wedges. Snapshots [0],[2],[4] are the EVEN phase
    # (entering rings 1,3,5). A signal whose even-phase values are NOT all equal drifts within-phase = the
    # accumulator that differs between ring-1-start and ring-5-start.
    samephase, mono, other = [], [], []
    for k in sorted(common):
        vals = [snaps[r][k] for r in (0, 1, 2, 3, 4)]
        if len(set(vals)) == 1:
            continue
        even = [vals[0], vals[2], vals[4]]
        d = [vals[i + 1] - vals[i] for i in range(1, 4)]
        if len(set(even)) > 1:
            samephase.append((k, vals))
        elif all(x == d[0] and x != 0 for x in d):
            mono.append((k, vals))
        else:
            other.append((k, vals))
    pr(f"\n[dbus] *** SAME-PHASE DRIFT ({len(samephase)}) — differs ring1-start vs ring5-start = ACCUMULATOR:")
    for k, vals in samephase: pr(f"    {k:50s} {vals}")
    pr(f"\n[dbus] monotonic-drift ({len(mono)}):")
    for k, vals in mono: pr(f"    {k:50s} {vals}")
    pr(f"\n[dbus] other varying ({len(other)}) [sample]:")
    for k, vals in other[:30]: pr(f"    {k:50s} {vals}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
