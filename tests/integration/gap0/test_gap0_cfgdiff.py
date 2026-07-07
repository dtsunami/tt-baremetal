"""GAP-0 CONFIG-DRIFT TRACER (host-side, via exalens get_tensix_state) — the read the x280 tracer will do.

resident_mm_elw_perf stalls deterministically at ring 5. The accumulator survives full pipeline drains, so
it's a persistent CONFIG register. This snapshots the worker Tensix's FULL config state at the quiescent
point (after DONE=r, threads parked on wait_ring) after a clean boot and after each healthy ring 1..4, then
DIFFS field-by-field to find the register(s) that drift MONOTONICALLY across rings = the accumulator.

Run: /home/starboy/bhtop/.venv/bin/python scratchpad/test_gap0_cfgdiff.py
"""
import sys, time, dataclasses
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens import tt_exalens_lib as L
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident

LOC = "1,2"
DB, DONE = 0x16000, 0x16010
A_ADDR, B_ADDR, D_ADDR, OUT = 0x21000, 0x31000, 0x61000, 0x51000
enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])


def flatten(prefix, obj, out):
    """Flatten a get_tensix_state dataclass/list/dict into {path: int} leaves."""
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            flatten(f"{prefix}.{f.name}", getattr(obj, f.name), out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}[{k}]", v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(f"{prefix}[{i}]", v, out)
    else:
        try:
            out[prefix] = int(obj)
        except (TypeError, ValueError):
            out[prefix] = obj


def snapshot(ctx):
    st = L.get_tensix_state(LOC, device_id=0)
    out = {}
    flatten("cfg", st, out)
    return out


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord

    A = [((i * 7 + k * 3) % 13) * 0.1 for i in range(32) for k in range(32)]
    B = [((k * 5 + j * 2) % 11) * 0.1 for k in range(32) for j in range(32)]
    D = [((i + j) % 7) * 0.1 + 0.05 for i in range(32) for j in range(32)]

    ov = {"ELTWISE_BINARY_OP": "constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
    b = llk_run.build("resident_mm_elw_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False, overrides=ov)
    assert b["ok"], b["log"][-2000:]
    boot_resident("resident_mm_elw_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=48)
    time.sleep(0.3)
    wr(coord, A_ADDR, enc(A), context=ctx); wr(coord, B_ADDR, enc(B), context=ctx); wr(coord, D_ADDR, enc(D), context=ctx)
    pr("[cfgdiff] booted + staged\n")

    snaps = {}
    snaps[0] = snapshot(ctx)   # clean, before ring 1
    for r in range(1, 5):      # rings 1..4 are all healthy (stall is ring 5)
        wr(coord, DB, [r], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != r:
            time.sleep(0.004)
        assert rd(coord, DONE, context=ctx) == r, f"ring {r} did not complete (unexpected)"
        time.sleep(0.02)       # let threads settle at wait_ring
        snaps[r] = snapshot(ctx)
        pr(f"[cfgdiff] ring {r} done + snapshot ({len(snaps[r])} config leaves)")

    keys = sorted(snaps[0].keys())
    pr(f"\n[cfgdiff] diffing {len(keys)} config leaves across clean(0) + rings 1..4")
    pr("[cfgdiff] === registers that CHANGE across the snapshots ===")
    n_drift = 0
    for k in keys:
        vals = [snaps[r].get(k) for r in (0, 1, 2, 3, 4)]
        if not all(isinstance(v, int) for v in vals):
            continue
        if len(set(vals)) == 1:
            continue   # constant -> not the accumulator
        # classify: monotonic drift vs toggling
        deltas = [vals[i + 1] - vals[i] for i in range(4)]
        ring_deltas = deltas[1:]   # ring1->2, 2->3, 3->4 (steady-state, exclude boot->ring1)
        monotonic = all(d == ring_deltas[0] and d != 0 for d in ring_deltas)
        tag = " <<< MONOTONIC DRIFT" if monotonic else ("  (toggles)" if len(set(vals)) == 2 else "  (varies)")
        pr(f"    {k:52s} {vals}{tag}")
        n_drift += 1
    pr(f"\n[cfgdiff] {n_drift} config leaves changed; MONOTONIC-DRIFT ones (steady per-ring delta) are the accumulator.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
