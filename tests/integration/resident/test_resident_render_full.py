"""Full-tile fused resident render + grid (one ring per tile). Boot resident_render_perf, stage phi
once + consts per tile, ring ONCE per tile (kernel loops all pixel-groups internally). Single worker +
N-worker parallel grid. Uses the het.render_resident driver.

Run: cd ~/bhtop && .venv/bin/python scratchpad/test_resident_render_full.py [N_workers]
"""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher, worker_coords
from bhtop.het import render_resident as RR


def main(N=8):
    ctx = init_ttexalens()
    RR.build(fp32_acc=False)
    consts, phis, groups, gold, gs, order = RR.build_operands()

    # ---- single worker, one ring/tile, with telemetry ----
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    RR.boot(coord, ctx=ctx, phis=phis, ng=len(groups), consts=consts)
    time.sleep(0.05)
    rgb, t = RR.render_tile(coord, ctx=ctx, groups=groups, size=16, telem=True)
    p1 = RR.psnr(rgb, gold)
    print(f"[full] single worker {coord}: 1 ring/tile PSNR={p1:.1f} dB | tile={t['tile_cycles']} dev cyc "
          f"({t['groups']} groups), host {t['host_ms']:.1f} ms")

    # ---- grid: same tile on N workers, one ring each, in parallel ----
    ws = worker_coords(ctx)[:N]
    for w in ws:
        RR.boot(w, ctx=ctx, phis=phis, ng=len(groups), consts=consts)
    time.sleep(0.1)
    from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
    t0 = time.time()
    rings = {}
    for w in ws:                                             # ring ALL workers (parallel render)
        r = rd(w, RR.DONE, context=ctx) + 1
        wr(w, RR.DB, [r], context=ctx); rings[str(w)] = r
    psnrs = []
    for w in ws:                                            # then collect each
        t1 = time.time()
        while time.time() - t1 < 6.0 and rd(w, RR.DONE, context=ctx) != rings[str(w)]:
            time.sleep(0.004)
        rgb_w = RR._collect_rgb(w, ctx=ctx, groups=groups) if hasattr(RR, "_collect_rgb") else None
        # inline collect (driver has no standalone collector): read NG output tiles
        from bhtop.tensix import matmul as MM
        from ttexalens.tt_exalens_lib import read_words_from_device as rds
        rgbw = [[0.0, 0.0, 0.0] for _ in range(256)]; pp = 0
        for gi, g in enumerate(groups):
            c = MM.untilize32(MM.unpack_bf16_words(rds(w, RR.OUT_C + gi * RR.STRIDE, word_count=512, context=ctx)))
            for rr in range(len(g)):
                rgbw[pp] = [c[rr * 32], c[rr * 32 + 1], c[rr * 32 + 2]]; pp += 1
        psnrs.append(RR.psnr(rgbw, gold))
    dt = time.time() - t0
    ok = (p1 >= 40.0) and all(p >= 40.0 for p in psnrs)
    print(f"[full] {N}-worker grid (1 ring each, parallel): PSNRs={['%.1f'%p for p in psnrs]} "
          f"({dt*1e3:.0f}ms for {N} tiles)")
    print(f"[full] FUSED RESIDENT RENDER {'PASS' if ok else 'CHECK'} (1 ring/tile, single + {N}-worker grid)")
    return ok


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sys.exit(0 if main(N) else 1)
