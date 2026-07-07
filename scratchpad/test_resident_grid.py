"""R3-core: the RESIDENT GRID — N Tensix workers each hold a resident doorbell kernel, driven in
PARALLEL by the host (ring them all, then collect them all), each bit-exact, none reloaded.

This is the orchestration mechanism the 120-worker resident render needs: one build, N boots, then the
host drives every worker per-tile by staging operands + ringing a doorbell (NO per-op ELF reload/reboot,
the cost that makes the host-orchestrated render serial + slow). Proves independent residency +
concurrent drive across the grid, on the R1-proven resident matmul kernel.

Run: ~/bhtop/.venv/bin/python ~/bhtop/scratchpad/test_resident_grid.py [N]
"""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import worker_coords
from bhtop.tensix.resident import ResidentMatmul


def operands(seed):
    a = [((i * 7 + k * 3 + seed) % 16) for i in range(32) for k in range(32)]
    b = [((k * 5 + j * 2 + seed * 3) % 16) for k in range(32) for j in range(32)]
    return a, b


def main(N=8, rounds=2):
    ctx = init_ttexalens()
    ws = worker_coords(ctx)
    coords = ws[:N]
    print(f"[grid] booting resident_mm_perf on {N} workers: {[str(c) for c in coords]}")

    workers = []
    t_boot = time.time()
    for idx, c in enumerate(coords):
        rm = ResidentMatmul(c, ctx=ctx, out_format="fp32")
        rm.boot(build=(idx == 0))     # build ELFs once; every worker reuses them
        workers.append(rm)
    print(f"[grid] {N} workers booted in {(time.time()-t_boot):.2f}s (build once, boot N)")
    time.sleep(0.3)

    ok_total = 0
    try:
        for rnd in range(1, rounds + 1):
            # PARALLEL drive: ring every worker (distinct operands), THEN collect every worker.
            t0 = time.time()
            for idx, rm in enumerate(workers):
                a, b = operands(seed=rnd * 100 + idx)
                rm.ring_async(a, b)
            t_ring = time.time() - t0
            results = [rm.collect(timeout=5.0) for rm in workers]
            t_all = time.time() - t0

            n_ok = sum(1 for r in results if r["done_ok"] and r.get("bit_exact"))
            ok_total += n_ok
            hbs = [r["hb"] for r in results]
            done_all = all(r["done_ok"] for r in results)
            print(f"[grid] round {rnd}: {n_ok}/{N} workers bit-exact | all_done={done_all} "
                  f"heartbeats={hbs} | ring-all {t_ring*1e3:.0f}ms, ring+collect {t_all*1e3:.0f}ms")
            bad = [r for r in results if not (r["done_ok"] and r.get("bit_exact"))]
            for r in bad[:3]:
                print(f"[grid]   !! {r['coord']}: done_ok={r['done_ok']} mism={r.get('mismatches')} "
                      f"sample={r.get('sample')}")
    finally:
        for rm in workers:
            rm.close()

    # heartbeats == rounds on every worker => each worker's ONE resident kernel serviced every round
    resident = all(r["hb"] == rounds for r in results)
    verdict = (ok_total == N * rounds and resident)
    print(f"\n[grid] {ok_total}/{N*rounds} (worker,round) results bit-exact; every worker heartbeat=="
          f"{rounds} => single resident kernel per worker across all rounds (NO reload)")
    print(f"[grid] RESIDENT {N}-WORKER GRID: {'PASS' if verdict else 'CHECK'}")
    return verdict


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ok = main(N=N)
    sys.exit(0 if ok else 1)
