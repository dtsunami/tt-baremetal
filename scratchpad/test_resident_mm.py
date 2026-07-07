"""R1 keystone: prove RESIDENT 3-thread Tensix doorbell matmul on silicon.

Boot the resident kernel ONCE, then ring it 3x with DIFFERENT operands (no reload between rings) and
verify each result is bit-exact vs the pure-Python integer golden. If this passes, Tensix 3-thread
doorbell residency works => the 120-worker resident render grid is engineering, not an unknown.

Run: ~/bhtop/.venv/bin/python ~/bhtop/scratchpad/test_resident_mm.py
"""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix.resident import ResidentMatmul
from bhtop.tensix import matmul as MM


def gen_operands(seed):
    """Two distinct 32x32 int matrices, values in [0,15] (inside the bf16-exact window <=63)."""
    a = [((i * 7 + k * 3 + seed) % 16) for i in range(32) for k in range(32)]
    b = [((k * 5 + j * 2 + seed * 3) % 16) for k in range(32) for j in range(32)]
    return a, b


def main():
    ctx = init_ttexalens()
    L = TensixLauncher.at(1, 2, ctx=ctx)
    rm = ResidentMatmul(L.coord, ctx=ctx, out_format="fp32")
    print(f"[R1] booting resident_mm_perf on {rm.coord} (out=fp32) ...")
    rm.boot()
    time.sleep(0.3)  # let all 3 threads reach INIT + doorbell spin

    hb0 = None
    results = []
    try:
        for n in range(1, 4):
            a, b = gen_operands(n)
            r = rm.ring(a, b, timeout=4.0)
            results.append(r)
            print(f"[R1] ring {n}: done={r['done']} done_ok={r['done_ok']} "
                  f"hb={r['hb']} elapsed={r['elapsed_ms']:.1f}ms "
                  f"bit_exact={r.get('bit_exact')} mism={r.get('mismatches')} "
                  f"C[0,0] gold={r.get('corner_gold')} dev={r.get('corner_dev')}")
            if not r["done_ok"]:
                print(f"[R1] !! ring {n} did NOT complete (kernel may be wedged) sample={r.get('sample')}")
                break
            if not r.get("bit_exact"):
                print(f"[R1] !! ring {n} not bit-exact, sample={r.get('sample')}")
    finally:
        rm.close()

    n_ok = sum(1 for r in results if r.get("done_ok") and r.get("bit_exact"))
    hbs = [r["hb"] for r in results]
    resident = (len(results) >= 2 and all(r["done_ok"] for r in results)
                and hbs == sorted(hbs) and len(set(hbs)) == len(hbs))
    print(f"\n[R1] {n_ok}/{len(results)} rings bit-exact; heartbeats={hbs} (monotone distinct => "
          f"single resident kernel serviced every ring, NO reload)")
    verdict = (n_ok == 3 and resident)
    print(f"[R1] RESIDENT 3-THREAD TENSIX DOORBELL MATMUL: {'PASS' if verdict else 'CHECK'}")
    return verdict


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
