"""GAP-0 A/B: multi-ring residency of resident_train_perf on ONE boot (no per-ring reboot).

Boot the fused training kernel ONCE, then drive rings 1,2,3 back-to-back. Each ring re-poisons the
outputs so a stalled ring can't masquerade as passing on ring-1's stale data, and verifies fwd RGB +
leaf grads against the exact-float golden. On a stall (DONE != ring after the timeout) it takes an
llk_triage-style per-RISC snapshot (PC sampled twice 50ms apart -> advancing vs stuck, plus
is_halted/is_in_reset/is_ebreak) of UNPACK/MATH/PACK and prints pack's FLAG scoreboard, then STOPS —
no unbounded host spin. NoC0-safe: only L1 + risc_debug of the one worker; recover a wedge with
`tt-smi -r 0`.

  E1 = run against the CURRENT source (reset present)  -> expect ring 1 PASS, ring>=2 STALL
  E2 = run after deleting the 4 ring-boundary reset lines -> expect rings 1..3 all PASS bit-exact

Run: /home/starboy/bhtop/.venv/bin/python /home/starboy/bhtop/scratchpad/test_gap0_ab.py [NRINGS]
"""
import sys, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import (read_word_from_device as rd, read_words_from_device as rds,
                                      write_words_to_device as wr)
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run, splat as SP
from bhtop.tensix.resident import boot_resident

NRINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RING_TIMEOUT = 4.0
DB, DONE, HB, DBG_U, DBG_M, DBG_P = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050
FLAG_BASE, NST = 0x16060, 27
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, O_dLdop, O_dLdpsi = 0x42800, 0x51000, 0x52000
POISON = 0xBADF00D5
K, SIZE, P = 16, 16, 32


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
def mmg(A, B): return MM.matmul_golden(pad(A), pad(B))
def sub(flat, rows, cols): return [[flat[r * 32 + c] for c in range(cols)] for r in range(rows)]
def relerr(dev, gr, rows, cols):
    n = d = 0.0
    for r in range(rows):
        for c in range(cols):
            n += abs(dev[r * 32 + c] - gr[r][c]); d += abs(gr[r][c])
    return n / (d + 1e-12)


def golden():
    gs = SP.scene_rgb(k=K, seed=5, span=float(SIZE))
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]; Iden = [Mcomb[K + r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    pixels = [(x, y) for y in range(SIZE) for x in range(SIZE)][:P]
    phi_g = [[float(x), float(y), 1.0] for (x, y) in pixels]
    V = mmg(phi_g, psi_rows); Vsq = [v * v for v in V]
    E = mmg(sub(Vsq, 32, 32), Ppair); ar = [math.exp(min(e, 80.0)) for e in E]
    alpha = mmg(sub(ar, 32, 32), Dop); nalpha = mmg(sub(ar, 32, 32), Dnop)
    lpa = [math.log(a) if a > 1e-30 else -80.0 for a in alpha]
    la = [math.log1p(na) if na > -0.999999 else -80.0 for na in nalpha]
    logw = [a + b for a, b in zip(mmg(sub(la, 32, K), Stri), mmg(sub(lpa, 32, K), Iden))]
    w = [math.exp(min(x, 80.0)) for x in logw]
    C = mmg(sub(w, 32, 32), color); Cg = sub(C, P, 3)
    gt = [[0.7 * Cg[p][c] for c in range(3)] for p in range(P)]
    wg = sub(w, P, K); ag = sub(alpha, P, K); arg = sub(ar, P, K); vg = sub(V, P, 2 * K)
    dLdCg = [[Cg[p][c] - gt[p][c] for c in range(3)] for p in range(P)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2 = [[2.0 * x, 2.0 * y, 2.0] for (x, y) in pixels]; phi2T = [[phi2[p][r] for p in range(P)] for r in range(3)]
    dw = sub(mmg(dLdCg, colorT), P, K)
    dwW = [[dw[p][k] * wg[p][k] for k in range(K)] for p in range(P)]
    suf = sub(mmg(dwW, U), P, K)
    recA = [[1.0 / ag[p][k] for k in range(K)] for p in range(P)]
    Tv = [[wg[p][k] * recA[p][k] for k in range(K)] for p in range(P)]
    oneMa = [[1.0 - ag[p][k] for k in range(K)] for p in range(P)]
    recOM = [[1.0 / oneMa[p][k] for k in range(K)] for p in range(P)]
    t1 = [[dw[p][k] * Tv[p][k] for k in range(K)] for p in range(P)]
    t2 = [[suf[p][k] * recOM[p][k] for k in range(K)] for p in range(P)]
    dLda = [[t1[p][k] - t2[p][k] for k in range(K)] for p in range(P)]
    dae = [[dLda[p][k] * arg[p][k] for k in range(K)] for p in range(P)]
    dLdop_g = [[sum(dae[p][k] for p in range(P)) for k in range(K)]]
    dLdE = [[dae[p][k] * op[k] for k in range(K)] for p in range(P)]
    dLdVsq = sub(mmg(dLdE, PpairT), P, 2 * K)
    dLdV = [[dLdVsq[p][c] * vg[p][c] for c in range(2 * K)] for p in range(P)]
    dLdpsi_g = sub(mmg(phi2T, dLdV), 3, 2 * K)
    return dict(psi_rows=psi_rows, Ppair=Ppair, Dop=Dop, Dnop=Dnop, Stri=Stri, color=color, gt=gt,
                op=op, colorT=colorT, PpairT=PpairT, U=U, phi2T=phi2T, phi_g=phi_g,
                Cg=Cg, dLdop_g=dLdop_g, dLdpsi_g=dLdpsi_g)


def snap_riscs(rdbg):
    """llk_triage-style: PC x2 (50ms) + halted/reset/ebreak per thread."""
    def one(h):
        s = {}
        try: s["reset"] = h.is_in_reset()
        except Exception as e: s["reset"] = f"err:{e}"; return s
        if s["reset"]: return s
        for f, call in (("halted", h.is_halted), ("ebreak", h.is_ebreak_hit), ("pc", h.get_pc)):
            try: s[f] = call()
            except Exception as e: s[f] = f"err:{e}"
        return s
    a = {c: one(rdbg[c]) for c in ("UNPACK", "MATH", "PACK")}
    time.sleep(0.05)
    for c in ("UNPACK", "MATH", "PACK"):
        h = rdbg[c]
        if a[c].get("reset") is True: a[c]["pc2"] = None; continue
        try: a[c]["pc2"] = h.get_pc()
        except Exception as e: a[c]["pc2"] = f"err:{e}"
    return a


def fmt_snap(a):
    out = []
    for c in ("UNPACK", "MATH", "PACK"):
        s = a[c]
        if s.get("reset") is True: out.append(f"    {c:<6} in_reset=True"); continue
        pc, pc2 = s.get("pc"), s.get("pc2")
        mv = ("advancing" if isinstance(pc, int) and isinstance(pc2, int) and pc != pc2
              else "STUCK" if isinstance(pc, int) and pc == pc2 else "?")
        pcs = f"0x{pc:x}" if isinstance(pc, int) else str(pc)
        out.append(f"    {c:<6} halted={s.get('halted')} ebreak={s.get('ebreak')} pc={pcs} [{mv}]")
    return "\n".join(out)


def main():
    pr = lambda *a: print(*a, flush=True)
    g = golden()
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    pr(f"[gap0] exalens up; worker={coord}; driving {NRINGS} rings on ONE boot (no reboot)")

    b = llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    assert b["ok"], b["log"][-2000:]
    rdbg = boot_resident("resident_train_perf", coord, ctx=ctx,
                         runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=80)
    time.sleep(0.3)
    pr("[gap0] booted ONCE")

    def st(name, rc): wr(coord, H[name], enc(pad(rc)), context=ctx)
    st("phi", g["phi_g"]); st("psi", g["psi_rows"]); st("Ppair", g["Ppair"]); st("Dop", g["Dop"])
    st("Dnop", g["Dnop"]); st("Stri", g["Stri"]); st("color", g["color"]); st("gt", g["gt"])
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    opB_pad = [[(g["op"][k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    wr(coord, H["opB"], enc(pad(opB_pad)), context=ctx)
    st("colorT", g["colorT"]); st("PpairT", g["PpairT"]); st("U", g["U"]); st("phi2T", g["phi2T"])
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx); st("ones1P", [[1.0] * P])
    pr("[gap0] operands staged once\n")

    results = []
    for r in range(1, NRINGS + 1):
        # poison outputs so a stalled ring can't pass on stale ring-(r-1) data
        for a in (S_C, O_dLdop, O_dLdpsi):
            wr(coord, a, [POISON] * 512, context=ctx)
        wr(coord, DBG_M, [0], context=ctx); wr(coord, DBG_P, [0], context=ctx)
        wr(coord, DB, [r], context=ctx)
        t0 = time.time()
        while time.time() - t0 < RING_TIMEOUT and rd(coord, DONE, context=ctx) != r:
            time.sleep(0.004)
        done = rd(coord, DONE, context=ctx)
        dt = time.time() - t0
        if done != r:
            flags = rds(coord, FLAG_BASE, word_count=NST, context=ctx)
            pack_stage = 1 + max([i for i in range(NST) if flags[i] == r], default=-1)
            m, p, u = rd(coord, DBG_M, context=ctx), rd(coord, DBG_P, context=ctx), rd(coord, DBG_U, context=ctx)
            mk, sem, pk = rd(coord, 0x16100, context=ctx), rd(coord, 0x16104, context=ctx), rd(coord, 0x16108, context=ctx)
            uk = rd(coord, 0x1610C, context=ctx)
            pr(f"[ring {r}] *** STALL *** DONE={done} (want {r}) after {dt:.1f}s  HB={rd(coord,HB,context=ctx)}")
            pr(f"          DBG_U=0x{u:x}  DBG_M=0x{m:x}  DBG_P=0x{p:x}  pack FLAG scoreboard reached stage {pack_stage}/{NST}")
            pr(f"          MK_PH=0x{mk:x} (r{mk>>16} s{(mk>>8)&0xff} sub{mk&0xff})  SEM_M={sem}  "
               f"PK_PH=0x{pk:x} (r{pk>>16} s{(pk>>8)&0xff} sub{pk&0xff})  UK_PH=0x{uk:x} (r{uk>>16} s{(uk>>8)&0xff} sub{uk&0xff})")
            pr(fmt_snap(snap_riscs(rdbg)))
            results.append((r, "STALL", None))
            pr("[gap0] stopping (no unbounded host spin). Recover kernel with: tt-smi -r 0")
            break
        Cdev = dec(S_C, ctx, coord)
        if Cdev[0] == 0.0 and all(Cdev[p * 32 + c] == 0.0 for p in range(P) for c in range(3)):
            pr(f"[ring {r}] DONE but output all-zero/stale -> CHECK"); results.append((r, "CHECK", None)); continue
        mse = sum((Cdev[p * 32 + c] - g["Cg"][p][c]) ** 2 for p in range(P) for c in range(3)) / (P * 3)
        psnr = 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)
        e_op = relerr(dec(O_dLdop, ctx, coord), g["dLdop_g"], 1, K)
        e_psi = relerr(dec(O_dLdpsi, ctx, coord), g["dLdpsi_g"], 3, 2 * K)
        ok = psnr >= 35.0 and e_op < 0.2 and e_psi < 0.2
        results.append((r, "PASS" if ok else "CHECK", (psnr, e_psi, e_op)))
        ring_cyc, cfg_cyc = rd(coord, 0x16114, context=ctx), rd(coord, 0x16118, context=ctx)
        pr(f"[ring {r}] DONE in {dt:.2f}s  PSNR={psnr:5.2f}dB  dLdpsi={e_psi:.2e}  dLdop={e_op:.2e}  "
           f"-> {'PASS' if ok else 'CHECK'}   [telem: ring={ring_cyc} cyc, reinit={cfg_cyc} cyc]")

    pr("\n[gap0] ===== SUMMARY =====")
    for r, verdict, metr in results:
        extra = f"  PSNR={metr[0]:.2f}dB psi={metr[1]:.1e} op={metr[2]:.1e}" if metr else ""
        pr(f"   ring {r}: {verdict}{extra}")
    n_pass = sum(1 for _, v, _ in results if v == "PASS")
    resident_ok = (len(results) == NRINGS and n_pass == NRINGS)
    pr(f"[gap0] {n_pass}/{NRINGS} rings PASS on ONE boot  =>  "
       f"{'MULTI-RING RESIDENT ✓ (reboot workaround can be removed)' if resident_ok else 'NOT resident (stall reproduced)'}")
    return resident_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
