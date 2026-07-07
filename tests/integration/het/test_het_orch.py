"""x280-ORCHESTRATED single-tile loop — the host is OUT of the per-tile control path. Per step the host rings
just 3 het doorbells (cmd2 project, cmd7 orchestrate, cmd1 Adam) + reads 1 loss. cmd7 produces operands,
raises the worker's flag in x280 GDDR, and waits its ack; the RESIDENT conductor on the worker's BRISC polls
that flag, reads operands, drives the render for 8 groups (copying pre-staged phi/gt), writes grads back, and
acks. No per-group host rings, no cb_io host rings. Convergence proves the orchestrated path end-to-end."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coord
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord
import gap1_proj_golden as PG

K, SIZE, P, STEPS = 12, 16, 32, (int(sys.argv[1]) if len(sys.argv) > 1 else 15)
WX, WY = 11, 2; HUB = (8, 3)
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/het_x280.c"
H = dict(Ppair=0x23000, Stri=0x26000, Iden=0x27000, PpairT=0x2C000, U=0x2D000, ones=0x2F000, ones1P=0x30000)
X_HDR, X_CAM, X_IDL, X_LOSS, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x300050A0, 0x30005B00, 0x30004000, 0x30004010, 0x30004020
PARAM, OPBASE, GINORCH = 0x30100000, 0x31000000, 0x33000000    # orch-respaced (slot stride 0x10000, no collide)
FLAG, ACK = 0x30006400, 0x30006800   # per-slot 0x40 stride (own 64B line — avoid concurrent-write granule race)
TGT_IMG, PXBASE = 0x30200000, 0x32000000              # x280 GDDR: resident target image (SZ*SZ*3 f32) +
                                                       # produced phi/phi2T/gt tiles (het cmd8). NO host pixel data.
CFG = 0x3200                                          # conductor cfg block (worker L1)
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 26.0; cx = cy = 8.0
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bff = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
grp = [[(x, y) for y in range(SIZE) for x in range(SIZE)][i:i + 32] for i in range(0, SIZE * SIZE, 32)]


def rand_scene3d(seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 0.7, (K, 3)); sl = rng.normal(-1.9, 0.2, (K, 3))
    q = rng.normal(0, 1, (K, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (K, 1)); col = rng.uniform(0.1, 0.9, (K, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def gs_of(param):
    return [(*PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rv, tv, fx, fy, cx, cy)[:1],)  # placeholder
            for o in range(K)]


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); coord = worker_coord(ctx, WX, WY); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass

    tgt = rand_scene3d(11); init = rand_scene3d(22)
    gsT = []
    for o in range(K):
        u, v, dep, a, b, c, _ = PG.project_forward(tgt[o, :3], tgt[o, 3:6], tgt[o, 6:10], Rv, tv, fx, fy, cx, cy)
        gsT.append((u, v, a, b, c, float(tgt[o, 10]), *[float(x) for x in tgt[o, 11:14]], dep))
    target = np.array(SP._golden_render(gsT, SIZE)).reshape(SIZE, SIZE, 3)

    # render kernel + static operands (once)
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    boot_resident("resident_train_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    Ppair = [[0.0] * K for _ in range(2 * K)]
    for i in range(K): Ppair[2 * i][i] = -0.5; Ppair[2 * i + 1][i] = -0.5
    Mcomb = [[(1.0 if r < c else 0.0) for c in range(K)] for r in range(2 * K)]
    for i in range(K): Mcomb[K + i][i] = 1.0
    Stri = [Mcomb[r] for r in range(K)]; PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    for nm, m in [("Ppair", Ppair), ("Stri", Stri), ("PpairT", PpairT), ("U", U)]:
        wr(coord, H[nm], enc(pad(m)), context=ctx)
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx); wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)

    # DATA PLANE ON-DEVICE: the target image lives resident in x280 GDDR; het cmd8 computes phi/phi2T from the
    # tile origin and gathers gt from it, tilizing all 8 groups into PXBASE. NO host pixel/target staging.
    dev.wr(0, TGT_IMG, [fb(v) for v in target.reshape(-1)])          # resident SZ*SZ*3 f32 target (once)

    # het hub + resident params + zeroed Adam
    dev.wr(0, X_HDR, [K, 0]); dev.wr(0, X_HDR + 21 * 4, [0])
    dev.wr(0, X_HDR + 22 * 4, [0, 0])                                # tile origin ox,oy (hdr[24]=X_CAM: DON'T touch)
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(init[o, j]) for o in range(K) for j in range(14)])
    M = PARAM + K * 14 * 4; V = M + K * 14 * 4; GACC = V + K * 14 * 4
    dev.wr(0, M, [0] * (K * 14)); dev.wr(0, V, [0] * (K * 14)); dev.wr(0, GACC, [0] * (K * 9))
    dev.wr(0, FLAG, [0]); dev.wr(0, ACK, [0]); dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0])
    dev.wr(0, 0x300027F0, [1])   # NHARTS_A=1: single-tile runs cmd2/cmd1 on hart 0 only (no worker harts booted)
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
        time.sleep(0.03)
    else: pr("FAIL: het not resident"); return

    # RESIDENT conductor on worker BRISC (config: hub, slot, opbase, ginbase, flag, ack, PXBASE, -, -, ng)
    wr(coord, CFG, [bm_coord(*HUB), 0, OPBASE, GINORCH, FLAG, ACK, PXBASE, 0, 0, 8], context=ctx)
    BareMetal(WX, WY, ctx=ctx, risc="brisc").run(BareMetal.build("conductor"))
    time.sleep(0.2)
    # NB: the conductor's cold-start local-store ring wakes all 3 render threads on the FIRST ring — no host
    # "prime" needed. (Earlier stall was the pre-stage scratch clobbering the TRISC code, NOT ring coherency.)
    pr(f"[setup] het hub + render + RESIDENT conductor on ({WX},{WY}) — x280 orchestrates, host rings 3/step")

    def het(cmd, extra=None):
        if extra: dev.wr(0, X_HDR, extra)
        dev.wr(0, X_CMD, [cmd]); r = dev.rd(0, X_DB) + 1; dev.wr(0, X_DB, [r])
        t = time.time()
        while dev.rd(0, X_DONE) != r and time.time() - t < 8.0: time.sleep(0.0005)

    het(8)   # ONCE: x280 produces phi/phi2T (tile origin) + gt (resident image) -> PXBASE; fixed tile => constant
    pr(f"[setup] pixel/target data produced ON-DEVICE (het cmd8 -> PXBASE 0x{PXBASE:x}); host stages NO pixel data")
    # VERIFY cmd8 output vs host reference (group 0): phi/phi2T/gt bit-compare
    g0 = grp[0]
    exp = {"phi": enc(pad([[float(x), float(y), 1.0] for (x, y) in g0])),
           "phi2T": enc(pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g0] for r in range(3)])),
           "gt": enc(pad([[target[g0[p][1], g0[p][0]][ch] for ch in range(3)] for p in range(len(g0))]))}
    for j, nm in enumerate(("phi", "phi2T", "gt")):
        got = list(dev.rdn(0, PXBASE + j * 0x800, 512)); mism = [(i, exp[nm][i], got[i]) for i in range(512) if exp[nm][i] != got[i]]
        pr(f"    [verify cmd8] {nm:5s} mism={len(mism)}/512  first5={mism[:5]}")

    LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
    for step in range(1, STEPS + 1):
        t0 = time.time()
        het(2, [K, step]); dev.wr(0, X_HDR + 21 * 4, [0]); dev.wr(0, X_IDL, [K] + list(range(K)))
        het(7)                                             # ORCHESTRATE (produce+signal+wait+consume, host-free)
        if step <= 2:
            cdbg = list(rds(coord, 0x2100, word_count=9, context=ctx))
            htel = dev.telemetry(0, slots=6, hart=0)
            fl = dev.rd(0, FLAG); ak = dev.rd(0, ACK)
            rDB, rDONE, rHB = (rds(coord, a, word_count=1, context=ctx)[0] for a in (0x16000, 0x16010, 0x16020))
            u, m, p, ukp, mkp, pkp = (rds(coord, a, word_count=1, context=ctx)[0] for a in (0x16030, 0x16040, 0x16050, 0x1610C, 0x16100, 0x16108))
            pr(f"    [diag] conductor dbg(flag,last,nflag,g,rdone,ring,ack,DBwr)={cdbg[:8]}  het cmd7 cyc={htel[2]:,}  TELE[ack,ring]={htel[4:6]}")
            pr(f"    [diag] render L1 (host-view via NoC): DB@16000={rDB}  DONE@16010={rDONE}  HB@16020={rHB}  <- HB bumps every render loop iter")
            pr(f"    [diag] render threads: DBG_U={u:#x} DBG_M={m:#x} DBG_P={p:#x} | UK_PH={ukp:#x} MK_PH={mkp:#x} PK_PH={pkp:#x}  (0=thread never ran)")
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        het(1, [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR])
        loss = bff(rds(1, X_LOSS, word_count=1)[0] if False else dev.rdn(0, X_LOSS, 1)[0])
        psnr = 99.0 if loss < 1e-9 else 10 * math.log10((SIZE * SIZE * 3) / loss)
        pr(f"  step {step:2d}: loss={loss:.4f}  ~PSNR={psnr:5.2f} dB  ({time.time()-t0:.2f}s)  [host: 3 doorbells]")
    pr("done — x280-ORCHESTRATED: host issued 3 doorbells/step, conductor drove the whole tile on-device")


if __name__ == "__main__":
    main()
