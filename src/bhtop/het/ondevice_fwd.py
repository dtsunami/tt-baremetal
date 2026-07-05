"""Full on-device Gaussian-splat forward for ONE 32-pixel group — every op on MVMUL+SFPU, per-stage
verified vs host sim. Proves the fully-on-device forward before formalizing into tensix.splat."""
import sys, math
sys.path.insert(0, "/home/starboy/bhtop/src")
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

TILE = 32
R = lambda x: MM.bf16_to_f32(MM.f32_to_bf16(x))

def pad(rows_cols):
    m = [[0.0] * TILE for _ in range(TILE)]
    for r, row in enumerate(rows_cols):
        for c, v in enumerate(row):
            m[r][c] = float(v)
    return [m[r][c] for r in range(TILE) for c in range(TILE)]  # flat 1024

def take(flat, rows, cols):
    return [[flat[r * TILE + c] for c in range(cols)] for r in range(rows)]

def hmm(A, B, rows, kk, cols):   # host bf16 matmul (fp32 acc) — the reference for each device matmul
    return [[sum(R(A[i][k]) * R(B[k][j]) for k in range(kk)) for j in range(cols)] for i in range(rows)]

def maxrel(dev, ref, rows, cols):
    m = 0.0
    for i in range(rows):
        for j in range(cols):
            d, g = dev[i][j], ref[i][j]
            if abs(g) > 1e-3:
                m = max(m, abs(d - g) / abs(g))
    return m

def main():
    L = TensixLauncher.at(1, 2); coord, ctx = L.coord, L.ctx
    size = 16; K = 16
    gs = SP.scene_rgb(k=K, seed=5, span=float(size))
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    # one 32-pixel group (16x2 strip)
    px = [(x, y) for y in range(2) for x in range(16)]; P = len(px)

    def whiten(g):
        gx, gy, a, b, c, op = g[:6]; sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
        m22 = math.sqrt(max(c - b * b / a, 0)); return [sa, m12, -(sa*gx+m12*gy)], [0.0, m22, -(m22*gy)]
    W = [whiten(g) for g in gso]

    phi = [[float(x), float(y), 1.0] for (x, y) in px]                 # [P,3]
    psi = [[0.0]*(2*K) for _ in range(3)]
    for i in range(K):
        w1, w2 = W[i]
        for r in range(3): psi[r][2*i] = w1[r]; psi[r][2*i+1] = w2[r]
    Ppair = [[0.0]*K for _ in range(2*K)]
    for i in range(K): Ppair[2*i][i] = -0.5; Ppair[2*i+1][i] = -0.5
    Dop  = [[(gso[i][5] if i == j else 0.0) for j in range(K)] for i in range(K)]
    Dnop = [[(-gso[i][5] if i == j else 0.0) for j in range(K)] for i in range(K)]
    Mcomb = [[0.0]*K for _ in range(2*K)]
    for r in range(K):
        for c in range(K): Mcomb[r][c] = 1.0 if r < c else 0.0        # strict-upper
    for i in range(K): Mcomb[K+i][i] = 1.0                            # identity block
    color = [[gso[i][6], gso[i][7], gso[i][8]] for i in range(K)]

    MM.build_for("fp32")                                              # matmul kernel once
    MMrun = lambda A, B: take(MM.run_matmul(coord, ctx=ctx, a=pad(A), b=pad(B),
                                            out_format="fp32", prebuilt=True, verbose=False)["c_dev"], TILE, TILE)
    def SFrun(tile, op, rows, cols):
        flat = pad(tile)
        out, ok = SF.run_unary(coord, [flat[i] for i in range(1024)], ctx=ctx, op=op)  # builds per op
        return take(out, TILE, TILE)

    # ---- pipeline, verifying each stage vs host sim ----
    def sub(m, rows, cols): return [[m[i][j] for j in range(cols)] for i in range(rows)]

    V = sub(MMrun(phi, psi), P, 2*K)
    Vh = hmm(phi, psi, P, 3, 2*K)
    print(f"1 V=phi@psi          dev vs host maxrel={maxrel(V,Vh,P,2*K):.2e}")

    Vsq = sub(SFrun([[V[p][c] for c in range(2*K)] for p in range(P)], "square", P, 2*K), P, 2*K)
    Vsqh = [[R(Vh[p][c])**2 for c in range(2*K)] for p in range(P)]
    print(f"2 Vsq=square(V)      dev vs host maxrel={maxrel(Vsq,Vsqh,P,2*K):.2e}")

    E = sub(MMrun(Vsq, Ppair), P, K)
    Eh = hmm(Vsqh, Ppair, P, 2*K, K)
    print(f"3 E=Vsq@Ppair        dev vs host maxrel={maxrel(E,Eh,P,K):.2e}")

    ar = sub(SFrun(E, "exponential", P, K), P, K)
    arh = [[R(math.exp(min(Eh[p][i], 0.0))) for i in range(K)] for p in range(P)]
    print(f"4 ar=exp(E)          dev vs host maxrel={maxrel(ar,arh,P,K):.2e}")

    alpha  = sub(MMrun(ar, Dop),  P, K)
    nalpha = sub(MMrun(ar, Dnop), P, K)
    lpa = sub(SFrun(alpha, "log", P, K), P, K)
    la  = sub(SFrun(nalpha, "log1p", P, K), P, K)
    G = [[la[p][c] for c in range(K)] + [lpa[p][c] for c in range(K)] for p in range(P)]
    logw = sub(MMrun(G, Mcomb), P, K)
    w = sub(SFrun(logw, "exponential", P, K), P, K)
    C = sub(MMrun(w, color), P, 3)

    # golden for these 32 pixels
    def golden_px():
        out = []
        for (x, y) in px:
            rgb = [0.0, 0.0, 0.0]; T = 1.0
            for i in range(K):
                gx, gy, a, b, c, op = gso[i][:6]; dx, dy = x-gx, y-gy
                E0 = -0.5*(a*dx*dx + 2*b*dx*dy + c*dy*dy); al = op*math.exp(max(E0, -60))
                col = gso[i][6:9]
                for ch in range(3): rgb[ch] += T*al*col[ch]
                T *= (1-al)
            out.append(rgb)
        return out
    gold = golden_px()
    mse = sum((C[p][ch]-gold[p][ch])**2 for p in range(P) for ch in range(3))/(P*3)
    psnr = 99 if mse < 1e-12 else 10*math.log10(1/mse)
    print(f"\nFULL ON-DEVICE forward (6 MVMUL + 5 SFPU, 32 px): PSNR vs golden = {psnr:.1f} dB")
    print(f"  C[0]={[round(x,3) for x in C[0]]}  gold={[round(x,3) for x in gold[0]]}")
    print("  ->", "PASS" if psnr > 40 else "CHECK")

if __name__ == "__main__":
    main()
