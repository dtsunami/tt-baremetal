"""Cross-check the scalar C (gap1_proj.c) against the numpy golden (gap1_proj_golden.py).
Feeds identical random cases; compares fwd (gx,gy,a,b,c) + bwd (dmean3,dsl3,dq4). float32 C vs
float64 golden, so tolerance is fp32-scale (~1e-4 rel)."""
import subprocess, numpy as np
import gap1_proj_golden as G

Rv = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]); tv = np.array([0.0, 0.0, 6.0])
fx = fy = 70.0; cx = cy = 32.0
rng = np.random.default_rng(123)

cases, golden = [], []
for _ in range(300):
    mean = rng.normal(0, 1.5, 3); sl = np.log(0.25 + rng.random(3) * 0.5); q = rng.normal(0, 1, 4)
    da, db, dc, dgx, dgy = rng.normal(0, 1, 5)
    cases.append((mean, sl, q, da, db, dc, dgx, dgy))
    gx, gy, dep, a, b, c, cache = G.project_forward(mean, sl, q, Rv, tv, fx, fy, cx, cy)
    dm, ds, dq = G.project_backward(da, db, dc, dgx, dgy, cache)
    golden.append([gx, gy, a, b, c, *dm, *ds, *dq])

inp = "\n".join(" ".join(f"{v:.9g}" for v in (*m, *s, *qq, da, db, dc, dgx, dgy))
                for (m, s, qq, da, db, dc, dgx, dgy) in cases)
subprocess.run(["gcc", "-O2", "-o", "gap1_proj", "gap1_proj.c", "-lm"], check=True, cwd=".")
out = subprocess.run(["./gap1_proj"], input=inp, capture_output=True, text=True, check=True).stdout
cout = np.array([[float(x) for x in ln.split()] for ln in out.strip().splitlines()])
gold = np.array(golden)

labels = ["gx", "gy", "a", "b", "c", "dmx", "dmy", "dmz", "dsl0", "dsl1", "dsl2", "dq0", "dq1", "dq2", "dq3"]
absd = np.abs(cout - gold)
reld = absd / (np.abs(gold) + 1e-6)
print(f"{'field':>5} {'max|abs|':>11} {'max|rel|':>11}")
worst = 0.0
for i, lb in enumerate(labels):
    print(f"{lb:>5} {absd[:, i].max():11.2e} {reld[:, i].max():11.2e}")
    worst = max(worst, reld[:, i].max())
print(f"\nworst rel error (fp32 C vs fp64 golden) = {worst:.2e}")
print("GAP1_C_MATCHES_GOLDEN" if worst < 3e-3 else "GAP1_C_MISMATCH")
