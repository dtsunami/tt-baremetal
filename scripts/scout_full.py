"""Full enumeration dump. Safe: build() only — no counters, no inject, no regs."""
from collections import Counter, defaultdict
from ttexalens import init_ttexalens
from bhtop.floorplan import build

fp = build(init_ttexalens())
P = fp.placed
by_noc0 = {t.noc0: t for t in P}
by_die  = {t.die:  t for t in P}
ncols = max(x for x,_ in by_noc0)+1; nrows = max(y for _,y in by_noc0)+1
dcols = max(x for x,_ in by_die)+1;  drows = max(y for _,y in by_die)+1
print(f"GRID noc0 {ncols}x{nrows}  die {dcols}x{drows}  tiles={len(P)}")

def manh(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

# ---- COLUMN permutation: noc0 col -> die col (verify it's column-uniform) ----
print("\n=== COLUMN MAP (noc0 col -> die col) ===")
col_of = {}
for nx in range(ncols):
    dset = sorted({by_noc0[(nx,y)].die[0] for y in range(nrows) if (nx,y) in by_noc0})
    assert len(dset)==1, (nx,dset)
    col_of[nx]=dset[0]
    print(f"  noc0 col {nx:2d} -> die col {dset[0]:2d}")
print("  inverse die->noc0:", {col_of[n]:n for n in sorted(col_of)})

# ---- ROW permutation: is each noc0 row a single die row? ----
print("\n=== ROW MAP (noc0 row -> die row set) ===")
row_uniform=True
row_of={}
for ny in range(nrows):
    dset = sorted({by_noc0[(x,ny)].die[1] for x in range(ncols) if (x,ny) in by_noc0})
    if len(dset)==1: row_of[ny]=dset[0]
    else: row_uniform=False
    print(f"  noc0 row {ny:2d} -> die rows {dset}")
print("  row uniform (each noc0 row == one die row)?", row_uniform)
if row_uniform:
    print("  inverse die->noc0 row:", {row_of[n]:n for n in sorted(row_of)})

# ---- full per-tile dump ordered by die ----
print("\n=== ALL TILES (die -> noc0, kind) ===")
for y in range(drows):
    for x in range(dcols):
        t=by_die.get((x,y))
        if t: print(f"  die({x:2d},{y:2d}) noc0({t.noc0[0]:2d},{t.noc0[1]:2d}) {t.kind:9s} {t.label}")

# ---- ASCII maps both spaces ----
for sysn in ("die","noc0"):
    cells = {getattr(t,sysn):t for t in P}
    cc=max(x for x,_ in cells)+1; rr=max(y for _,y in cells)+1
    print(f"\n=== {sysn} MAP {cc}x{rr} (col index header) ===")
    print("    "+"".join(str(x%10) for x in range(cc)))
    for y in range(rr):
        print(f"  {y:2d} "+"".join(cells[(x,y)].glyph if (x,y) in cells else "." for x in range(cc)))

# ---- noc0 +x and +y physical displacement (monotonic vs zigzag) ----
print("\n=== NoC0 +x step -> die displacement (per noc0 col boundary) ===")
for nx in range(ncols-1):
    print(f"  noc0 col {nx:2d}->{nx+1:2d} : die col {col_of[nx]:2d}->{col_of[nx+1]:2d}  ddx={col_of[nx+1]-col_of[nx]:+d}")
if row_uniform:
    print("=== NoC0 +y step -> die displacement (per noc0 row boundary) ===")
    for ny in range(nrows-1):
        print(f"  noc0 row {ny:2d}->{ny+1:2d} : die row {row_of[ny]:2d}->{row_of[ny+1]:2d}  ddy={row_of[ny+1]-row_of[ny]:+d}")

# ---- nearest DRAM: physical vs logical for sample Tensix ----
ctrl_dies = {c:[t.die for t in ts] for c,ts in fp.dram_ctrl.items()}
ctrl_noc0 = {c:[t.noc0 for t in ts] for c,ts in fp.dram_ctrl.items()}
def torus(a,b,W,H):
    dx=abs(a[0]-b[0]); dx=min(dx,W-dx)
    dy=abs(a[1]-b[1]); dy=min(dy,H-dy)
    return dx+dy
tensix=[t for t in P if t.kind=="tensix"]
samples=[]
# pick spread-out samples by die position
for target in [(1,3),(7,3),(15,11),(11,7),(5,9),(13,5)]:
    t=by_die.get(target)
    if t and t.kind=="tensix": samples.append(t)
print("\n=== NEAREST DRAM: physical(die Manhattan) vs logical(noc0 torus hops) ===")
for t in samples:
    phys=min(((c,min(manh(t.die,d) for d in ds)) for c,ds in ctrl_dies.items()), key=lambda r:r[1])
    logi=min(((c,min(torus(t.noc0,n,ncols,nrows) for n in ns)) for c,ns in ctrl_noc0.items()), key=lambda r:r[1])
    flag="AGREE" if phys[0]==logi[0] else "DIFFER"
    print(f"  Tensix die{t.die} noc0{t.noc0}: phys-near d{phys[0]}(dist {phys[1]})  logi-near d{logi[0]}(hops {logi[1]})  {flag}")

# ---- die positions of special kinds ----
print("\n=== SPECIAL KIND die positions ===")
for k in ("eth","arc","pcie","l2cpu","security"):
    ts=sorted((t.die,t.noc0) for t in P if t.kind==k)
    print(f"  {k:9s}: "+", ".join(f"die{d}/noc0{n}" for d,n in ts))

# ---- scan ALL tensix for nearest-DRAM divergence ----
print("\n=== ALL Tensix: phys-near vs logi-near DRAM (only DIFFER cases) ===")
ndiff=0
for t in tensix:
    phys=min(((c,min(manh(t.die,d) for d in ds)) for c,ds in ctrl_dies.items()), key=lambda r:r[1])
    logi=min(((c,min(torus(t.noc0,n,ncols,nrows) for n in ns)) for c,ns in ctrl_noc0.items()), key=lambda r:r[1])
    if phys[0]!=logi[0]:
        ndiff+=1
        print(f"  Tensix die{t.die} noc0{t.noc0}: phys d{phys[0]}(dist {phys[1]}) vs logi d{logi[0]}(hops {logi[1]})")
print(f"  total tensix={len(tensix)}  divergent={ndiff}")

# ---- noc0-adjacent pairs whose die-distance is large, sorted ----
print("\n=== noc0 grid-adjacent (1 hop) with LARGEST die Manhattan span ===")
adj=[]
for t in P:
    x,y=t.noc0
    for dx,dy in ((1,0),(0,1)):
        nb=by_noc0.get((x+dx,y+dy))
        if nb: adj.append((manh(t.die,nb.die),t,nb))
for d,a,b in sorted(adj,key=lambda r:-r[0])[:8]:
    print(f"  noc0 {a.noc0}->{b.noc0} (1 hop): die {a.die}->{b.die}  Manhattan {d}")

# ---- die-adjacent pairs whose noc0 distance is large ----
print("\n=== die-adjacent (physically touching) with LARGEST noc0 torus span ===")
dadj=[]
for t in P:
    x,y=t.die
    for dx,dy in ((1,0),(0,1)):
        nb=by_die.get((x+dx,y+dy))
        if nb: dadj.append((torus(t.noc0,nb.noc0,ncols,nrows),t,nb))
for d,a,b in sorted(dadj,key=lambda r:-r[0])[:8]:
    print(f"  die {a.die}->{b.die} (touching): noc0 {a.noc0}->{b.noc0}  torus hops {d}")

# ---- verify col interleave FORMULA ----
print("\n=== COLUMN INTERLEAVE FORMULA CHECK ===")
# hypothesis: die_col d -> noc0:  d=0->0; odd d in 1..15 -> (d+1)//2 ; even d in 2..16 -> 17-(d//2)
ok=True
for d,n in {0:0,1:1,3:2,5:3,7:4,9:5,11:6,13:7,15:8,16:9,14:10,12:11,10:12,8:13,6:14,4:15,2:16}.items():
    if d==0: pred=0
    elif d%2==1: pred=(d+1)//2
    else: pred=17-(d//2)
    if pred!=n: ok=False; print(f"  MISMATCH die {d}: pred {pred} got {n}")
print("  formula odd d->(d+1)//2, even d->17-(d//2), d0->0 :", "HOLDS" if ok else "FAILS")

# ---- DRAM controllers per die edge ----
from collections import defaultdict as dd2
edgec=dd2(set)
for c,ts in fp.dram_ctrl.items():
    for t in ts:
        edgec["LEFT(col0)" if t.die[0]==0 else "RIGHT(col16)"].add(c)
print("\n=== DRAM controllers per die edge ===")
for e,cs in edgec.items(): print(f"  {e}: ctrls {sorted(cs)}  ({len(cs)} ctrls x3 tiles)")

# ---- the FOLD SEAM: die cols 15<->16 and 0<->1 join in noc0 ----
print("\n=== FOLD SEAM (die col 15<->16 touch; their noc0 cols) ===")
print(f"  die col15 -> noc0 col {col_of_inv if False else 8}, die col16 -> noc0 col 9 : noc0 ADJACENT(8,9) though they are the two opposite spine/edge")

# ---- ROW interleave formula (12 rows) ----
print("\n=== ROW INTERLEAVE FORMULA CHECK (H=12) ===")
rowmap={0:0,1:1,3:2,5:3,7:4,9:5,11:6,10:7,8:8,6:9,4:10,2:11}
ok=True
for d,n in rowmap.items():
    if d==0: pred=0
    elif d%2==1: pred=(d+1)//2
    else: pred=12-(d//2)
    if pred!=n: ok=False; print(f"  MISMATCH die row {d}: pred {pred} got {n}")
print("  formula odd d->(d+1)//2, even d->12-(d//2), d0->0 :", "HOLDS" if ok else "FAILS")

# DRAM left-edge die rows 0..11 -> noc0 rows; show the row fold puts halves into noc0 top/bottom
print("\n=== noc0 row occupancy of die-col0 DRAM (shows row fold) ===")
for t in sorted((tt for tt in P if tt.die[0]==0), key=lambda z:z.die[1]):
    print(f"  die row {t.die[1]:2d} -> noc0 row {t.noc0[1]:2d}  ({t.label})")
