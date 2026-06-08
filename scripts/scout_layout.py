"""Scout die<->noc0 mapping to quantify physical-vs-logical. Safe: build() only."""
from collections import Counter, defaultdict
from ttexalens import init_ttexalens
from bhtop.floorplan import build

fp = build(init_ttexalens())
by_noc0 = {t.noc0: t for t in fp.placed}
by_die  = {t.die:  t for t in fp.placed}
ncols = max(x for x,_ in by_noc0)+1; nrows = max(y for _,y in by_noc0)+1
dcols = max(x for x,_ in by_die)+1;  drows = max(y for _,y in by_die)+1
print(f"noc0 grid {ncols}x{nrows}   die grid {dcols}x{drows}   tiles={len(fp.placed)}")

def manh(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

# 1) physical (die) distance of each noc0 NON-wrap adjacency (both tiles present)
adj_dist = []
for t in fp.placed:
    x,y = t.noc0
    for dx,dy in ((1,0),(0,1)):
        nb = by_noc0.get((x+dx,y+dy))
        if nb: adj_dist.append((manh(t.die, nb.die), t, nb))
dd = Counter(d for d,_,_ in adj_dist)
print("\n[1] noc0 grid-adjacent links: physical(die) Manhattan distance histogram")
for d in sorted(dd): print(f"   die-dist {d:2d} : {dd[d]:3d} links")
print("   --> links where 1 logical hop = a LONG physical span (die-dist>=4):")
for d,a,b in sorted(adj_dist, key=lambda r: -r[0])[:14]:
    print(f"      noc0 {a.noc0}{a.glyph} -> {b.noc0}{b.glyph}  die {a.die}->{b.die}  dist={d}")

# 2) torus WRAPAROUND links (noc0 col/row edge wrap) physical distance
print("\n[2] torus wraparound links (logical 1-hop, physical span):")
wrap=[]
for t in fp.placed:
    x,y=t.noc0
    if x==ncols-1:
        nb=by_noc0.get((0,y))
        if nb: wrap.append(("col-wrap",t,nb,manh(t.die,nb.die)))
    if y==nrows-1:
        nb=by_noc0.get((x,0))
        if nb: wrap.append(("row-wrap",t,nb,manh(t.die,nb.die)))
wd=Counter(d for _,_,_,d in wrap)
print("   wraparound die-dist histogram:", dict(sorted(wd.items())))
print(f"   (a wrap link is logically 1 hop but spans up to ~{max((d for *_,d in wrap), default=0)} die cells)")

# 3) per-kind die-edge placement (where each I/O type physically is)
print("\n[3] physical (die) placement by kind:")
edge=defaultdict(lambda: Counter())
for t in fp.placed:
    x,y=t.die
    pos = []
    if x==0: pos.append("LEFT")
    if x==dcols-1: pos.append("RIGHT")
    if y==0: pos.append("TOP")
    if y==drows-1: pos.append("BOTTOM")
    edge[t.kind][tuple(pos) or ("interior",)] += 1
for k,c in edge.items():
    print(f"   {k:9s}: "+", ".join(f"{'+'.join(p)}x{n}" for p,n in c.items()))

# 4) DRAM controller -> die positions (which edge each controller sits on)
print("\n[4] GDDR6 controllers: die positions (edge => must be adjacent to its DRAM pkg):")
for c,ts in sorted(fp.dram_ctrl.items()):
    dies=sorted(t.die for t in ts); n0=sorted(t.noc0 for t in ts)
    edges=set("L" if d[0]==0 else "R" if d[0]==dcols-1 else "?" for d in dies)
    print(f"   d{c}: die {dies}  edge={''.join(sorted(edges))}   noc0 {n0}")

# 5) noc0 columns -> which die columns they pull from (the fold)
print("\n[5] the FOLD: each noc0 column -> set of die columns it contains")
for nx in range(ncols):
    dcols_here = sorted({by_noc0[(nx,y)].die[0] for y in range(nrows) if (nx,y) in by_noc0})
    kinds = sorted({by_noc0[(nx,y)].kind[:1] for y in range(nrows) if (nx,y) in by_noc0})
    print(f"   noc0 col {nx:2d}: die cols {dcols_here}  kinds {kinds}")
