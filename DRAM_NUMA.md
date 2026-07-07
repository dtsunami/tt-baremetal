# Blackhole DRAM NUMA map — the affinity that shapes the at-scale fused splat flow

Measured live from the ttexalens device description (noc0 coords). **8 GDDR6 chips (4 GiB each = 32 GiB)
surface as 24 DRAM NoC tiles** — **3 NoC ports per chip, all wired to the GDDR xbar** → any of a chip's 3
ports reaches its full 4 GiB. 4 chips per DRAM column (x=0 and x=9), each chip = 3 ports at 3 y-positions.

**Bandwidth consequence (the reason the 3 ports matter):** a chip's 4 GiB can be fed by up to **3 parallel
NoC streams** (its 3 ports). For operand streaming — dozens of workers pulling their tiles' operands from the
same chip — round-robin the tiles/reads across the chip's 3 ports so concurrent reads hit different ports
(≈3× port bandwidth up to the bank limit) instead of serializing on one. The flip side is xbar contention:
all 3 ports share the bank, so many ports hammering the *same rows* contends — keep each port's access
row-friendly (aligned, sequential). (Exact port↔chip grouping isn't in ttexalens; it's the umd/SoC GDDR
channel map — but placement is column- + nearest-port-dominated, which is what we have.)

## The floorplan (noc0: x → 0..16, y ↓ 2..11 for compute)

```
 x= 0        1 2 3 4 5 6     7      8        9        10     11 12 13 14 15 16
   ┌────┐   ┌───────────┐  ┌──┐  ┌────┐   ┌────┐   ┌──┐  ┌────────────┐
 y │DRAM│   │  LEFT      │  │hv│  │L2CPU│  │DRAM│   │hv│  │  RIGHT      │
   │x=0 │   │  workers   │  │  │  │x280 │  │x=9 │   │  │  │  workers    │
   │4   │   │  60 tiles  │  │  │  │(8,y)│  │4   │   │  │  │  60 tiles   │
   │chips   │  x=1..6    │  │  │  │3,5,7,9  │chips   │  │  │  x=11..16   │
   │16GiB   │            │  │  │  │      │  │16GiB   │  │  │             │
   └────┘   └───────────┘  └──┘  └────┘   └────┘   └──┘  └────────────┘
   LEFT bank   left NUMA          the 4      x280-local     right NUMA
              domain              hubs       bank (1 hop)   domain
```
Other tiles: pcie (2,0), arc (8,0), security (8,2), eth (y=1), 20 harvested workers (x=7,10).

## Affinity (Manhattan noc0 hops to nearest DRAM tile)

| consumer | nearest DRAM | hops (mean) | to x=0 bank | to x=9 bank |
|---|---|---|---|---|
| **L2CPU / x280** (x=8) | **x=9, 1 hop** | **1.0** | 8.0 (far) | **1.0** |
| **RIGHT workers** (x=11..16) | x=9 | 4.5 | 13.5 (very far) | 4.5 |
| **LEFT workers** (x=1..6) | x=0 | 3.5 | 3.5 | 5.5 |

Per-hub local bank: L2CPU (8,3)→DRAM (9,3) · (8,5)→(9,5) · (8,7)→(9,7) · (8,9)→(9,9), all **1 hop**.
This is the 4 GiB @ 0x30000000 window each x280 sees (Gap 5): its NUMA-local x=9 chip.

## The two NUMA domains — the load-bearing fact

- **RIGHT domain = { the 4 x280 hubs (x=8), the x=9 DRAM (16 GiB), the 60 right workers (x=11..16) }.**
  Everything within ~1–5 hops. The x280 produces operands into its local x=9 bank; right workers read them
  near; write grads back near. **Tight, NUMA-clean, 60-way parallel.**
- **LEFT domain = { x=0 DRAM (16 GiB), the 60 left workers (x=1..6) }** — but **NO x280 hub** (all four are
  at x=8, 8 hops from x=0). Left workers are local to x=0, but the *producer* (x280) is far from x=0.

**Consequence:** the x280-produces-operands model is NUMA-optimal for the right block and pays a cross-chip
penalty for the left block (x280 writes to x=0 = 8 hops, or left workers read x=9 = 5.5 hops).

## Data-placement strategy (drives the at-scale build)

1. **Params + Adam state + coeff buffer live in the x=9 (x280-local) bank.** The optimizer is 1 hop from
   its state — cheapest place for the per-step Adam sweep. (Replicate read-only params to x=0 if the left
   block is used — params are read-only within a step, mutated only at Adam.)
2. **Operands + grad inbox: place each tile's in the bank nearest ITS worker.** Right tiles → x=9 (near
   x280, cheap to produce). Left tiles → x=0 (near left workers, but a far x280 write).
3. **Stage 1 (this build): use the RIGHT 60-worker domain only.** NUMA-clean, no cross-chip traffic,
   10,000 tiles / 60 ≈ 167 tiles/worker. Simplest correct at-scale flow; the telemetry then guides tuning.
   ⚠ **All the lever de-risk tests ran on worker (1,2) = the LEFT block** — the *worst-case* cross-chip path
   (5.5 hops from the x280-local x=9 bank). They were bit-exact anyway, so correctness is path-independent;
   the at-scale flow should switch to the RIGHT block (x=11..16) for the ~1–5-hop NUMA-local path.
4. **Stage 2 (full 120): dual-domain.** Shard tiles by side; the x280 produces right-operands into x=9 and
   left-operands into x=0 (or a left-side producer strategy — e.g. left workers self-produce, or accept the
   8-hop x280→x=0 write since it's one write per tile amortized over 8 render rings + a grad read-back).
5. **Reserve one x280 hart as the on-chip telemetry tracer** (Dan's constraint) — separate from the
   compute/scheduler harts.

Query: `ttexalens dev.get_block_locations("dram"|"functional_workers"|"l2cpu")`. Scripted in this session.
Nuance: noc0 is dimension-ordered/unidirectional (request one way, response via noc1/wrap) — Manhattan is
the affinity *ranking* proxy; exact latency is directional but the column-dominated ranking holds.
