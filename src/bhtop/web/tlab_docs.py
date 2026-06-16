"""
Curated docs for the Tensix Compute Lab (tlab) — the analog of lab_docs.py for nlab.
Plain markdown strings; the frontend renders them in the shared <DocsPane> alongside the
live tt-isa-documentation tree. Written for someone meeting the Tensix compute engine.
"""

_COMPUTE = """# The Tensix compute core — 5 baby RISC-Vs

Each **Tensix core** (the `T` tiles on the chip) is driven by **five small 32-bit RISC-V
cores** sharing one **L1 memory**:

```
  RISC-V 1   RISC-V 2   RISC-V 3   RISC-V 4   RISC-V 5
  (reader)   UNPACK     MATH       PACK       (writer)
     |        \\         |         /              |
  Router 0     ---- Compute ----            Router 1
     \\__________________ L1 Memory __________________/
```

- **2 data-movement cores** (RISC-V 1 + 5, a.k.a. NCRISC/BRISC) run your **data-movement
  kernels** — they pull tiles from DRAM into L1 (reader, via Router 0) and push results back
  out (writer, via Router 1) with `noc_async_read/write` + circular buffers.
- **3 compute cores** (RISC-V 2/3/4 = **UNPACK / MATH / PACK**) run your **one compute
  kernel** — the *same source compiled three times*, gated by `#ifdef TRISC_UNPACK/MATH/PACK`.
  UNPACK fetches tiles from L1 into the source registers, MATH runs the matrix/vector FPU into
  the DST accumulator, PACK writes DST back to L1.

So **3 user C kernels program one Tensix core**: 1 compute (→ 3 cores) + 2 data-movement.

## A compute kernel, in one tile-op
```cpp
tile_regs_acquire();              // MATH locks the DST accumulator
cb_wait_front(cb_in0, 1);         // UNPACK waits for an input tile in L1
cb_wait_front(cb_in1, 1);
add_tiles(cb_in0, cb_in1, 0, 0, 0);   // UNPACK feeds, MATH computes DST[0]=A+B
tile_regs_commit();               // MATH: results ready
tile_regs_wait();                 // PACK waits for MATH
pack_tile(0, cb_out0);            // PACK writes DST[0] to the output CB
tile_regs_release();
```
Matmul is the same shape with `mm_init()` + `matmul_tiles()` (which *accumulates* into DST).
SFPU ops (`exp_tile`, `sigmoid`, ...) run elementwise vector functions on DST.
"""

_OCCUPANCY = """# Reading the per-engine occupancy

After a run, bhtop reads the device profiler and shows, **per Tensix core**, how many cycles
each of the 5 RISC-Vs was busy in its kernel. The headline is **MATH occupancy** =
MATH-engine busy cycles ÷ the core's wall time.

- **High MATH occupancy (→100%)** = *compute-bound*: the matrix engine is the bottleneck,
  the data movement is keeping it fed. This is what you want for a GEMM.
- **Low MATH occupancy** = *memory-bound*: the FPU is idle waiting on UNPACK / the
  data-movement cores. The kernel is dominated by moving tiles, not crunching them.

Measured on this silicon:

| example | MATH occupancy | verdict |
|---|---|---|
| `matmul_single_core` | **99.9%** | compute-bound (FPU saturated) |
| `add_2_integers_in_compute` | **7.4%** | memory-bound (one tile add, lots of I/O) |

That single number tells you instantly where to optimize: feed the MATH engine better, or
make the math denser. (This is the *coarse* per-engine span — a true per-op FLOPs/utilization
breakdown needs instrumented kernels, a future tier.)
"""

_FIDELITY = """# Numerical knobs — MathFidelity & DST

The compute kernel's `ComputeConfig` trades precision for speed:

- **MathFidelity**: `LoFi` (fastest, fewest passes) · `HiFi2` · `HiFi3` · `HiFi4` (most
  accurate, most matrix passes). LoFi multiplies fewer mantissa bits per pass; HiFi4 does the
  full product. For bf8/bf16 inputs LoFi is often enough; for fp16/fp32-ish accuracy use HiFi.
- **fp32_dest_acc_en**: the DST accumulator is bf16 by default; set true for fp32 accumulation
  (more accurate partial sums, doubles DST SRAM use).
- **math_approx_mode**: faster SFPU transcendentals (exp/sigmoid/...) at lower accuracy.

These change *both* the result and the MATH-engine cycle count — flip MathFidelity and watch
the occupancy move.
"""

DOCS = [
    {"id": "compute", "title": "Tensix compute core (5 RISC-Vs)", "body": _COMPUTE},
    {"id": "occupancy", "title": "Reading occupancy", "body": _OCCUPANCY},
    {"id": "fidelity", "title": "MathFidelity & DST", "body": _FIDELITY},
]


def docs_index():
    return [{"id": d["id"], "title": d["title"], "kind": "md"} for d in DOCS]


def doc(doc_id):
    for d in DOCS:
        if d["id"] == doc_id:
            return {"id": doc_id, "title": d["title"], "markdown": d["body"]}
    raise ValueError(f"unknown doc: {doc_id}")
