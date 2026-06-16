"""
Curated, offline reference docs for the Kernel Lab right-hand pane.

Kept as plain markdown so the reference lives next to the editor without a network
round-trip. Counter facts are cross-checked against tenstorrent/tt-isa-documentation
BlackholeA0/NoC/Counters.md and bhtop's own noc_counters.py decode.
"""

_GUIDE = """# Kernel Lab — how it works

A Blackhole kernel is two layers:

- **Host program** (`test_*.cpp`) — runs on the x86 host. Creates the `Program`,
  places kernels on a core grid with `CreateKernel(...)`, sets runtime args, and
  launches with `EnqueueMeshWorkload` / `Finish`. **Editing this needs a rebuild.**
- **Device kernels** (`kernels/*.cpp`) — run on the Tensix RISC-V cores. tt-metal
  **JIT-compiles these at run time**, so editing a device kernel needs **no rebuild** —
  just hit **Run** and the change is live in seconds.

### The loop
1. Pick a file on the left. `device` files are the fast path; `host` files need a build.
2. Edit, **Save**.
3. If you changed a `host` file, **Build** (incremental `ninja`; compile errors land
   in the Debug tab). Device-only edits skip this.
4. **Run**. tt-metal owns the device for the run, so live polling pauses; when it
   finishes you get the per-NoC **footprint** overlaid on the chip (Telemetry tab),
   the profiler bandwidth, and any **DPRINT** output (Debug tab).

### Debugging on device
- `DPRINT("x={}\\n", v);` from a kernel prints to host stdout when DPRINT is enabled
  (toggle in the toolbar; sets `TT_METAL_DPRINT_CORES`). `DPRINT_DATA0/1` select the
  RISCV_0/RISCV_1 data-movement core.
- The hardware NIU counters are the ground truth — see *NoC counters*. They can't be
  read while tt-metal owns the device, so bhtop reads them **right after** the run;
  the footprint you see is cumulative since tt-metal's device init = this kernel.
"""

_COUNTERS = """# NoC NIU counters (Blackhole)

Every tile has **two NIUs** (NoC interface units): NIU#0 → NoC0 (`0xFFB2_0000`),
NIU#1 → NoC1 (`0xFFB3_0000`). Each exposes a **62-entry array of 32-bit counters at
`NIU_BASE + 0x0200`**. A NoC "data word" is one **512-bit (64-byte) flit**, so
`bytes = count * 64`. Counters free-run and wrap at 2³²; bandwidth is the delta
between two samples over wall-clock — this is exactly what bhtop's poller does.

### Flit counters bhtop uses for bandwidth
| idx | name | meaning |
|----:|------|---------|
| 3  | `MST_RD_DATA_WORD_RECEIVED` | read data this tile pulled **in** (requester) |
| 8  | `MST_NONPOSTED_WR_DATA_WORD_SENT` | write data pushed **out** (nonposted) |
| 9  | `MST_POSTED_WR_DATA_WORD_SENT` | write data pushed **out** (posted) |
| 51 | `SLV_RD_DATA_WORD_SENT` | read data this tile **served** (e.g. DRAM reads) |
| 56 | `SLV_NONPOSTED_WR_DATA_WORD_RECEIVED` | write data **landed** here (nonposted) |
| 57 | `SLV_POSTED_WR_DATA_WORD_RECEIVED` | write data **landed** here (posted) |

Directional rollups (see `noc_counters.py`): `tx_master=[8,9]`, `rx_master=[3]`,
`tx_slave=[51]`, `rx_slave=[56,57]`.

### Full layout
- **0–15** master-side (`MST_*`): atomic/wr-ack/rd-resp received, cmd accepted,
  rd/atomic/wr requests + data words sent & started.
- **16–31** `MST_REQS_OUTSTANDING_ID(0..15)` — **8-bit** in-flight depth per txn id.
- **32–47** `MST_WRITE_REQS_OUTGOING_ID(0..15)` — **8-bit** write drain depth.
- **48–61** slave-side (`SLV_*`): atomic/wr-ack/rd-resp sent, req accepted,
  rd/atomic/wr received & started.

> ⚠ Indices 16–47 are **8-bit** (wrap at 256). bhtop only diffs the 32-bit flit
> indices above, so its `& 0xFFFFFFFF` delta is correct — but don't delta 16–47.

### Gotchas
- A *clear* register exists at `NIU_BASE + 0x0060`; bhtop ignores it and uses deltas.
- Memory ordering: read back `NOC_CMD_CTRL` before the first counter read after issuing
  traffic. bhtop's injector already polls `CMD_CTRL`→0 before sampling.
- Touching ARC / Security / PCIe / L2CPU NIUs hangs NoC0 — bhtop only reads
  Tensix / DRAM / Eth tiles.
"""

_DATAFLOW = """# Dataflow API cheat-sheet

Device-kernel calls (from `api/dataflow/dataflow_api.h`), as used by the 3-hop kernel.

### Addressing
```cpp
uint64_t a = get_noc_addr(phys_x, phys_y, l1_addr);  // pack remote NoC address
```
On **Blackhole the same physical (NoC-0) coords are used for both NoCs** — direction
(X-then-Y east/south on NOC0, Y-then-X north/west on NOC1) is a hardware property of
the NoC, *not* coordinate math. Don't mirror coords for NOC1.

### Move data
```cpp
noc_async_write(src_l1, dst_noc_addr, n_bytes, noc_index, vc);  // push local -> remote
noc_async_read (src_noc_addr, dst_l1,  n_bytes, noc_index, vc); // pull remote -> local
noc_async_write_barrier();   // drain all outstanding writes
noc_async_read_barrier();    // drain all outstanding reads
```
- `noc_index` is implicit from the kernel's `DataMovementConfig.noc` binding.
- **One barrier per NoC, OUTSIDE the loops, INSIDE the timed zone.** A per-iteration
  barrier measures latency, not bandwidth.
- `vc` ≤ 4 (only 4 unicast write VCs).

### Compile-time vs runtime args
```cpp
constexpr uint32_t v = get_compile_time_arg_val(0);  // from .compile_args
uint32_t x = get_arg_val<uint32_t>(0);               // from SetRuntimeArgs
```

### Profiler (drives the bandwidth numbers)
```cpp
{ DeviceZoneScopedN("RISCV0"); /* timed region */ }
DeviceTimestampedData("Per-core bytes", per_core_bytes);  // REQUIRED for asymmetric traffic
DeviceTimestampedData("Transaction size in bytes", n);
```

### Blackhole specifics
64B page · 16384B single-packet fast path · 64 B/cyc per NoC · explicit flush before
any cross-core signal.

### Host side (`test_*.cpp`)
```cpp
auto k = CreateKernel(program, "..../writer.cpp", core_set,
    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                       .noc=NOC::NOC_0, .compile_args=cta});
SetRuntimeArgs(program, k, core, {phys_x, phys_y});
// two kernels (RISCV_0/NOC_0 + RISCV_1/NOC_1) on one core = both NoCs concurrently.
```
"""

_BANDWIDTH = """# Three bandwidth numbers to reconcile

The 3-hop benchmark can be measured three independent ways — agreement is the
sanity check.

1. **Per-core effective** — `num_tx * tx_size / duration_cycles`, median across cores.
   Filter per NoC: NOC0 on `riscv_0`, NOC1 on `riscv_1`. Cap: 64 B/cyc/NoC.
2. **Aggregate / combined** — `total_bytes / wall_clock_time`, where
   `wall_clock_time = max(end) − min(start)` across cores and `total_bytes` sums each
   core's `"Per-core bytes"` stamp. This is the headline all-core number. Sum
   NOC0+NOC1 for total dual-NoC throughput.
3. **bhtop NIU flit-counter delta** (hardware ground truth) — counter delta × 64 B/flit
   over the run. Independent of the kernel's self-reported byte counts.

(2) should track (3) within flit/header overhead. (1)×active-cores ≈ (2) only if traffic
is uniform — it is **not** here (edge cores have fewer partners, gather ≠ scatter), which
is why `"Per-core bytes"` must be stamped per core. Read-request flits make NOC1 (gather)
flit counts in (3) exceed payload bytes in (2); account for that on the read direction.
"""

DOCS = [
    {"id": "guide", "title": "Kernel Lab — how it works", "body": _GUIDE},
    {"id": "counters", "title": "NoC NIU counters", "body": _COUNTERS},
    {"id": "dataflow", "title": "Dataflow API cheat-sheet", "body": _DATAFLOW},
    {"id": "bandwidth", "title": "Three bandwidth numbers", "body": _BANDWIDTH},
]
