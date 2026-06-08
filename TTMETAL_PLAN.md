# tt-metal NoC benchmark plan (verified against repos, June 2026)

Goal: reproduce the Blackhole data-movement bandwidth table with on-chip kernels (host
injection can't — capped at 32 B/flit) and see the NoC data live in bhtop.

## The big shortcut
tt-metal **already ships** the all-core NoC microbenchmarks at
`tests/tt_metal/tt_metal/data_movement/`:
| slide row | stock test |
|---|---|
| local 94 TB/s | `loopback` |
| neighbor 47 TB/s | `one_to_one` |
| row/col/multicast 24 TB/s | `one_to_all_multicast` |
| gather 16 TB/s | `all_from_all` (3-hop partner) |
| scatter / 10-hop 5 TB/s | `all_to_all` |
| DRAM row 512 GB/s | `dram_unary` |
| eth col 1 TB/s | ethernet dataflow test |
→ most of the table reproduces with **zero new code**. Only the custom all-core 3-hop/10-hop
gather/scatter variant is real work (the long pole, 3-6 days).

## Build (source; pip wheel insufficient for custom kernels + profiler) — ~0.5-1 day, mostly compile
1. `git clone --recurse-submodules` (in progress at ~/tt-metal)
2. `sudo ./install_dependencies.sh` (cmake4, clang/LLVM-20, hugepages) — needs sudo
3. env: `TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole PYTHONPATH=$(pwd) LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`
4. `./build_metal.sh --build-tests --build-programming-examples` (profiler ON by default — **no** `--enable-profiler`)
5. `./create_venv.sh && source python_env/bin/activate && pip install -r tt_metal/python_env/requirements-dev.txt`
6. smoke: `python -c 'import ttnn; d=ttnn.open_device(device_id=0); print(ttnn.get_arch_name()); ttnn.close_device(d)'` → `blackhole`
- KMD 2.8.0 is adequate; pin a recent commit (no stable v0.72.x). tt-smi -r warm reset unreliable on BH — keep host reboot in reserve.

## Measure (reconcile 4 numbers)
- wall-clock around EnqueueProgram/Finish (floor, dispatch-contaminated)
- **device profiler** cycles: `DeviceZoneScopedN` + `tt::tt_metal::detail::ReadDeviceProfilerResults(device)`
  (NOT `DumpDeviceProfileResults` — doesn't exist), `TT_METAL_DEVICE_PROFILER=1`, → `generated/profiler/.logs/profile_log_device.csv`, /1.35 GHz
- **bhtop silicon counters** (ground truth): snapshot 0x200 flit counters before/after → exact bytes/tile/NoC
- **tt-npe** (model): `profile_this.py --collect-noc-traces` → `npe_viz/*.json` → ttnn-visualizer `/npe` tab

## See NoC data with benchmark (the integration)
1. **In-process delta (recommended, safe):** snapshot bhtop 0x200 counters before `EnqueueProgram`, after `Finish`; delta×64 = exact bytes moved per tile/NoC → color the mesh. No device contention.
2. **Read-only live stream:** tt-metal owns the device; launch `tt-exalens --server` (port 5555), bhtop sampler attaches read-only (UMD robust mutex coordinates), modest poll rate, keep observer-calibration ON. Only safe 0x200/0x500 reads. Never reset/inject from the observer.

## Honesty flags
- Host injection is hard-capped at 32 B/flit → on-chip kernels are mandatory for the dense numbers (confirms our earlier measurement).
- The workflow claimed the **16 TB/s is a 32-chip Galaxy *system DRAM* number, not single-card on-chip** — TREAT SKEPTICALLY: the slide labels it SRAM, and the single-card on-chip injection ceiling is ~num_NIUs×60.9 B/cyc×1.35 GHz ≈ ~20 TB/s, so 16 TB/s single-card is not obviously impossible. We'll resolve it by measuring `all_from_all`.

Corrected tooling sharp edges: no `--enable-profiler`; `--debug` not `CONFIG=Debug`; `ttnn.get_arch_name()` takes no arg; NPE tab is a ttnn-visualizer feature not tt-npe.
