# tt-metal local additions (preserved)

Local, **non-upstream** work that lived in the working tree of `~/tt-metal` on `ttstar`.
`~/tt-metal`'s `origin` is upstream `tenstorrent/tt-metal` (no push access), so this work is
mirrored here (bhtop pushes to `dtsunami`) to survive a box wipe.

**Base commit:** `tt-metal` `8e6f2bab553` (v0.74.0-dev20260621, `main`).

## Contents
- `tt_metal/programming_examples/contributed/bootloader/` — resident Tensix bootloader (all 5 baby RISCs), device-validated. See [tensix-bootloader] / `BOOTLOADER_PARITY_PLAN.md`.
- `tt_metal/programming_examples/contributed/jit/` — JIT kernel examples.
- `tests/.../data_movement/gather_scatter_3hop/` — the 3-hop NoC gather/scatter kernel + test.
- `tt_metal/hw/inc/dataflow_api.h` — patched dataflow header.
- `agg_bw.py` — bandwidth aggregation script.
- `tracked-modified.patch` — delta for the 3 tracked files (`sources.cmake`,
  `contributed/CMakeLists.txt`, `multicast.cpp`); the full post-edit content of those files is
  also copied here alongside.

## Restore onto a fresh tt-metal
```bash
cd ~/tt-metal && git checkout 8e6f2bab553
cp -r <this-dir>/tt_metal <this-dir>/tests <this-dir>/agg_bw.py ~/tt-metal/
# (the copied files ARE the final versions; tracked-modified.patch is reference only)
```
