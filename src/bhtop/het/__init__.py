"""Heterogeneous bare-metal compute on Blackhole: the x280 (L2CPU RISC-V + RVV) and the Tensix grid
cooperating as co-equal dataflow peers over tt-exalens, with ZERO tt-metal, sharing GDDR.

The x280 owns the irregular tier (depth sort / gather / scatter-add); the Tensix grid owns the dense
tier (matmul-eval, raster, SFPU). Demonstrated end-to-end on a Gaussian-splatting trainer.

Substrate lives in `bhtop.tensix` (BareMetal launcher, matmul/sfpu/splat kernels, LLK boot lane) and
`bhtop.l2cpu` (x280 bringup). The C kernels are `bhtop/kernels/tensix/baremetal/` (Tensix cold-boot
canon) and `bhtop/kernels/x280/het/` (the x280 POC kernels). This package holds the orchestration
drivers and the interview artifact (`poc/`). See README.md and `../../../baremetal_plan.md`.
"""
