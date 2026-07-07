#!/bin/bash
set -e

# Make docs dir
mkdir -p docs/plans

# Move md files (except README.md)
git mv BOOTLOADER_PARITY_PLAN.md docs/plans/
git mv DRAM_NUMA.md docs/plans/
git mv FUSED_TO_TRAIN.md docs/plans/
git mv LABS_UNIFICATION_SCOPE.md docs/plans/
git mv RESIDENT_GRID.md docs/plans/
git mv RESIDENT_SCALE.md docs/plans/
git mv TTMETAL_PLAN.md docs/plans/
git mv baremetal_plan.md docs/plans/
git mv l2cpu-copilot-plan.md docs/plans/

# CRT cleanup
mkdir -p tests/integration/crt
mkdir -p scripts/design/crt
mkdir -p src/prototypes/crt

git mv crt/run_x280.py tests/integration/crt/
git mv crt/sweep.py scripts/design/crt/
git mv crt/crt_matmul.py src/prototypes/crt/
git mv crt/kernel/* src/prototypes/crt/
rmdir crt/kernel
rmdir crt

# update git
git add .
git commit -m "chore: clean up markdown plans and restructure crt"
git push origin main
echo "Done"
