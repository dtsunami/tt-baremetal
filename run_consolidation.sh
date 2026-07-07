#!/bin/bash
set -e

# Create branch to save current state
git branch baremetal

# Create directories
mkdir -p tests/integration/gap0 \
         tests/integration/gap_next \
         tests/integration/resident \
         tests/integration/het \
         tests/integration/lever \
         tests/integration/core \
         scripts/diagnostics \
         scripts/benchmarks \
         src/prototypes \
         tests/fixtures/telemetry \
         outputs/profiler

# Gap0 tests
git mv scratchpad/test_gap0_*.py tests/integration/gap0/

# Gap_next tests
git mv scratchpad/test_gap1_*.py tests/integration/gap_next/
git mv scratchpad/test_gap2_*.py tests/integration/gap_next/
git mv scratchpad/test_gap5_*.py tests/integration/gap_next/

# Resident tests
git mv scratchpad/test_resident_*.py tests/integration/resident/

# Het tests
git mv scratchpad/test_het_*.py tests/integration/het/

# Lever tests
git mv scratchpad/test_lever*.py tests/integration/lever/

# Core/Misc tests
git mv scratchpad/test_baremetal_trainer.py tests/integration/core/
git mv scratchpad/test_cb_io.py tests/integration/core/
git mv scratchpad/test_cb_operands_verify.py tests/integration/core/
git mv scratchpad/test_handshake.py tests/integration/core/
git mv scratchpad/test_opt_step.py tests/integration/core/

# Diagnostics
git mv scratchpad/diag*.py scripts/diagnostics/
git mv scratchpad/dma_probe.py scripts/diagnostics/
git mv scratchpad/probe_ring.py scripts/diagnostics/
git mv scratchpad/validate_bin.py scripts/diagnostics/
git mv scratchpad/verify_*.py scripts/diagnostics/

# Benchmarks
git mv scratchpad/bench_exalens.py scripts/benchmarks/
git mv scratchpad/multitile_render.py scripts/benchmarks/
git mv scratchpad/profile_real.py scripts/benchmarks/

# Prototypes
git mv scratchpad/gap1_proj* src/prototypes/
git mv scratchpad/proj* src/prototypes/
git mv scratchpad/proto_*.py src/prototypes/
git mv scratchpad/gap2_bin_golden.py src/prototypes/

# Telemetry Data
git mv scratchpad/*telemetry*.json tests/fixtures/telemetry/

# Profiler HTML
git mv scratchpad/*.html outputs/profiler/

# Clean up empty directory
rm -rf scratchpad

# Commit changes
git add .
git commit -m "chore: Consolidate scratchpad into tests, scripts, src/prototypes, and outputs"

echo "Consolidation complete!"
