#!/usr/bin/env bash
# Build one LLK perf kernel ON TOP OF llk_lib — the distilled, readable form of tt-llk's own build
# recipe (tests/python_tests/helpers/test_config.py). Each kernel is split across the three Tensix
# compute threads, so it compiles three times — once per component with -DLLK_TRISC_{UNPACK,MATH,PACK}
# — each pulling in its llk_lib headers (llk_unpack_*/llk_math_*/llk_pack_*).
#
#   Usage:  build.sh <kernel-name> [BUILD_H=path/to/build.h]
#   e.g.    build.sh eltwise_binary_fpu_perf
#
# Sources are the tracked canon copies here (kernels/tensix/llk/<name>/<name>.cpp); the include roots
# (llk_lib, common/inc, helpers, sfpi) come from the tt-llk tree under $TT_METAL_HOME. Produces one
# object per present thread in the gitignored working tree (~/bhtop/kernels/tensix/llk/_build),
# proving the kernel builds on llk_lib. Linking to per-thread ELFs + running on device is the
# tt-llk pytest path (`pytest --compile-producer` generates the per-variant build.h and links with
# helpers/ld/{unpack,math,pack}.ld + tmu-crt0.S); see README.md. Pure host (no device).
set -euo pipefail
NAME="${1:?usage: build.sh <kernel-name> [path-to-build.h]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
KDIR="$HERE/$NAME"
SRC="$KDIR/$NAME.cpp"
[ -f "$SRC" ] || { echo "no canon source: $SRC" >&2; exit 1; }

METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
TESTS="$METAL/tt_metal/tt-llk/tests"
LLK="$METAL/tt_metal/tt-llk/tt_llk_blackhole"
SFPI="${SFPI:-$METAL/runtime/sfpi}"
GPP="$SFPI/compiler/bin/riscv-tt-elf-g++"
OUT="${OUT:-$HOME/bhtop/kernels/tensix/llk/_build/$NAME}"
mkdir -p "$OUT"

# Per-variant config header (op / format / fidelity / tile-count). Defaults to the kernel's shipped
# build.example.h; override with a 2nd arg or BUILD_H=. The harness normally generates this.
BUILD_H="${2:-${BUILD_H:-$KDIR/build.example.h}}"
[ -f "$BUILD_H" ] || { echo "no build.h variant config: $BUILD_H (pass one as 2nd arg)" >&2; exit 1; }
VD="$(mktemp -d)"; cp "$BUILD_H" "$VD/build.h"; trap 'rm -rf "$VD"' EXIT

# Recipe lifted verbatim from test_config.py (BLACKHOLE compute build). The link uses ONLY tt-llk's
# own startup + linker scripts (helpers/src/trisc.cpp + helpers/ld/{<comp>,sections,memory}.ld +
# tmu-crt0.S) — NO llrt, NO jit_build. Each thread links to its own ELF.
# -DLLK_BOOT_MODE_TRISC: T0 (unpack) self-boots — runs device_setup() + clear_trisc_soft_reset() to
# release T1/T2 — so the cockpit can drive a run by deasserting only TRISC0 over exalens (no BRISC).
FLAGS="-mcpu=tt-bh-tensix -g -O3 -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit \
-ffast-math -fno-exceptions -fno-rtti -fno-use-cxa-atexit -mno-tt-fix-whbhebreak \
-DTENSIX_FIRMWARE -DENV_LLK_INFRA -DKERNEL_BUILD -DENABLE_LLK_ASSERT -DARCH_BLACKHOLE -DRUNTIME_FORMATS \
-DLLK_BOOT_MODE_TRISC"
LINK="-nostdlib -nostartfiles -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16"
INC="-I$VD -I$SFPI/include -I$LLK/llk_lib -I$LLK/common/inc -I$LLK/common/inc/sfpu \
-I$LLK/../common -I$METAL/tt_metal/hw/inc -I$TESTS/helpers/include -I$METAL/tt_metal/hostdevcommon/api \
-I$METAL/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$METAL/tt_metal/hw/ckernels/blackhole/metal/llk_api"
LDD="$TESTS/helpers/ld"
# the kernel source, addressable as <sources/..> or by abs path; trisc.cpp via helpers/src.
REL="$(python3 -c "import os;print(os.path.relpath('$SRC','$TESTS'))" 2>/dev/null || echo "$SRC")"

echo "building $NAME on llk_lib -> per-thread ELFs (build.h=$(basename "$BUILD_H"))"
rc=0
for pair in UNPACK:0 MATH:1 PACK:2; do
    comp="${pair%%:*}"; idx="${pair##*:}"
    grep -q "LLK_TRISC_$comp" "$SRC" || { printf "  %-7s — (not implemented, skipped)\n" "$comp"; continue; }
    # compile+link in one g++ invocation, wrapper from stdin (mirrors test_config.py build_kernel_part)
    if printf '#include <%s>\n#include <trisc.cpp>\n' "$REL" | \
       (cd "$TESTS" && "$GPP" $FLAGS -I. -Ihelpers/src $INC -DLLK_TRISC_"$comp" -DCOMPILE_FOR_TRISC="$idx" \
            $LINK -T"$LDD/memory.blackhole.ld" -T"$LDD/$(echo "$comp" | tr A-Z a-z).ld" -T"$LDD/sections.ld" \
            -x c++ - -lc -o "$OUT/$comp.elf" 2>"$OUT/$comp.err"); then
        sz=$("$SFPI/compiler/bin/riscv-tt-elf-size" "$OUT/$comp.elf" | awk 'NR==2{print $1"B text"}')
        printf "  %-7s -> %s.elf  %s  (on llk_lib)\n" "$comp" "$comp" "$sz"
    else
        printf "  %-7s FAILED (see %s)\n" "$comp" "$OUT/$comp.err"; head -4 "$OUT/$comp.err"; rc=1
    fi
done
[ $rc = 0 ] && echo "built -> $OUT  (per-thread ELFs; load onto TRISCs via exalens — see README)"
exit $rc
