#!/usr/bin/env bash
# Build all bhtop bootloader overlays -> $OUT/*.bin (raw, ready to bl-stage).
# Sources are folder-per-kernel canon ({name}/{name}.c) + the shared overlay.h / overlay.ld here.
# Output goes to the gitignored per-user working tree (~/bhtop/kernels/tensix/overlays/_build),
# matching overlays.py BUILD_DIR — so built binaries are never committed. Pure host (no device).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SFPI="${SFPI:-$HOME/tt-metal/runtime/sfpi}"
GPP="$SFPI/compiler/bin/riscv-tt-elf-g++"
OC="$SFPI/compiler/bin/riscv-tt-elf-objcopy"
OUT="${OUT:-$HOME/bhtop/kernels/tensix/overlays/_build}"
mkdir -p "$OUT"

CFLAGS="-Os -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -fno-exceptions -fno-rtti -I$HERE"
for src in counter l1bw matrix sfpu; do
    "$GPP" $CFLAGS -T "$HERE/overlay.ld" "$HERE/$src/$src.c" -o "$OUT/$src.elf"
    "$OC" -O binary -j .text -j .rodata "$OUT/$src.elf" "$OUT/$src.bin"
    sz=$(stat -c%s "$OUT/$src.bin")
    h=$(sha256sum "$OUT/$src.bin" | cut -c1-12)
    entry=$("$SFPI/compiler/bin/riscv-tt-elf-nm" "$OUT/$src.elf" | grep ' T run' | cut -d' ' -f1)
    printf "  %-8s %4d B  sha %s  run@0x%s\n" "$src" "$sz" "$h" "$entry"
done
echo "overlays built -> $OUT"
