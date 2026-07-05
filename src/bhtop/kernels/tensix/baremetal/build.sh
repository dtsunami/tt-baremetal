#!/usr/bin/env bash
# Build all bare-metal Tensix cold-boot kernels -> $OUT/*.bin (crt0 + {name}/{name}.c, linked at L1 0x0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SFPI="${SFPI:-$HOME/tt-metal/runtime/sfpi}"
GCC="$SFPI/compiler/bin/riscv-tt-elf-gcc"; OC="$SFPI/compiler/bin/riscv-tt-elf-objcopy"; NM="$SFPI/compiler/bin/riscv-tt-elf-nm"
OUT="${OUT:-$HOME/bhtop/kernels/tensix/baremetal/_build}"; mkdir -p "$OUT"
CF="-march=rv32im -mabi=ilp32 -Os -nostdlib -ffreestanding -fno-exceptions -fno-rtti -I$HERE"
for k in hello nocread; do
  "$GCC" $CF -T "$HERE/link.ld" "$HERE/crt0.s" "$HERE/$k/$k.c" -o "$OUT/$k.elf"
  "$OC" -O binary -j .text -j .rodata "$OUT/$k.elf" "$OUT/$k.bin"
  printf "  %-10s %4d B  _start@0x%s\n" "$k" "$(stat -c%s "$OUT/$k.bin")" "$($NM "$OUT/$k.elf" | grep ' _start' | cut -d' ' -f1)"
done
echo "bare-metal kernels -> $OUT"
