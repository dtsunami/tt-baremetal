// SPDX-License-Identifier: Apache-2.0
//
// overlay_blink.cpp — a SAMPLE code overlay. This is NOT a metal kernel; it is a freestanding
// blob you compile yourself (riscv32, -nostdlib), linked to run at BL_CODE_SLOT_A (see
// overlay.ld), then stage into L1 over the NoC and trigger with BL_CMD_EXEC. The resident
// bootloader invalidates the i-cache and calls run(ctrl).
//
// It writes an incrementing counter into the telemetry region for `iters` rounds (taken from
// a live param) so you can watch it advance with `bhtop-tensix watch`, then returns.
//
// Build (find the toolchain in your sfpi install; see README):
//   ~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++ -Os -march=rv32im -mabi=ilp32 -nostdlib \
//       -ffreestanding -fno-exceptions -fno-rtti -T overlay.ld overlay_blink.cpp -o overlay_blink.elf
//   ~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy -O binary -j .text -j .rodata \
//       overlay_blink.elf overlay_blink.bin

#include <cstdint>

// keep in sync with bootloader_abi.h
static constexpr uint32_t TELEM_BASE = 0x00141000u;
enum { PARAM0 = 4 };   // word index off ctrl base: iters

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    volatile uint32_t* telem = (volatile uint32_t*)TELEM_BASE;
    uint32_t iters = ctrl[PARAM0];
    if (iters == 0) iters = 1000;
    for (uint32_t i = 0; i < iters; i++) {
        telem[0] = i;               // bhtop-tensix watch 1 2 0x141000 4 sees this advance
        telem[1] = i * i;
    }
    telem[2] = 0xC0FFEEu;           // done marker
    return iters;                   // -> stored in BL_OVL_RET
}
