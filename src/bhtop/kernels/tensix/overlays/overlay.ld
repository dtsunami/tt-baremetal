/* SPDX-License-Identifier: Apache-2.0
 * overlay.ld — link an overlay at the canonical code slot A (BRISC region = 0x102000). run()
 * lands at the slot base so the bootloader's function-pointer call enters it. Overlays are tiny,
 * self-contained, and telemetry/scratch are ctrl-RELATIVE (overlay.h), so the same .bin runs
 * correctly at any RISC's slot A even though they sit at different addresses (0x102000 BRISC,
 * 0x112000 NCRISC, 0x122000/0x132000/0x142000 TRISC0-2). Slot A is 24 KiB; slot B is scratch.
 */
ENTRY(run)
MEMORY { SLOT (rx) : ORIGIN = 0x00102000, LENGTH = 0x00006000 }
SECTIONS {
    .text : { KEEP(*(.text.entry)) *(.text*) *(.rodata*) } > SLOT
    .bss (NOLOAD) : { *(.bss*) *(COMMON) } > SLOT
}
