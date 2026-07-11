/* SPDX-License-Identifier: Apache-2.0
 *
 * overlay.ld — link a code overlay to run at a FIXED L1 address (BL_CODE_SLOT_A).
 * Entry (run) is placed first so the slot base == the function pointer the bootloader calls.
 * This is the software analogue of metal's kernel_text_offset / TRISC reset-PC override:
 * fixed-address placement instead of position-independent code.
 *
 * To target slot B, change ORIGIN to 0x00160000 and stage there.
 */
ENTRY(run)

MEMORY {
    SLOT (rx) : ORIGIN = 0x00150000, LENGTH = 0x00010000   /* BL_CODE_SLOT_A, 64 KiB */
}

SECTIONS {
    .text : {
        KEEP(*(.text.entry))   /* run() lands at slot base */
        *(.text*)
        *(.rodata*)
    } > SLOT

    /* Overlays are .text/.rodata only; no .data/.bss (use the telemetry/param region for
     * mutable state). Flag if the linker reports non-empty .data/.bss. */
    .bss (NOLOAD) : { *(.bss*) *(COMMON) } > SLOT
}
