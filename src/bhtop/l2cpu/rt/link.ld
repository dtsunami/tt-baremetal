/* bhtop.l2cpu flat-image linker script for the x280.
 * Places the entry (_start) first at LOAD_ADDR, packs rodata/data, reserves
 * bss + stack. Override the base with:  -Wl,--defsym=LOAD_ADDR=0x...
 * .bss is NOLOAD (not in the flat .bin) so crt0/_start must zero it. */
OUTPUT_FORMAT("elf64-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

LOAD_ADDR = DEFINED(LOAD_ADDR) ? LOAD_ADDR : 0x30008000;  /* toolchain passes --defsym=LOAD_ADDR=<regmap.CODE_ADDR> */
STACK_SIZE = 0x2000;          /* 8 KiB — keeps __stack_top under the writable GDDR window the minimal
                               * bringup leaves (no L3-cache config); ample for these freestanding kernels */

SECTIONS
{
    . = LOAD_ADDR;

    .text : ALIGN(4) {
        KEEP(*(.text._start))         /* entry must be first */
        *(.text .text.*)
    }
    .rodata : ALIGN(8) { *(.rodata .rodata.* .srodata .srodata.*) }
    .data   : ALIGN(8) { *(.data .data.* .sdata .sdata.*) }

    .bss (NOLOAD) : ALIGN(8) {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(8);
        __bss_end = .;
    }

    . = ALIGN(16);
    . = . + STACK_SIZE;
    __stack_top = .;

    /DISCARD/ : { *(.comment) *(.note .note.*) *(.eh_frame) *(.riscv.attributes) }
}
