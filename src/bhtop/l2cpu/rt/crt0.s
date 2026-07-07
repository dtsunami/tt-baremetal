# bhtop.l2cpu C runtime: _start for freestanding C. Sets sp, zeroes .bss, calls
# main(), then parks. Linked first (.text._start) so it sits at LOAD_ADDR. With this,
# a C "kernel" is just:  int main(void) { ... }
#
# MULTI-HART: hart 0 uses the linked __stack_top (single-hart kernels are BYTE-FOR-BYTE UNCHANGED). Harts >0
# get their OWN 32 KiB stack in dedicated GDDR (0x30070000 + hart*0x8000) so co-running harts don't clobber
# each other's stack, and they skip the .bss zero (hart 0 already did it). Region is otherwise-unused GDDR.
    .option norvc
    .section .text._start, "ax", @progbits
    .globl _start
    .type  _start, @function
    .equ   HART_STK_BASE, 0x30070000    # per-hart stack region (hart h top = BASE + h*0x8000, grows down)
_start:
    csrr    t2, mhartid
    la      sp, __stack_top          # hart 0 default (absolute; no-PIC, --no-relax)
    li      t0, 0x6000               # mstatus.FS=Dirty: enable the FPU BEFORE main. The compiler may emit
    csrs    mstatus, t0              # FP register saves (fsd) in a kernel's PROLOGUE, which run before main's
                                     # body could enable FP; harmless (no-op) for int kernels.
    bnez    t2, 5f                   # hart > 0 -> per-hart stack, skip bss-zero
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f               # zero bss (8-byte aligned by link.ld) — hart 0 only
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  call    main
3:  wfi                              # main returned -> park
    j       3b
5:  li      t3, HART_STK_BASE        # hart > 0: sp = BASE + hartid*0x8000 (its own 32 KiB), then main
    slli    t4, t2, 15               # hartid * 0x8000
    add     sp, t3, t4
    call    main
6:  wfi
    j       6b
    .size _start, .-_start
