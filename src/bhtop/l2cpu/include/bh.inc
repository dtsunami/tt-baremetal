# bh.inc — named addresses for x280 assembly kernels. Mirrors include/bh.h / regmap.py.
# Pull it in with:   .include "bh.inc"     (the loader puts include/ on the search path)
#
#   .include "bh.inc"
#   .section .text._start,"ax"; .globl _start
#   _start:
#       li    t0, BH_TELE_BASE         # hart 0 window
#       csrr  t2, mhartid              # who am I (0..3)
#       slli  t2, t2, 8                # * BH_TELE_STRIDE (0x100)
#       add   t0, t0, t2               # -> THIS hart's window (per-hart, no collision)
#       csrr  t1, mhartid
#       sw    t1, 0(t0)                # TELE[0] = hartid
#   1:  j 1b

    # ---- DRAM windows ----
    .equ BH_TRAMP_BASE,  0x30000000    # RNMI redirect trampoline
    .equ BH_CODE_BASE,   0x30001000    # your kernel loads + runs here
    .equ BH_TELE_BASE,   0x30002000    # telemetry: hart 0's 64-slot window
    .equ BH_TELE_STRIDE, 0x100         # per-hart stride; hart N window = BASE + N*0x100
    .equ BH_ARCH_BASE,   0x30003000    # arch-state dump: hart 0's register-file block
    .equ BH_ARCH_STRIDE, 0x200         # per-hart stride; hart N block = BASE + N*0x200

    # ---- L2CPU peripheral registers (per-hart = base + N*stride) ----
    .equ BH_RESET_VEC,   0x20010000    # +N*8   hart N initial PC
    .equ BH_HART_STATUS, 0x20010400    # all-harts run/halt status
    .equ BH_TRIGGER,     0x20010414    # write 1<<N -> RNMI on hart N
    .equ BH_RNMI_TRAP,   0x20010418    # +N*16  hart N RNMI vector
    .equ BH_RNMI_EXC,    0x20010420    # +N*16  hart N exception vector

    # ---- handy CSR numbers (csrr/csrw take the name directly on this assembler) ----
    .equ CSR_MHARTID,  0xF14
    .equ CSR_MCYCLE,   0xB00
    .equ CSR_MNSTATUS, 0x353
