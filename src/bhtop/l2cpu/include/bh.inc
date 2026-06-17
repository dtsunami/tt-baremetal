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

    # ---- DRAM windows (canonical map = regmap.py; the lab toolchain injects these via
    #      -Wa,--defsym on every build, so they never drift. The .ifndef fallbacks below are
    #      only for a standalone assemble; an injected --defsym value always wins.) ----
    .ifndef BH_TRAMP_BASE
    .equ BH_TRAMP_BASE,  0x30000000    # RNMI redirect trampoline
    .endif
    .ifndef BH_CODE_BASE
    .equ BH_CODE_BASE,   0x30008000    # your kernel loads + runs here (above the data blocks)
    .endif
    .ifndef BH_TELE_BASE
    .equ BH_TELE_BASE,   0x30002000    # telemetry: hart 0's 64-slot window
    .endif
    .ifndef BH_TELE_STRIDE
    .equ BH_TELE_STRIDE, 0x100         # per-hart stride; hart N window = BASE + N*0x100
    .endif
    .ifndef BH_ARCH_BASE
    .equ BH_ARCH_BASE,   0x30003000    # arch-state dump: hart 0's register-file block
    .endif
    .ifndef BH_ARCH_STRIDE
    .equ BH_ARCH_STRIDE, 0x200         # per-hart stride; hart N block = BASE + N*0x200
    .endif

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
