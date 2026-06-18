# heartbeat.s — minimal x280 bare-metal: bump THIS hart's telemetry slot 0 forever.
# asm provides its own _start (no crt0). Load + run with:
#   bhtop-l2cpu load 0 0 <path>/heartbeat.s
# then watch it:  bhtop-l2cpu tele 0 0   (tile 0, hart 0)
    .include "bh.inc"
    .option norvc
    .section .text._start, "ax", @progbits
    .globl _start
_start:
    li    t0, BH_TELE_BASE       # hart 0's telemetry window base
    csrr  t2, mhartid            # which hart am I (0..3)
    slli  t2, t2, 8              # * BH_TELE_STRIDE (0x100)
    add   t0, t0, t2             # -> THIS hart's window (no collision with other harts)
    li    t1, 0
1:  addi  t1, t1, 1
    sw    t1, 0(t0)              # TELE[0] = heartbeat
    j     1b
