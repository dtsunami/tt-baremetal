# heartbeat.s — minimal x280 bare-metal: bump telemetry slot 0 forever.
# asm provides its own _start (no crt0). Load + run with:
#   bhtop-l2cpu load 0 0 <path>/heartbeat.s
# then watch it:  bhtop-l2cpu tele 0
    .option norvc
    .section .text._start, "ax", @progbits
    .globl _start
_start:
    lui   t0, 0x30002            # TELE_BASE = 0x30002000 (telemetry block)
    li    t1, 0
1:  addi  t1, t1, 1
    sw    t1, 0(t0)              # TELE[0] = heartbeat
    j     1b
