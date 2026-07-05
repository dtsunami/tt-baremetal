# SPDX-License-Identifier: Apache-2.0
# crt0 — bare-metal Tensix baby-RISC entry. Set sp, pull 4 params from the L1 arg block, call
# bm_main(a0..a3), park. Placed first (.text.start) so _start sits at L1 0x0.
.section .text.start, "ax"
.global _start
_start:
    li   sp, 0x0003FF00
    li   t0, 0x00001000
    lw   a0, 0(t0)
    lw   a1, 4(t0)
    lw   a2, 8(t0)
    lw   a3, 12(t0)
    call bm_main
1:  j    1b
