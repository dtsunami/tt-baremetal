/* vf_margin.c — SELF-CHECKING RVV stress kernel for V/F margining (all 4 x280 harts).
 *
 * Each pass runs a DETERMINISTIC max-Hamming datapath toggler: 8 independent chains of vmacc (the wide
 * integer multiplier = a long combinational path) + vadd/vxor (ALU), seeded from a fixed 0xAAAAAAAA /
 * 0x55555555 pair so every issue flips ~half the bits (max switching = a real power/dI-dt virus) while the
 * 8-wide ILP keeps the pipe saturated. All arithmetic wraps mod 2^32 -> the result is EXACTLY reproducible.
 * The 8 accumulators are stored + XOR-folded to a 32-bit checksum. The FIRST pass captures it as GOLDEN
 * (do this at a known-good operating point); every later pass COMPARES -> a datapath bit-error from an
 * undervolt/overclock bumps `errs` and records the first bad checksum. That is the margin fail signal.
 * The compute has no dependence on hartid, so all 4 harts converge to the SAME golden (a cross-hart sanity).
 *
 * Heartbeat = passes (TELE[0]); a frozen heartbeat = the hart WEDGED (hard fail, needs tt-smi -r 0).
 *
 * Telemetry (per-hart 64-slot window): [0]=passes [1]=golden [2]=last_chk [3]=errs [4]=first_bad
 *   [5]=golden_set [6]=hartid [7]=vlenb [8]=vl [9]=ITERS
 * Mailbox (bhtop-l2cpu cmd <t> <h> <op>): 4=park 5=run 6=reset-golden (clear golden+errs, recapture next). */
#include <bh.h>
#ifndef ITERS
#define ITERS 8192u
#endif
enum { OP_PARK = 4, OP_RUN = 5, OP_RESET = 6 };

int main(void) {
    volatile u32 *cmd = bh_cmd();
    unsigned last_seq = cmd[0];
    unsigned passes = 0, golden = 0, gset = 0, errs = 0, first_bad = 0, mode = 1;
    u32 buf[128];                                   /* 8 vregs * VLMAX(e32,m1)=16 = 128 words */

    bh_vec_enable();
    TELE[6] = bh_hartid(); TELE[7] = bh_vlenb(); TELE[9] = ITERS;

    for (;;) {
        unsigned seq = bh_cmd_seq(cmd);
        if (seq != last_seq) {
            last_seq = seq;
            unsigned op = cmd[1];
            if (op == OP_PARK)       mode = 0;
            else if (op == OP_RUN)   mode = 1;
            else if (op == OP_RESET) { gset = 0; errs = 0; first_bad = 0; }
        }
        if (mode) {
            unsigned vl = 0;
            __asm__ volatile(
                ".option arch, +v\n"
                "li a0, 256\n vsetvli %0, a0, e32, m1, ta, ma\n"
                "li t0, 0xAAAAAAAA\n vmv.v.x v16, t0\n"      /* max-Hamming operand A */
                "li t0, 0x55555555\n vmv.v.x v17, t0\n"      /* operand B (= ~A)      */
                "vid.v v18\n"                                /* per-lane variety      */
                /* seed 8 distinct, nonzero accumulators */
                "vadd.vv v0,v18,v16\n vxor.vv v1,v18,v17\n vsub.vv v2,v16,v18\n vadd.vv v3,v17,v18\n"
                "vxor.vv v4,v16,v17\n vadd.vv v5,v17,v18\n vsub.vv v6,v16,v17\n vmv.v.x v7,t0\n"
                "li t3, %2\n1:\n"                            /* ITERS tight loop */
                "vmacc.vv v0,v16,v17\n vmacc.vv v1,v17,v18\n vmacc.vv v2,v16,v18\n vmacc.vv v3,v18,v17\n"
                "vadd.vv  v4,v4,v16\n  vxor.vv  v5,v5,v17\n  vmacc.vv v6,v16,v16\n vadd.vv  v7,v7,v18\n"
                "addi t3,t3,-1\n bnez t3,1b\n"
                /* store v0..v7 (64 B each) -> buf, folded in C */
                "mv t1,%1\n"
                "vse32.v v0,(t1)\n addi t1,t1,64\n vse32.v v1,(t1)\n addi t1,t1,64\n"
                "vse32.v v2,(t1)\n addi t1,t1,64\n vse32.v v3,(t1)\n addi t1,t1,64\n"
                "vse32.v v4,(t1)\n addi t1,t1,64\n vse32.v v5,(t1)\n addi t1,t1,64\n"
                "vse32.v v6,(t1)\n addi t1,t1,64\n vse32.v v7,(t1)\n"
                : "=&r"(vl) : "r"(buf), "i"(ITERS) : "a0", "t0", "t1", "t3", "memory");

            unsigned chk = 0, n = vl * 8; if (n > 128) n = 128;
            for (unsigned i = 0; i < n; i++) chk ^= buf[i];
            if (!gset) { golden = chk; gset = 1; }
            else if (chk != golden) { if (!errs) first_bad = chk; errs++; }
            TELE[2] = chk; TELE[8] = vl;
        }
        passes++;
        TELE[0] = passes; TELE[1] = golden; TELE[3] = errs; TELE[4] = first_bad; TELE[5] = gset;
    }
}
