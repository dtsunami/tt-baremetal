/* vf_margin_fp.c — SELF-CHECKING FP-FMA stress kernel for V/F margining (all 4 x280 harts).
 *
 * The integer-multiplier virus (vf_margin.c) ran clean at 1750/800mV while REAL training wedged at 711mV
 * -> the int path is not the worst timing path. This exercises the FP FMA (vfmacc: multiply + align + add +
 * normalize + round = the deepest FP combinational path) at FULL THROUGHPUT (8 independent chains keep the
 * FMA pipe saturated -> max activity -> the IR-droop that turns a marginal-voltage timing slack into a wrong
 * result). Each iter the addend is refreshed from a vector xorshift LFSR (toggling) and is ~O(10) while the
 * accumulators grow to ~O(1e3-1e4) -> a large exponent difference forces big mantissa ALIGNMENT shifts every
 * FMA (the align/normalize path). All ops are deterministic FP -> the folded checksum is exactly reproducible
 * and captured as a golden at a safe point; any bit-error from undervolt/overclock bumps `errs`.
 * hartid does not enter the compute -> all 4 harts share one golden (cross-hart sanity).
 *
 * Telemetry: [0]=passes [1]=golden [2]=last_chk [3]=errs [4]=first_bad [5]=golden_set [6]=hartid
 *   [7]=vlenb [8]=vl [9]=ITERS   Mailbox: 4=park 5=run 6=reset-golden. */
#include <bh.h>
#ifndef ITERS
#define ITERS 2048u
#endif
enum { OP_PARK = 4, OP_RUN = 5, OP_RESET = 6 };

int main(void) {
    volatile u32 *cmd = bh_cmd();
    unsigned last_seq = cmd[0];
    unsigned passes = 0, golden = 0, gset = 0, errs = 0, first_bad = 0, mode = 1;
    u32 buf[160];                                   /* 8 FP acc + 1 int + 1 lfsr = 10 vregs * 16 = 160 */

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
                /* constants */
                "li t0, 0x31800000\n vmv.v.x v16, t0\n"      /* 2^-28  (uint->addend scale) */
                "li t0, 0x3F000000\n vmv.v.x v17, t0\n"      /* 0.5    (FMA multiplicand)   */
                "li t0, 0x3D800000\n vmv.v.x v23, t0\n"      /* 2^-4   (seed scale)         */
                "li t0, 0x3F800000\n vmv.v.x v24, t0\n"      /* 1.0    (per-chain offset)   */
                "vid.v v22\n"
                /* LFSR state = seed + lane */
                "li t0, 0x2545F491\n vmv.v.x v20, t0\n vadd.vv v20,v20,v22\n"
                /* seed 8 distinct FP accumulators: v0=(lane/16), v1=v0+1, ... */
                "vfcvt.f.xu.v v0,v22\n vfmul.vv v0,v0,v23\n"
                "vfadd.vv v1,v0,v24\n vfadd.vv v2,v1,v24\n vfadd.vv v3,v2,v24\n vfadd.vv v4,v3,v24\n"
                "vfadd.vv v5,v4,v24\n vfadd.vv v6,v5,v24\n vfadd.vv v7,v6,v24\n"
                "vxor.vv v8,v8,v8\n"                         /* int checksum chain = 0 */
                "li t3, %2\n1:\n"
                /* xorshift32 the LFSR (v19 = temp) */
                "vsll.vi v19,v20,13\n vxor.vv v20,v20,v19\n"
                "vsrl.vi v19,v20,17\n vxor.vv v20,v20,v19\n"
                "vsll.vi v19,v20,5\n  vxor.vv v20,v20,v19\n"
                /* addend = (float)lfsr * 2^-28  (~O(10), small vs the growing accumulators) */
                "vfcvt.f.xu.v v18,v20\n vfmul.vv v18,v18,v16\n"
                /* 8 independent FP-FMA chains: v_k += 0.5 * addend  (full-throughput FMA activity) */
                "vfmacc.vv v0,v17,v18\n vfmacc.vv v1,v17,v18\n vfmacc.vv v2,v17,v18\n vfmacc.vv v3,v17,v18\n"
                "vfmacc.vv v4,v17,v18\n vfmacc.vv v5,v17,v18\n vfmacc.vv v6,v17,v18\n vfmacc.vv v7,v17,v18\n"
                "vxor.vv v8,v8,v20\n"                        /* fold the toggling LFSR into the int chain */
                "addi t3,t3,-1\n bnez t3,1b\n"
                /* store v0..v7 (FP) + v8 (int) + v20 (lfsr) -> buf, folded in C */
                "mv t1,%1\n"
                "vse32.v v0,(t1)\n addi t1,t1,64\n vse32.v v1,(t1)\n addi t1,t1,64\n"
                "vse32.v v2,(t1)\n addi t1,t1,64\n vse32.v v3,(t1)\n addi t1,t1,64\n"
                "vse32.v v4,(t1)\n addi t1,t1,64\n vse32.v v5,(t1)\n addi t1,t1,64\n"
                "vse32.v v6,(t1)\n addi t1,t1,64\n vse32.v v7,(t1)\n addi t1,t1,64\n"
                "vse32.v v8,(t1)\n addi t1,t1,64\n vse32.v v20,(t1)\n"
                : "=&r"(vl) : "r"(buf), "i"(ITERS)
                : "a0", "t0", "t1", "t3", "memory");

            unsigned chk = 0, n = vl * 10; if (n > 160) n = 160;
            for (unsigned i = 0; i < n; i++) chk ^= buf[i];
            if (!gset) { golden = chk; gset = 1; }
            else if (chk != golden) { if (!errs) first_bad = chk; errs++; }
            TELE[2] = chk; TELE[8] = vl;
        }
        passes++;
        TELE[0] = passes; TELE[1] = golden; TELE[3] = errs; TELE[4] = first_bad; TELE[5] = gset;
    }
}
