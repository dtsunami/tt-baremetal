/* vec_virus.c — RVV (x280 512-bit vector) power-virus + per-instruction max-IPC probe,
 * now STEERABLE by instruction and with an exposed/settable seed.
 *
 * For each vector instruction CLASS we run a tight loop of 8 *independent* feedback chains
 * (v0..v7), each fed its own result back through a max-Hamming operand stream (seedA / seedB,
 * default 0xAAAAAAAA / 0x55555555) so every issue flips ~half the datapath bits — maximum
 * switching activity (the "virus") — while the 8-wide ILP hides the vector pipe's occupancy
 * so the loop runs at the unit's *throughput* (= max IPC). mcycle/minstret bracket each class.
 *
 * LIVE CONTROL via the DRAM command mailbox (bh_cmd(); host: `bhtop-l2cpu cmd <t> <h> <op> <a0>`):
 *   op 10 select_class : arg0 = class 0..11, or 0xFFFFFFFF = sweep ALL (default)
 *   op 11 set_seed     : arg0 = seedA  (seedB := ~seedA, so the max-Hamming pair is kept)
 *   op 12 mutate       : arg0 = 0 fixed seed | 1 auto-randomize (xorshift the seed every pass)
 *   op  4 park / op 5 run
 * Selecting a single class focuses ALL power on that one functional unit and refreshes only
 * its IPC; "all" runs the full sweep/table. The seed is echoed to telemetry and is settable,
 * so you can dial the toggle pattern (e.g. 0xFFFFFFFF/0x00000000 = max swing, or 0 = quiet).
 *
 * Each pass also calls bh_dump_vec() -> the host/cockpit can watch v0..v31 + vector CSRs live.
 *
 * Telemetry (this hart's window):
 *   [0]=passes [1]=VLEN [2]=vlenb [3]=misa [4]=class running [5]=NCLASSES [6]=vl@m1 [7]=vl@m4
 *   [ 8+i]=dcycles[i]  [24+i]=dinstret[i]  [40+i]=NOPS[i]   (i=0..11)
 *   [52]=active class (0xFFFF=all)  [53]=seedA  [54]=seedB  [55]=mailbox seq  [56]=mode */
#include <bh.h>

#define NCLASSES 12
#define ITERS    256u
#define NOPS     (ITERS * 32u)
#define ALL      0xFFFFFFFFu

#define SLOT_CYC(i) (8u  + (i))
#define SLOT_RET(i) (24u + (i))
#define SLOT_OPS(i) (40u + (i))

enum { OP_PARK = 4, OP_RUN = 5, OP_SELECT = 10, OP_SEED = 11, OP_MUTATE = 12 };

/* xorshift32 — cheap on-hart PRNG so the seed can mutate/randomize itself every pass */
static inline unsigned xs32(unsigned x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x ? x : 0x1234567u;            /* never let it stick at 0 */
}

#define VV8(op,a,b) \
 op " v0,v0," a "\n" op " v1,v1," b "\n" op " v2,v2," a "\n" op " v3,v3," b "\n" \
 op " v4,v4," a "\n" op " v5,v5," b "\n" op " v6,v6," a "\n" op " v7,v7," b "\n"
#define MA8(op,a,b) \
 op " v0," a "," b "\n" op " v1," a "," b "\n" op " v2," a "," b "\n" op " v3," a "," b "\n" \
 op " v4," a "," b "\n" op " v5," a "," b "\n" op " v6," a "," b "\n" op " v7," a "," b "\n"
#define RD8(op,s,i) \
 op " v0," s "," i "\n" op " v1," s "," i "\n" op " v2," s "," i "\n" op " v3," s "," i "\n" \
 op " v4," s "," i "\n" op " v5," s "," i "\n" op " v6," s "," i "\n" op " v7," s "," i "\n"
#define GA8 \
 "vrgather.vv v0,v8,v18\n vrgather.vv v1,v8,v18\n vrgather.vv v2,v8,v18\n vrgather.vv v3,v8,v18\n" \
 "vrgather.vv v4,v8,v18\n vrgather.vv v5,v8,v18\n vrgather.vv v6,v8,v18\n vrgather.vv v7,v8,v18\n"
#define X4(g) g g g g

#define MEASURE(idx, BODY) do {                                                  \
    u64 _c0,_c1,_i0,_i1;                                                          \
    TELE[4] = (idx);                                                             \
    __asm__ volatile(                                                            \
        ".option arch, +v\n"                                                    \
        "csrr %0, mcycle\n   csrr %2, minstret\n"                               \
        "li   t3, %4\n"                                                          \
        "1:\n" BODY                                                              \
        "addi t3, t3, -1\n   bnez t3, 1b\n"                                      \
        "csrr %1, mcycle\n   csrr %3, minstret\n"                               \
        : "=&r"(_c0), "=&r"(_c1), "=&r"(_i0), "=&r"(_i1)                         \
        : "i"(ITERS) : "t3", "memory");                                          \
    TELE[SLOT_CYC(idx)] = (u32)(_c1 - _c0);                                      \
    TELE[SLOT_RET(idx)] = (u32)(_i1 - _i0);                                      \
    TELE[SLOT_OPS(idx)] = NOPS;                                                  \
} while (0)

#define FM4_4 "vfmacc.vv v0,v16,v20\n vfmacc.vv v4,v16,v20\n vfmacc.vv v8,v16,v20\n vfmacc.vv v12,v16,v20\n"

int main(void) {
    volatile u32 *cmd = bh_cmd();
    unsigned hb = 0, vl1 = 0, vl4 = 0, mode = 1, mutate = 0;
    unsigned active = ALL, seedA = 0xAAAAAAAA, seedB = 0x55555555;
    unsigned last_seq = cmd[0];

    bh_vec_enable();
    TELE[2] = bh_vlenb(); TELE[1] = bh_vlenb() * 8;
    TELE[3] = (u32)BH_CSR_READ(misa); TELE[5] = NCLASSES;

    for (;;) {
        unsigned seq = bh_cmd_seq(cmd);              /* poll the doorbell (uncached scratch) */
        if (seq != last_seq) {
            last_seq = seq;
            unsigned op = cmd[1], a0 = cmd[2];
            if (op == OP_SELECT)      active = a0;
            else if (op == OP_SEED) { seedA = a0; seedB = ~a0; }
            else if (op == OP_MUTATE) mutate = a0;
            else if (op == OP_PARK)   mode = 0;
            else if (op == OP_RUN)    mode = 1;
        }
        if (mutate) { seedA = xs32(seedA); seedB = ~seedA; }   /* evolve the operand stream */
        TELE[52] = active; TELE[53] = seedA; TELE[54] = seedB; TELE[55] = last_seq;
        TELE[56] = mode;  TELE[57] = mutate;

        if (mode) {
            /* (re)seed operands from the live seed pair (host-settable) */
            __asm__ volatile(
                ".option arch, +v\n"
                "li a0, 256\n vsetvli %0, a0, e32, m1, ta, ma\n"
                "vmv.v.x v16, %1\n vmv.v.x v17, %2\n vid.v v18\n vmv.v.x v8, %1\n"
                : "=r"(vl1) : "r"(seedA), "r"(seedB) : "a0");
            TELE[6] = vl1;

            if (active == ALL || active == 0)  MEASURE(0,  X4(VV8("vadd.vv",  "v16","v17")));
            if (active == ALL || active == 1)  MEASURE(1,  X4(VV8("vsub.vv",  "v16","v17")));
            if (active == ALL || active == 2)  MEASURE(2,  X4(VV8("vxor.vv",  "v16","v17")));
            if (active == ALL || active == 3)  MEASURE(3,  X4(VV8("vsll.vv",  "v16","v17")));
            if (active == ALL || active == 4)  MEASURE(4,  X4(VV8("vmul.vv",  "v16","v17")));
            if (active == ALL || active == 5)  MEASURE(5,  X4(MA8("vmacc.vv", "v16","v17")));
            if (active == ALL || active == 6)  MEASURE(6,  X4(VV8("vfadd.vv", "v16","v17")));
            if (active == ALL || active == 7)  MEASURE(7,  X4(VV8("vfmul.vv", "v16","v17")));
            if (active == ALL || active == 8)  MEASURE(8,  X4(MA8("vfmacc.vv","v16","v17")));
            if (active == ALL || active == 9)  MEASURE(9,  X4(RD8("vredsum.vs","v16","v17")));
            if (active == ALL || active == 10) MEASURE(10, X4(GA8));
            if (active == ALL || active == 11) {
                u64 c0, c1, i0, i1;
                TELE[4] = 11;
                __asm__ volatile(
                    ".option arch, +v\n"
                    "li a0, 64\n vsetvli %4, a0, e32, m4, ta, ma\n"
                    "vmv.v.x v16, %6\n vmv.v.x v20, %7\n"
                    "csrr %0, mcycle\n csrr %2, minstret\n li t3, %5\n"
                    "1:\n" FM4_4 FM4_4 FM4_4 FM4_4 FM4_4 FM4_4 FM4_4 FM4_4
                    "addi t3, t3, -1\n bnez t3, 1b\n"
                    "csrr %1, mcycle\n csrr %3, minstret\n"
                    : "=&r"(c0), "=&r"(c1), "=&r"(i0), "=&r"(i1), "=r"(vl4)
                    : "i"(ITERS), "r"(seedA), "r"(seedB) : "a0", "t3", "memory");
                TELE[7] = vl4;
                TELE[SLOT_CYC(11)] = (u32)(c1 - c0);
                TELE[SLOT_RET(11)] = (u32)(i1 - i0);
                TELE[SLOT_OPS(11)] = NOPS;
            }
        }

        bh_dump_vec();               /* snapshot v0..v31 + vector CSRs for the cockpit */
        hb++; TELE[0] = hb;
    }
}
