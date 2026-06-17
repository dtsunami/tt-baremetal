/* mailbox.c — DRAM-triggered live register update on the x280 (poll / keypress).
 *
 * The host can't poke an x280 CSR/GPR/vector register over the NoC — only this hart can,
 * from its own code. So we make a cooperative DOORBELL: the host writes a value into this
 * hart's command mailbox (uncached peripheral scratch, so it lands with no flush) and bumps a seq word;
 * we poll it every loop and, when seq changes, APPLY the value to a real register. This is
 * the light, non-preemptive counterpart to the RNMI redirect (which swaps code, host-forced).
 *
 * "Poll" = this loop. "Keypress" = the host simply rings the doorbell from a key handler:
 *     bhtop-l2cpu cmd <tile> <hart> <op> <arg0>      # one-shot
 *     L2cpu.command(tile, hart, op, arg0)            # from the cockpit / a key binding
 *
 * Commands (cmd[1]=op, cmd[2]=arg0):
 *   1 set_csr   : csrw mscratch, arg0      -> updates a CSR        (echoed in slot 4)
 *   2 set_vreg  : vmv.v.x v16, arg0        -> updates a vector reg (echoed in slot 5)
 *   3 set_vtype : 0=e32/m1, 1=e32/m4       -> updates vtype        (vl echoed in slot 7)
 *   4 park   5 run   0 nop
 *
 * Telemetry:
 *   [0]=poll count  [1]=last seq acked  [2]=last op  [3]=last arg0
 *   [4]=mscratch (live)  [5]=v16[0] (live)  [7]=vl  [8]=mode(1=run,0=park) */
#include <bh.h>

enum { OP_NOP = 0, OP_SET_CSR = 1, OP_SET_VREG = 2, OP_SET_VTYPE = 3, OP_PARK = 4, OP_RUN = 5 };

int main(void) {
    volatile u32 *cmd = bh_cmd();
    unsigned hb = 0, mode = 1, vl = 0;
    unsigned last_seq = cmd[0];          /* baseline: ignore whatever was already in the box */

    bh_vec_enable();
    __asm__ volatile(".option arch, +v\n li a0,256\n vsetvli %0,a0,e32,m1,ta,ma"
                     : "=r"(vl) : : "a0");   /* a vtype so vmv.v.x / vmv.x.s are valid */
    for (unsigned i = 1; i < 9; i++) TELE[i] = 0;   /* clean baseline (no stale DRAM echoes) */

    for (;;) {
        unsigned seq = bh_cmd_seq(cmd);              /* poll the doorbell (uncached scratch) */
        if (seq != last_seq) {                       /* host rang it -> a new command */
            last_seq = seq;
            unsigned op = cmd[1], a0 = cmd[2];
            switch (op) {
            case OP_SET_CSR:                         /* update a CSR from a DRAM-supplied value */
                __asm__ volatile("csrw mscratch, %0" : : "r"(a0));
                break;
            case OP_SET_VREG:                        /* update a vector register, live */
                __asm__ volatile(".option arch, +v\n vmv.v.x v16, %0" : : "r"(a0));
                break;
            case OP_SET_VTYPE:                       /* update vtype (LMUL) — re-vsetvli */
                if (a0 == 0)
                    __asm__ volatile(".option arch, +v\n li a0,256\n vsetvli %0,a0,e32,m1,ta,ma"
                                     : "=r"(vl) : : "a0");
                else
                    __asm__ volatile(".option arch, +v\n li a0,256\n vsetvli %0,a0,e32,m4,ta,ma"
                                     : "=r"(vl) : : "a0");
                break;
            case OP_PARK: mode = 0; break;
            case OP_RUN:  mode = 1; break;
            default: break;
            }
            TELE[1] = seq; TELE[2] = op; TELE[3] = a0;   /* ack: which command we consumed */
        }

        /* echo the LIVE register state every loop, so the host sees the update persist */
        unsigned ms, e0;
        __asm__ volatile("csrr %0, mscratch" : "=r"(ms));
        __asm__ volatile(".option arch, +v\n vmv.x.s %0, v16" : "=r"(e0));
        TELE[4] = ms; TELE[5] = e0; TELE[7] = vl; TELE[8] = mode;
        hb++; TELE[0] = hb;
    }
}
