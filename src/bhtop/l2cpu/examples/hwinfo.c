/* hwinfo.c — read real x280 hardware registers and surface them, using the harness.
 *
 *   bhtop-l2cpu load 0 0 src/bhtop/l2cpu/examples/hwinfo.c
 *   bhtop-l2cpu tele 0
 *
 * Shows the bhtop idea on the kernel side: the chip is named registers you just read.
 * Compare slot 1 (hartid) when you load this on hart 0 vs hart 1 — it differs. */
#include <bh.h>

int main(void) {
    u32 hb = 0;
    TELE[1] = bh_hartid();                       /* which hart am I (0..3) */
    TELE[5] = 0x1F0C0DE5;                         /* marker: "this kernel is running" */
    /* peek my own initial-PC register straight from the peripheral block: */
    TELE[6] = bh_rd32((u64)BH_RESET_VEC(bh_hartid()));   /* should read back BH_CODE_BASE */
    for (;;) {
        hb++;
        TELE[0] = hb;                            /* slot 0 = heartbeat (liveness)     */
        TELE[2] = (u32)bh_cycles();              /* free-running cycle counter (low32)*/
        TELE[3] = (u32)bh_instret();             /* instructions retired (low32)      */
        TELE[4] = (u32)(bh_cycles() >> 32);      /* cycle counter high32 (climbs slow)*/
    }
}
