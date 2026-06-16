/* dumpstate.c — snapshot this hart's WHOLE register file to DRAM every loop, so the
 * cockpit "Arch" tab (or `peek <tile> 0x30003000`) can show all 32 registers + CSRs.
 *
 *   deploy this, then open the Arch tab and hit ⟳ — you'll see sp (the stack pointer,
 *   ~0x30005xxx), mhartid (which hart), mcycle (a live timer), ra (return address), etc.
 *
 * The host can't read a CPU's registers directly, so bh_dump_state() is how a hart
 * surfaces its architectural state. */
#include <bh.h>

int main(void) {
    unsigned hb = 0;
    for (;;) {
        hb++;
        TELE[0] = hb;            /* heartbeat (liveness) */
        bh_dump_state();         /* <-- snapshot registers -> Arch tab */
    }
}
