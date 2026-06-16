/* perf.c — surface this hart's cycle + instructions-retired counters every loop, so the
 * cockpit's Plot tab can chart throughput (retired/sec) over time, per hart.
 *
 * Try: Deploy this to ALL harts (Deploy all), open the Plot tab, pick slot 63, turn on
 * "rate" — you'll see instructions-retired-per-second for each hart over time.
 *
 *   slot 0 = heartbeat   slot 62 = cycles (low32)   slot 63 = instructions retired (low32) */
#include <bh.h>

int main(void) {
    unsigned hb = 0;
    for (;;) {
        hb++;
        TELE[0] = hb;     /* heartbeat */
        bh_perf();        /* writes slots 62 (cycles) + 63 (retired) */
    }
}
