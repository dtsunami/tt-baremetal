/* scan.c — walk a block of DRAM, XOR-checksum it, and report. A real loop over memory
 * with live progress, so you can see the hart doing actual work in the telemetry view.
 *
 *   slot 0 = passes completed   slot 1 = checksum   slot 2 = progress (word index) */
#include <bh.h>

int main(void) {
    volatile unsigned *mem = (volatile unsigned *)0x30010000; /* some uncached DRAM */
    const unsigned n = 4096;                                  /* words to scan */
    for (;;) {
        unsigned sum = 0;
        for (unsigned i = 0; i < n; i++) {
            sum ^= mem[i];
            if ((i & 0x3FF) == 0) TELE[2] = i;     /* update progress every 1024 words */
        }
        TELE[0]++;            /* one full pass done */
        TELE[1] = sum;        /* the checksum of the block */
    }
}
