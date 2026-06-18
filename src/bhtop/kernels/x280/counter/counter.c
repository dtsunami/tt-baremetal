/* counter.c — minimal x280 bare-metal in C. Just write main(); crt0 handles _start.
 *   bhtop-l2cpu load 0 0 <path>/counter.c
 *   bhtop-l2cpu tele 0          # slot0 = heartbeat, slot1 = running sum, slot2 = squares
 */
#include <tele.h>

int main(void) {
    unsigned hb = 0, sum = 0;
    for (;;) {
        hb++;
        sum += hb;
        TELE[0] = hb;            /* liveness */
        TELE[1] = sum;           /* a metric you inserted */
        TELE[2] = hb * hb;       /* another */
    }
    return 0;
}
