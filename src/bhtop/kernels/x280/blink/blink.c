/* blink.c — the simplest "hello": flip slot 1 between 0 and 1 at a visible rate, so you
 * can watch it blink in the telemetry view. Shows a time delay using the cycle counter. */
#include <bh.h>

int main(void) {
    unsigned on = 0, hb = 0;
    for (;;) {
        on ^= 1;               /* toggle 0 <-> 1 */
        TELE[1] = on;          /* watch this blink in the Telemetry tab */
        TELE[0] = ++hb;        /* heartbeat */
        bh_spin(20000000);     /* wait ~tens of ms (busy-wait on mcycle) so it's visible */
    }
}
