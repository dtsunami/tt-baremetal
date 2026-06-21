// SPDX-License-Identifier: Apache-2.0
// counter — the "hello world" overlay. Spins an accumulator PARAM0 times and reports cycles.
// Exercises the BRISC integer pipe + the cycle-telemetry path. Always safe.
#include "overlay.h"

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t n = ovl_param(ctrl, 0);
    if (n == 0) n = 1000000;
    uint32_t c0 = ovl_cycles();
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc += i * 2654435761u;          // keep it from being optimized away
        asm volatile("" : "+r"(acc));
    }
    uint32_t c1 = ovl_cycles();
    ovl_publish(ctrl, n, c1 - c0, acc);
    return n;
}
