// SPDX-License-Identifier: Apache-2.0
// l1bw — L1 memory-bandwidth exerciser. Copies a scratch buffer src->dst REPS times and reports
// cycles, so the cockpit can show bytes/cycle (× clock = GB/s). Exercises the BRISC load/store
// path against local L1. Always safe (pure L1, no NoC, no tensix backend).
//   PARAM0 = words per buffer (<= 0x1000 = 16 KiB)   PARAM1 = repetitions
#include "overlay.h"

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t words = ovl_param(ctrl, 0);
    uint32_t reps = ovl_param(ctrl, 1);
    if (words == 0 || words > 0x1000) words = 0x1000;   // 16 KiB
    if (reps == 0) reps = 2000;

    volatile uint32_t* src = ovl_scratch(ctrl);            // slot B (32 KiB) doubles as scratch
    volatile uint32_t* dst = ovl_scratch(ctrl) + 0x1000u;  // + 16 KiB (src+dst = 32 KiB = slot B)
    for (uint32_t i = 0; i < words; i++) src[i] = i * 2654435761u;

    uint32_t c0 = ovl_cycles();
    uint32_t sum = 0;
    for (uint32_t r = 0; r < reps; r++)
        for (uint32_t i = 0; i < words; i++) { uint32_t v = src[i]; dst[i] = v; sum += v; }
    uint32_t c1 = ovl_cycles();

    uint32_t total_words = words * reps;
    ovl_publish(ctrl, total_words, c1 - c0, sum);
    ovl_telem(ctrl)[4] = words;     // bytes/copy detail for the cockpit
    ovl_telem(ctrl)[5] = reps;
    return total_words;
}
