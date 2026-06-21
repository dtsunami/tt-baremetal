// SPDX-License-Identifier: Apache-2.0
// rvv.h — harness for the X280 RVV (RISC-V Vector) example catalog: vector context init + a
// SAMPLED-TELEMETRY convention so the bhtop Hart-Lab cockpit can chart each instruction live.
//
// TWO RULES every RVV kernel here follows:
//   1. RVV_INIT() FIRST.  -march=rv64gcv -O2 auto-vectorizes ordinary scalar setup loops, so the
//      vector context (mstatus.VS) + chicken bits MUST be on before ANY other code — otherwise the
//      compiler-emitted vector ops trap and the kernel dies before main with no output.
//   2. BRANCHLESS compute.  The data path uses masks / predication (vmerge, vmax/vmin, vmadd…),
//      never a per-element if. Only the periodic telemetry publish has a (data-independent) branch.
//
// SAMPLED TELEMETRY (cockpit Plot tab charts these per hart; pick a slot, optionally rate mode):
//   TELE[0] = sample counter      (advancing => alive; rate = ops/s)
//   TELE[1] = ops per window      (= RVV_SAMPLE_N)
//   TELE[2] = milli-cycles / op   (measured live: 1000*dcycles/window)
//   TELE[3] = live result sample  (lane-0 or checksum — proves correctness while running)
//   TELE[4] = milli-instret / op  (1000*dinstret/window)
//   TELE[62],[63] = raw cycle/retired counters (bh_perf, for rate-mode plots)
#pragma once
#include <bh.h>
#include <riscv_vector.h>

#define RVV_INIT() do { __asm__ volatile("csrw 0x7c1, zero"); bh_vec_enable(); } while (0)

#ifndef RVV_SAMPLE_LOG
#define RVV_SAMPLE_LOG 16              // publish every 2^16 = 65536 ops
#endif
#define RVV_SAMPLE_N (1u << RVV_SAMPLE_LOG)

// Publish one telemetry sample. dcycles/dinstret are the deltas across the last window.
static inline void rvv_publish(unsigned long iter, unsigned long dcyc, unsigned long dret, u32 result) {
    TELE[0] = (u32)iter;
    TELE[1] = RVV_SAMPLE_N;
    TELE[2] = (u32)((dcyc * 1000u) >> RVV_SAMPLE_LOG);   // milli-cycles per op
    TELE[3] = result;
    TELE[4] = (u32)((dret * 1000u) >> RVV_SAMPLE_LOG);   // milli-instret per op
    bh_perf();
}

// The standard sampled driver: run `BODY` forever, publishing every RVV_SAMPLE_N iterations with
// `RESULT` (a u32 extracted from the demo's accumulator). BODY is branchless vector work.
#define RVV_RUN(BODY, RESULT)                                                        \
    do {                                                                             \
        unsigned long iter = 0, c0 = bh_cycles(), r0 = bh_instret();                 \
        for (;;) {                                                                   \
            BODY;                                                                    \
            if ((++iter & (RVV_SAMPLE_N - 1)) == 0) {                                \
                unsigned long c1 = bh_cycles(), r1 = bh_instret();                   \
                rvv_publish(iter, c1 - c0, r1 - r0, (RESULT));                       \
                c0 = c1; r0 = r1;                                                    \
            }                                                                        \
        }                                                                            \
    } while (0)

// 32 lanes of e16 is exactly one VLEN=512 register — the natural width for these demos.
static inline size_t rvv_vl32(void) { return __riscv_vsetvl_e16m1(32); }
