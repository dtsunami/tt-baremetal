// SPDX-License-Identifier: Apache-2.0
// nocread — bare-metal Blackhole NOC0 read overlay (the M1 keystone).
//
// Reads `len` bytes from a remote NoC endpoint (coord=PARAM0, local src=PARAM1) into this core's L1
// scratch, then publishes the payload to telemetry. This is tt-metal's noc_async_read distilled to
// ~11 command-register pokes with ZERO tt-metal includes — the first reusable brick of the
// tt-metal-free harness. Extracted from blackhole/noc_parameters.h + noc_nonblocking_api.h.
//
// M1 use: point it at the L2CPU tile the x280 wrote — PARAM0=(8,3)=0xC8, PARAM1=0x30002000 — and
// prove a bare-metal Tensix core reads what the x280 produced. Params are live-pokeable, so the
// same binary serves M2's circular-buffer consumer without recompile.
#include "overlay.h"

#define R(a)            (*(volatile uint32_t*)(uintptr_t)(a))
// ---- NOC0 BRISC read command buffer (index 1): 0xFFB20000 + (1<<11) = 0xFFB20800 ----
#define NOC_TARG_LO     0xFFB20800u   // remote source, low 32b
#define NOC_TARG_MID    0xFFB20804u   // remote source, bits[35:32]
#define NOC_TARG_COORD  0xFFB20808u   // remote (y<<6)|x  (= TARG_ADDR_HI)
#define NOC_RET_LO      0xFFB2080Cu   // local dest (my L1)
#define NOC_RET_MID     0xFFB20810u
#define NOC_RET_COORD   0xFFB20814u   // my own (y<<6)|x  (= RET_ADDR_HI)
#define NOC_CTRL_REG    0xFFB2081Cu   // command field
#define NOC_AT_LEN_BE   0xFFB20820u   // transfer length, bytes
#define NOC_CMD_CTRL    0xFFB20840u   // fire = 1, ready when reads back 0
#define NOC_NODE_ID     0xFFB20844u   // my own coordinate (y<<6)|x in low 12b
#define NOC_RD_RESP_CNT 0xFFB20208u   // NIU_MST_RD_RESP_RECEIVED status counter (NOC0)
#define NIU_CFG_0       0xFFB20100u   // bit14 = NOC_ID_TRANSLATE_EN

// CTRL read command = CPY | RD | RESP_MARKED(0x10) | VC_STATIC(0x80) | STATIC_VC(1)=0x2000
#define NOC_CTRL_READ   0x00002090u
#define NOC_SEND_REQ    0x00000001u

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t coord = ovl_param(ctrl, 0);              // remote target (y<<6)|x   e.g. (8,3)=0xC8
    uint32_t src   = ovl_param(ctrl, 1);              // remote local addr        e.g. 0x30002000
    uint32_t len   = ovl_param(ctrl, 2);              // bytes (keep 16B aligned)
    uint32_t flags = ovl_param(ctrl, 3);              // bit0 = clear NoC-id translation (use physical coords)
    if (len == 0) len = 32u;
    if (coord == 0) coord = 0xC8u;                    // default: L2CPU tile 0 = NOC0 (8,3)
    if (src == 0)   src   = 0x30002000u;              // default: x280 TELE window

    if (flags & 1u) R(NIU_CFG_0) &= ~(1u << 14);      // physical coords (translation off) — needed if metal left it on

    volatile uint32_t* scratch = ovl_scratch(ctrl);   // region+0x8000 (BRISC 0x108000), 16B aligned
    uint32_t dest = (uint32_t)(uintptr_t)scratch;
    for (uint32_t i = 0; i < 8u; i++) scratch[i] = 0xEEEEEEEEu;  // poison: distinguishes "read landed" from "no-op"

    uint32_t my_c = R(NOC_NODE_ID) & 0xFFFu;          // my (y<<6)|x for the return address
    uint32_t c0 = ovl_cycles();
    while (R(NOC_CMD_CTRL) != 0u) { }                 // wait cmd buf ready
    uint32_t before = R(NOC_RD_RESP_CNT);             // completion snapshot
    R(NOC_CTRL_REG)   = NOC_CTRL_READ;
    R(NOC_RET_LO)     = dest;
    R(NOC_RET_MID)    = 0u;
    R(NOC_RET_COORD)  = my_c;
    R(NOC_TARG_LO)    = src;
    R(NOC_TARG_MID)   = 0u;
    R(NOC_TARG_COORD) = coord;
    R(NOC_AT_LEN_BE)  = len;
    R(NOC_CMD_CTRL)   = NOC_SEND_REQ;                 // fire the read
    uint32_t spins = 0u;
    while ((R(NOC_RD_RESP_CNT) - before) == 0u) {     // bounded wait for the response
        if (++spins > 5000000u) break;                // don't hang the core if the read never returns
    }
    uint32_t c1 = ovl_cycles();

    volatile uint32_t* t = ovl_telem(ctrl);
    for (uint32_t i = 0; i < 8u; i++) t[4 + i] = scratch[i];  // payload -> telem[4..11]
    t[12] = my_c;                                     // debug: my coord
    t[13] = coord;                                    // debug: target coord used
    t[14] = spins;                                    // debug: 0 = instant, huge = timed out
    t[15] = R(NOC_RD_RESP_CNT) - before;              // debug: responses seen (1 = success)
    ovl_publish(ctrl, len, c1 - c0, scratch[0]);      // telem[0..3]: work / cycles / first-word / 0xC0FFEE
    return scratch[0];
}
