// SPDX-License-Identifier: Apache-2.0
// baremetal.h — bhtop BARE-METAL Tensix kernel ABI (NO tt-metal). A kernel = crt0.s + a bm_main().
// crt0 sets sp, loads up to 4 u32 params from the L1 arg block (BM_ARGS), calls bm_main(a0..a3),
// then parks. The host (tensix/baremetal.py) writes the kernel to L1 0x0, pokes params, and deasserts
// the RISC reset over exalens — the tt-metal-free sibling of overlays (metal-park) and llk (metal-run).
// Also carries the extracted-from-tt-metal Blackhole NOC0 read primitive as a reusable inline.
#pragma once
#include <stdint.h>
#define BM_ARGS    0x00001000u   // host pokes up to 4 u32 params here, then deasserts reset
#define BM_RESULT  0x00002000u   // kernels publish their payload here (host reads over exalens)
#define BM_DBG     0x00002100u   // per-kernel debug scratch
#define R(a)       (*(volatile uint32_t*)(uintptr_t)(a))
// ---- NOC0 BRISC read command buffer (index 1): 0xFFB20000 + (1<<11) = 0xFFB20800 ----
#define NOC_TARG_LO 0xFFB20800u
#define NOC_TARG_MID 0xFFB20804u
#define NOC_TARG_COORD 0xFFB20808u
#define NOC_RET_LO 0xFFB2080Cu
#define NOC_RET_MID 0xFFB20810u
#define NOC_RET_COORD 0xFFB20814u
#define NOC_CTRL_REG 0xFFB2081Cu
#define NOC_AT_LEN_BE 0xFFB20820u
#define NOC_CMD_CTRL 0xFFB20840u
#define NOC_NODE_ID 0xFFB20844u
#define NOC_RD_RESP_CNT 0xFFB20208u
#define NOC_WR_ACK_CNT 0xFFB20204u  // NIU_MST_WR_ACK_RECEIVED (NOC_STATUS cnt 0x1)
#define NIU_CFG_0 0xFFB20100u
#define NOC_CTRL_READ 0x00002090u   // CPY|RD|RESP_MARKED|VC_STATIC|STATIC_VC(1)
#define NOC_CTRL_WRITE 0x00002092u  // CPY|WR|RESP_MARKED|VC_STATIC|STATIC_VC(1)
static inline uint32_t bm_coord(uint32_t x, uint32_t y){ return ((y&0x3Fu)<<6)|(x&0x3Fu); }
// Blocking NOC0 read: len bytes from (coord):src into local L1 dst. Verified on silicon.
static inline uint32_t bm_noc0_read(uint32_t coord, uint32_t src, uint32_t dst, uint32_t len){
    R(NIU_CFG_0) &= ~(1u<<14);                        // physical coords (no NoC-id translation)
    uint32_t my = R(NOC_NODE_ID) & 0xFFFu;
    while (R(NOC_CMD_CTRL) != 0u) {}
    uint32_t before = R(NOC_RD_RESP_CNT);
    R(NOC_CTRL_REG)=NOC_CTRL_READ; R(NOC_RET_LO)=dst; R(NOC_RET_MID)=0u; R(NOC_RET_COORD)=my;
    R(NOC_TARG_LO)=src; R(NOC_TARG_MID)=0u; R(NOC_TARG_COORD)=coord; R(NOC_AT_LEN_BE)=len;
    R(NOC_CMD_CTRL)=1u;
    uint32_t s=0u; while((R(NOC_RD_RESP_CNT)-before)==0u){ if(++s>5000000u) break; }
    return R(NOC_RD_RESP_CNT)-before;                 // 1 = one clean response
}
// Blocking NOC0 write: len bytes from local L1 src to (coord):dst. Mirror of the read (TARG=local
// source, RET=remote dest); polls the write-ack counter. Verified on silicon. dst is a 32-bit local
// GDDR/L1 addr (MID=0). Enables the het CB consumer to ack the producer.
static inline uint32_t bm_noc0_write(uint32_t coord, uint32_t dst, uint32_t src, uint32_t len){
    R(NIU_CFG_0) &= ~(1u<<14);
    while (R(NOC_CMD_CTRL) != 0u) {}
    uint32_t before = R(NOC_WR_ACK_CNT);
    R(NOC_CTRL_REG)=NOC_CTRL_WRITE;
    R(NOC_TARG_LO)=src; R(NOC_TARG_MID)=0u;              // local source
    R(NOC_RET_LO)=dst; R(NOC_RET_MID)=0u; R(NOC_RET_COORD)=coord;   // remote dest
    R(NOC_AT_LEN_BE)=len;
    R(NOC_CMD_CTRL)=1u;
    (void)before;
    uint32_t s=0u; while(R(NOC_CMD_CTRL)!=0u){ if(++s>5000000u) break; }  // NIU accepted the cmd
    return 1u;                                           // write issued (data lands in-flight, NoC-ordered)
}
