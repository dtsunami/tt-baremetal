/* dram_sentinel.c — GAP-5 de-risk: prove the x280 can directly read/write a large span of off-chip GDDR
 * as DISTINCT DRAM (no aliasing, no unmapped holes), the foundation for GDDR-resident params (millions of
 * Gaussians). Writes a unique value at PROBE_N points spanning [PROBE_BASE, PROBE_TOP), then reads them ALL
 * back and verifies — aliasing (two offsets -> same DRAM) or an unmapped offset shows up as a mismatch.
 *
 *   PROBE_BASE default 0x30010000 (just above user code at 0x30008000)
 *   PROBE_TOP  default 0x40000000 (end of the documented 256 MB uncached-GDDR window)
 *   Override via compile defines to probe beyond the window (the other DRAM banks) — ONLY after the NoC
 *   topology says those addresses are DRAM, else a stray write could hit another tile.
 *
 *   TELE[0]='SENT' TELE[1]=n TELE[2]=bad_count TELE[3]=first_bad_addr>>4 TELE[4]=stride TELE[5]=readback@base */
#include <tele.h>
#include <stdint.h>
#ifndef PROBE_BASE
#define PROBE_BASE 0x30010000ULL
#endif
#ifndef PROBE_TOP
#define PROBE_TOP  0x40000000ULL
#endif
#ifndef PROBE_N
#define PROBE_N 240u
#endif
#define KEY(i) (0xC0DE0000u ^ ((uint32_t)(i) * 0x9E3779B1u))

int main(void){
    uint64_t base=(uint64_t)PROBE_BASE, top=(uint64_t)PROBE_TOP;
    uint32_t n=(uint32_t)PROBE_N;
    uint64_t stride=(top-base)/n; stride &= ~0xFULL; if(stride==0) stride=0x10;
    /* write pass: unique value per point */
    for(uint32_t i=0;i<n;i++){
        uint64_t a=base+(uint64_t)i*stride;
        *(volatile uint32_t*)a = KEY(i);
    }
    /* verify pass: every point must still hold its own value (else alias / no DRAM) */
    uint32_t bad=0; uint64_t first_bad=0;
    for(uint32_t i=0;i<n;i++){
        uint64_t a=base+(uint64_t)i*stride;
        uint32_t got=*(volatile uint32_t*)a;
        if(got != KEY(i)){ if(bad==0) first_bad=a; bad++; }
    }
    TELE[0]=0x53454E54u;                 /* 'SENT' */
    TELE[1]=n;
    TELE[2]=bad;
    TELE[3]=(uint32_t)(first_bad>>4);
    TELE[4]=(uint32_t)stride;
    TELE[5]=*(volatile uint32_t*)base;   /* readback at base (host cross-checks == KEY(0)) */
    for(;;){}
}
