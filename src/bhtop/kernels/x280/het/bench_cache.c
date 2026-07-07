/* bench_cache.c — GAP-5 throughput de-risk. The 1M-Gaussian Adam step is bound by UNCACHED GDDR: every
 * param/Adam word is a NoC round-trip. The ISA map exposes a CACHED alias of the SAME local GDDR at
 * 0x4000_3000_0000 (MemoryMap.md:34). This times a read-modify-write sweep over an NW-word buffer, NITER
 * times, through the UNCACHED window (0x30000000+X) vs the CACHED alias (0x400030000000+X) — the cached
 * path should collapse the per-word cost if the D-cache batches GDDR traffic, which is the lever for fast
 * training of millions. Each buffer is touched through exactly ONE alias (no cross-alias incoherence).
 *   TELE[1]=NW TELE[2]=NITER TELE[3]=unc_cyc_lo TELE[4]=unc_cyc_hi TELE[5]=cac_cyc_lo TELE[6]=cac_cyc_hi
 *   TELE[7]=checksum_unc TELE[8]=checksum_cac (must match -> same data, just different alias) */
#include <tele.h>
#include <stdint.h>
#define UNC_BASE 0x30100000ULL          /* GDDR offset 0x100000 (1 MiB in, above code), uncached */
#define CAC_BASE 0x400030100000ULL      /* SAME GDDR line, cached alias (0x4000_3000_0000 + 0x100000) */
#ifndef NW
#define NW 65536u                        /* 64K words = 256 KiB working set */
#endif
#ifndef NITER
#define NITER 16u
#endif
static inline uint64_t rdcycle(void){ uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }

static uint32_t sweep(volatile uint32_t *buf, uint32_t nw, uint32_t niter){
    for(uint32_t i=0;i<nw;i++) buf[i]=i*2654435761u;     /* seed */
    for(uint32_t it=0; it<niter; it++)
        for(uint32_t i=0;i<nw;i++) buf[i]=buf[i]*3u+1u;  /* read-modify-write */
    uint32_t s=0; for(uint32_t i=0;i<nw;i++) s+=buf[i];  /* checksum forces reads */
    return s;
}

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    uint32_t nw=NW, niter=NITER;
    uint64_t t0=rdcycle();
    uint32_t su=sweep((volatile uint32_t*)UNC_BASE, nw, niter);
    uint64_t t1=rdcycle();
    uint32_t sc=sweep((volatile uint32_t*)CAC_BASE, nw, niter);
    uint64_t t2=rdcycle();
    TELE[1]=nw; TELE[2]=niter;
    TELE[3]=(uint32_t)(t1-t0); TELE[4]=(uint32_t)((t1-t0)>>32);
    TELE[5]=(uint32_t)(t2-t1); TELE[6]=(uint32_t)((t2-t1)>>32);
    TELE[7]=su; TELE[8]=sc;
    TELE[0]=0x42454E43u;   /* 'BENC' */
    for(;;){}
}
