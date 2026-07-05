/* cb_scatter.c — x280 BACKWARD irregular tier: fp32 gradient SCATTER-ADD (the mirror of the forward
 * gather). For each tile, per-Gaussian gradients arrive in that tile's SORTED slot order; the x280
 * scatters each back to its original Gaussian id and ACCUMULATES across tiles into a per-Gaussian
 * gradient buffer. Enables the x280 scalar FPU (mstatus.FS) — the capability we dodged with the u32
 * sort, and the one on-x280 projection will also need.
 *   Buffers live in the OPEN uncached window (0x30005000+), clear of the tele-window boundaries that
 *   glitch contiguous multi-word host writes.
 *   ORDER 0x30005000 [T, K, T*K order ints] · GRAD 0x30005800 [T*K fp32] · OUT 0x30006000 [K fp32] */
#include <tele.h>
#include <stdint.h>

int main(void) {
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));   /* FS=Dirty -> scalar FPU on */
    volatile uint32_t *O = (volatile uint32_t *)0x30005000u;
    int T = (int)O[0], K = (int)O[1];
    if (K > 16) K = 16;
    if (T > 8) T = 8;
    volatile int   *order = (volatile int   *)(O + 2);       /* T*K sorted->original ids */
    volatile float *grad  = (volatile float *)0x30005800u;   /* T*K fp32 gradients */
    volatile float *out   = (volatile float *)0x30006000u;   /* K fp32 accumulator */
    for (int g = 0; g < K; g++) out[g] = 0.0f;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < K; i++) {
            int g = order[t * K + i];
            if (g >= 0 && g < K) out[g] += grad[t * K + i];  /* fp32 scatter-add */
        }
    TELE[0] = 0x53434154u;                                   /* 'SCAT' done */
    for (;;) { }
    return 0;
}
