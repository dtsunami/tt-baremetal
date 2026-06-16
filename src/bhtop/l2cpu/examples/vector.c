/* vector.c — the X280's RVV (512-bit RISC-V Vector) unit, on bare metal.
 *
 * Proof the vector engine runs: enable the vector context, read VLEN, and do a vector
 * reduction (sum 0..15) — all on silicon. This is the staging/projection workhorse for the
 * stage->crank pipeline: the x280 vectorizes the irregular projection (CRT residues,
 * Gaussian-splat deposits, layout/quantize) and hands dense tiles to the Tensix cores.
 *
 *   slot 0 = heartbeat   slot 1 = VLEN (bits, expect 512)   slot 2 = vredsum result (120)
 *
 * NOTE: vector ops trap as illegal until bh_vec_enable(). Write them with `.option arch,+v`
 * (no toolchain flag needed). For C-level RVV *intrinsics* you'd compile -march=rv64gcv. */
#include <bh.h>

int main(void) {
    unsigned hb = 0, vsum = 0;
    bh_vec_enable();                 /* mstatus.VS = on — required before vector ops */
    TELE[1] = bh_vlenb() * 8;        /* VLEN in bits (vlenb = bytes per vector reg) */
    __asm__ volatile(
        ".option arch, +v\n"
        "li      a0, 16\n"
        "vsetvli t0, a0, e32, m1, ta, ma\n"  /* 16 x 32-bit elements per vector */
        "vid.v   v0\n"                        /* v0 = [0,1,2,...,15] */
        "vmv.v.i v8, 0\n"
        "vredsum.vs v9, v0, v8\n"             /* v9[0] = sum(v0) = 120 */
        "vmv.x.s %0, v9\n"
        : "=r"(vsum) : : "a0", "t0");
    TELE[2] = vsum;                  /* 120 = 0x78 */
    for (;;) { hb++; TELE[0] = hb; }
}
