/* bhtop.l2cpu telemetry — the dead-simple way to surface values from x280 hart
 * code to the host. Write 32-bit slots to a fixed DRAM block; the host reads them
 * with `bhtop-l2cpu tele <tile> [hart]`.
 *
 *   #include <tele.h>
 *   TELE[0] = ++heartbeat;   // convention: slot 0 = liveness counter
 *   TELE[1] = my_metric;     // slots 1..63 = whatever you want
 *
 * PER-HART: each hart gets its OWN 64-slot window (hart N at TELE_BASE + N*0x100),
 * and TELE automatically targets the window of whichever hart is running. So if you
 * load different kernels on harts 0 and 1, their telemetry never collides — the host
 * reads each hart's slots separately. No CPU caches are involved (uncached DRAM), so
 * the host sees writes immediately. */
#pragma once
#define TELE_BASE 0x30002000u    /* hart 0's window */
#define TELE_STRIDE 0x100u       /* per-hart stride: hart N window = TELE_BASE + N*0x100 */
#define TELE_SLOTS 64u

/* the running hart's id (0..3) — picks its telemetry window */
static inline unsigned int bh__hartid(void) {
    unsigned int id; __asm__ volatile ("csrr %0, mhartid" : "=r"(id)); return id;
}
#define TELE ((volatile unsigned int *)(TELE_BASE + bh__hartid() * TELE_STRIDE))
static inline void tele(unsigned i, unsigned v) { TELE[i] = v; }
