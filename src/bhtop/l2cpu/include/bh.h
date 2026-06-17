/* bh.h — the friendly hardware harness for L2CPU (SiFive x280) kernels.
 *
 * Include this and the whole tile is named, not magic numbers:
 *
 *     #include <bh.h>
 *     int main(void) {
 *         TELE[0] = bh_hartid();          // which of the 4 harts am I?
 *         for (;;) TELE[1] = bh_cycles(); // free-running cycle counter
 *     }
 *
 * Build + run:  bhtop-l2cpu load <tile> <hart> your_kernel.c
 * Watch it:     bhtop-l2cpu tele <tile>
 *
 * The addresses below mirror regmap.py (the canonical map). For the full story of
 * what each one is, see HARDWARE.md or run `bhtop-l2cpu map`.
 *
 * IMPORTANT: this DRAM window is uncached, so writes are visible to the host with no
 * flush. But your *code* runs through the I-cache — the loader's trampoline does the
 * `fence.i` for you on every redirect, so you never have to think about it. */
#pragma once
#include <tele.h>     /* TELE[] telemetry slots — the easy way to surface values */

typedef unsigned int      u32;
typedef unsigned long     u64;   /* lp64d: 'long' is 64-bit on rv64 */

/* ---- DRAM windows we use (canonical map = regmap.py; the lab toolchain injects these as
 * -D on every build, so they never drift. The #ifndef fallbacks are only for a standalone
 * compile/linter; the injected value always wins. BH_CODE_BASE is injected per-compile =
 * the real load address.) --------------------------------------------------------------- */
#ifndef BH_TRAMP_BASE
#define BH_TRAMP_BASE 0x30000000u   /* RNMI redirect trampoline (installed by bringup) */
#endif
#ifndef BH_CODE_BASE
#define BH_CODE_BASE  0x30008000u   /* default load addr (above the data blocks)       */
#endif
#ifndef BH_TELE_BASE
#define BH_TELE_BASE  TELE_BASE     /* 0x30002000 telemetry block (from tele.h)        */
#endif

/* ---- L2CPU peripheral registers (x280 phys; per-hart = base + N*stride) ------------ *
 * These are the knobs the host loader uses to park/seize harts. A kernel rarely needs
 * them, but they're here so you can read your own state. Per-hart accessors take N.   */
#define BH_RESET_VEC(n)   ((volatile u64 *)(0x20010000u + (n) * 8u))   /* hart N init PC   */
#define BH_HART_STATUS    ((volatile u32 *)0x20010400u)               /* all-harts status */
#define BH_TRIGGER        ((volatile u32 *)0x20010414u)               /* RNMI trigger bits*/
#define BH_RNMI_TRAP(n)   ((volatile u64 *)(0x20010418u + (n) * 16u)) /* hart N RNMI vec  */
#define BH_RNMI_EXC(n)    ((volatile u64 *)(0x20010420u + (n) * 16u)) /* hart N exc vec   */

/* ---- generic memory-mapped access (the bhtop way: just poke an address) ------------ */
static inline u32 bh_rd32(u64 addr)          { return *(volatile u32 *)addr; }
static inline void bh_wr32(u64 addr, u32 v)  { *(volatile u32 *)addr = v; }
static inline u64 bh_rd64(u64 addr)          { return *(volatile u64 *)addr; }
static inline void bh_wr64(u64 addr, u64 v)  { *(volatile u64 *)addr = v; }

/* ---- CSRs: per-hart control registers, readable only from hart code ---------------- */
#define BH_CSR_READ(csr) ({ u64 __v; __asm__ volatile ("csrr %0, " #csr : "=r"(__v)); __v; })
#define BH_CSR_WRITE(csr, v) __asm__ volatile ("csrw " #csr ", %0" :: "r"((u64)(v)))

static inline u32 bh_hartid(void) { return (u32)BH_CSR_READ(mhartid); } /* 0..3: who am I */
static inline u64 bh_cycles(void) { return BH_CSR_READ(mcycle); }       /* cheap timer    */
static inline u64 bh_instret(void){ return BH_CSR_READ(minstret); }     /* insns retired  */

/* perf convention: surface this hart's cycle + instructions-retired counters into the
 * reserved high telemetry slots, so the cockpit's Plot tab can chart throughput over
 * time (per hart). Call bh_perf() in your loop; plot slot 63 in rate mode = retired/sec. */
#define BH_SLOT_CYCLES  62u
#define BH_SLOT_RETIRED 63u
static inline void bh_perf(void) {
    TELE[BH_SLOT_CYCLES] = (u32)bh_cycles();
    TELE[BH_SLOT_RETIRED] = (u32)bh_instret();
}

/* ---- RVV (the X280's 512-bit vector unit) ---------------------------------------------
 * The x280 has RISC-V Vector v1.0 (misa bit 21 = V; VLEN 512). Call bh_vec_enable() ONCE
 * before any vector instruction to turn on the vector context (mstatus.VS = Dirty); vector
 * ops trap as illegal until you do. Write vector asm with `.option arch, +v` (works under
 * the default -march=rv64gc). bh_vlenb() returns VLEN/8 (bytes per vector register = 64).
 * This is the staging/projection workhorse — vectorize CRT residues, Gaussian splats, etc. */
static inline void bh_vec_enable(void) { __asm__ volatile("csrs mstatus, %0" :: "r"(0x600u)); }
static inline u32 bh_vlenb(void) { u32 v; __asm__ volatile("csrr %0, 0xC22" : "=r"(v)); return v; }

/* ---- command mailbox: a DRAM doorbell the host rings to update a register live --------
 * The host can't write your CSRs/GPRs/vregs over the NoC — only you can, from here. So a
 * live "register update" is cooperative: the host writes a value into this hart's mailbox
 * and bumps the seq word; you poll bh_cmd() and apply it. The window is uncached, so the
 * host's write is visible to your next load with no flush. Layout (u32):
 *     cmd[0] = seq (doorbell — changes when a new command arrives)
 *     cmd[1] = op    cmd[2] = arg0    cmd[3] = arg1
 * Host side: L2cpu.command(tile,hart,op,arg0) or `bhtop-l2cpu cmd <t> <hart> <op> <arg0>`.
 * See examples/mailbox.c for the poll loop. This is the non-preemptive sibling of the RNMI
 * redirect (which the host uses to swap your CODE); here YOU stay in control and just read. */
#ifndef BH_CMD_BASE
#define BH_CMD_BASE   0x20010100u   /* uncached peripheral scratch (host writes are visible to */
#endif
#ifndef BH_CMD_STRIDE
#define BH_CMD_STRIDE 0x10u         /* hart reads here; a cached DRAM mailbox reads stale)     */
#endif
static inline volatile u32 *bh_cmd(void) {            /* this hart's mailbox window */
    return (volatile u32 *)(BH_CMD_BASE + bh_hartid() * BH_CMD_STRIDE);
}
/* Read the doorbell seq. The mailbox lives in UNCACHED peripheral scratch (BH_CMD_BASE), so the
 * host's NoC write is visible to this load with no flush — a plain `cmd[0]` poll always sees the
 * latest command. (A cached GDDR mailbox would read STALE: the hart's own writes reach GDDR so
 * telemetry works, but host writes don't snoop the hart's D-cache. cbo.inval would refetch, but
 * it TRAPS as illegal on this x280 — so we handle coherence by region choice, not a cache op.)
 * cmd[1..3] read after sit in the same uncached window. */
static inline u32 bh_cmd_seq(volatile u32 *cmd) {
    return cmd[0];   /* uncached peripheral scratch — no cache op needed (cbo.inval traps here) */
}

/* ---- tiny helpers ------------------------------------------------------------------- */
static inline void bh_spin(u64 cycles) {                 /* crude busy-wait (no clock dep)*/
    u64 end = bh_cycles() + cycles;
    while (bh_cycles() < end) { __asm__ volatile (""); }
}
static inline void bh_park(void) { for (;;) __asm__ volatile ("wfi"); } /* stop cleanly  */

/* ---- architectural-state dump --------------------------------------------------------
 * Call bh_dump_state() to snapshot ALL 32 GPRs + key CSRs into this hart's DRAM block
 * (0x30003000 + hartid*0x200). The host can't read a hart's registers directly, so this
 * is how the cockpit's "Arch" tab shows the whole register file. x5/x6 are saved via CSRs
 * so they read true; ra/sp reflect the call site (calling sets them). */
#define BH_ARCH_BASE   0x30003000u
#define BH_ARCH_STRIDE 0x200u
extern void bh_dump_state(void);
__asm__(
  ".globl bh_dump_state\n.type bh_dump_state,@function\n"
  "bh_dump_state:\n"
  "  csrw mscratch, t0\n  csrw 0x350, t1\n"              /* free t0,t1 (mscratch,mnscratch) */
  "  li t0, 0x30003000\n  csrr t1, mhartid\n  slli t1,t1,9\n  add t0,t0,t1\n" /* base+hart*0x200 */
  "  sd x0,0(t0)\n  sd x1,8(t0)\n  sd x2,16(t0)\n  sd x3,24(t0)\n  sd x4,32(t0)\n"
  "  csrr t1, mscratch\n  sd t1,40(t0)\n  csrr t1,0x350\n  sd t1,48(t0)\n"     /* x5,x6 originals */
  "  sd x7,56(t0)\n  sd x8,64(t0)\n  sd x9,72(t0)\n  sd x10,80(t0)\n  sd x11,88(t0)\n"
  "  sd x12,96(t0)\n  sd x13,104(t0)\n  sd x14,112(t0)\n  sd x15,120(t0)\n  sd x16,128(t0)\n"
  "  sd x17,136(t0)\n  sd x18,144(t0)\n  sd x19,152(t0)\n  sd x20,160(t0)\n  sd x21,168(t0)\n"
  "  sd x22,176(t0)\n  sd x23,184(t0)\n  sd x24,192(t0)\n  sd x25,200(t0)\n  sd x26,208(t0)\n"
  "  sd x27,216(t0)\n  sd x28,224(t0)\n  sd x29,232(t0)\n  sd x30,240(t0)\n  sd x31,248(t0)\n"
  "  csrr t1,mhartid\n  sd t1,256(t0)\n  csrr t1,mcycle\n  sd t1,264(t0)\n"
  "  csrr t1,minstret\n  sd t1,272(t0)\n  csrr t1,mstatus\n  sd t1,280(t0)\n"
  "  csrr t1,mepc\n  sd t1,288(t0)\n  csrr t1,mcause\n  sd t1,296(t0)\n  csrr t1,mtval\n  sd t1,304(t0)\n"
  "  csrr t1,0x351\n  sd t1,312(t0)\n  csrr t1,0x352\n  sd t1,320(t0)\n"       /* mnepc,mncause */
  "  auipc t1,0\n  sd t1,328(t0)\n  li t1,0x0D0DEAD0\n  sd t1,336(t0)\n"        /* pc, magic */
  "  csrr t1,0x350\n  csrr t0,mscratch\n  ret\n");

/* ---- vector-register dump (bh_dump_state covers GPRs+scalar CSRs but NOT vector state) ----
 * Snapshots all 32 vector registers (v0..v31, 64 B each) + the 7 vector CSRs into this hart's
 * VARCH block (0x30005000 + hartid*0x900), so the host (which can't read vregs over the NoC)
 * can decode them. Saves/restores the kernel's vtype+vl so calling it doesn't disturb your
 * vector config. Uses caller-saved a0..a5 only (it's a function call). Call it in your loop to
 * keep the cockpit's live vector view fresh. Decode host-side with L2cpu.vec_state(tile,hart). */
#ifndef BH_VARCH_BASE
#define BH_VARCH_BASE   0x30005000u
#endif
#ifndef BH_VARCH_STRIDE
#define BH_VARCH_STRIDE 0x900u
#endif
extern void bh_dump_vec(void);
__asm__(
  ".globl bh_dump_vec\n.type bh_dump_vec,@function\n"
  ".option arch, +v\n"
  "bh_dump_vec:\n"
  "  csrr a4, 0xC20\n  csrr a5, 0xC21\n"                 /* save kernel vl, vtype */
  "  li   a0, 0x30005000\n  csrr a1, mhartid\n  li a2, 0x900\n  mul a1,a1,a2\n  add a0,a0,a1\n"
  "  csrr a2, 0xC22\n"                                    /* a2 = vlenb (64) */
  "  vsetvli x0, a2, e8, m1\n  mv a1, a0\n"               /* byte view; a1 = cursor */
  "  vse8.v v0,(a1)\n  add a1,a1,a2\n  vse8.v v1,(a1)\n  add a1,a1,a2\n"
  "  vse8.v v2,(a1)\n  add a1,a1,a2\n  vse8.v v3,(a1)\n  add a1,a1,a2\n"
  "  vse8.v v4,(a1)\n  add a1,a1,a2\n  vse8.v v5,(a1)\n  add a1,a1,a2\n"
  "  vse8.v v6,(a1)\n  add a1,a1,a2\n  vse8.v v7,(a1)\n  add a1,a1,a2\n"
  "  vse8.v v8,(a1)\n  add a1,a1,a2\n  vse8.v v9,(a1)\n  add a1,a1,a2\n"
  "  vse8.v v10,(a1)\n add a1,a1,a2\n  vse8.v v11,(a1)\n add a1,a1,a2\n"
  "  vse8.v v12,(a1)\n add a1,a1,a2\n  vse8.v v13,(a1)\n add a1,a1,a2\n"
  "  vse8.v v14,(a1)\n add a1,a1,a2\n  vse8.v v15,(a1)\n add a1,a1,a2\n"
  "  vse8.v v16,(a1)\n add a1,a1,a2\n  vse8.v v17,(a1)\n add a1,a1,a2\n"
  "  vse8.v v18,(a1)\n add a1,a1,a2\n  vse8.v v19,(a1)\n add a1,a1,a2\n"
  "  vse8.v v20,(a1)\n add a1,a1,a2\n  vse8.v v21,(a1)\n add a1,a1,a2\n"
  "  vse8.v v22,(a1)\n add a1,a1,a2\n  vse8.v v23,(a1)\n add a1,a1,a2\n"
  "  vse8.v v24,(a1)\n add a1,a1,a2\n  vse8.v v25,(a1)\n add a1,a1,a2\n"
  "  vse8.v v26,(a1)\n add a1,a1,a2\n  vse8.v v27,(a1)\n add a1,a1,a2\n"
  "  vse8.v v28,(a1)\n add a1,a1,a2\n  vse8.v v29,(a1)\n add a1,a1,a2\n"
  "  vse8.v v30,(a1)\n add a1,a1,a2\n  vse8.v v31,(a1)\n add a1,a1,a2\n"   /* a1 = base+0x800 */
  "  csrr a3,0x008\n sd a3,0(a1)\n   csrr a3,0x009\n sd a3,8(a1)\n"       /* vstart, vxsat */
  "  csrr a3,0x00A\n sd a3,16(a1)\n  csrr a3,0x00F\n sd a3,24(a1)\n"      /* vxrm, vcsr   */
  "  sd a4,32(a1)\n  sd a5,40(a1)\n"                                       /* vl, vtype (saved) */
  "  csrr a3,0xC22\n sd a3,48(a1)\n  li a3,0x0D0DEC00\n sd a3,56(a1)\n"   /* vlenb, magic */
  "  vsetvl x0, a4, a5\n  ret\n");                                         /* restore vtype/vl */
