// bh.rs — the friendly hardware harness for L2CPU (x280) kernels, in Rust.
//
// Mirrors include/bh.h. Because we compile a single .rs file (no Cargo), you pull this
// in by *textually including* it — the loader sets the BH_RT env var to its directory:
//
//     #![no_std]
//     #![no_main]
//     include!(concat!(env!("BH_RT"), "/bh.rs"));   // brings in everything below
//
//     #[no_mangle]
//     extern "C" fn kmain() -> ! {                   // you only write this
//         let id = bh_hartid();
//         loop { unsafe { bh_tele(0, id); } }
//     }
//
// This file owns `_start` (sets the stack, calls kmain) and the #[panic_handler], so a
// kernel is just `kmain`. Build + run:  bhtop-l2cpu load <tile> <hart> your_kernel.rs
//
// (For a from-scratch example that does its OWN _start/panic instead, see blink.rs.)

use core::ptr::{read_volatile, write_volatile};

// ---- DRAM windows (see regmap.REGIONS) ----
pub const BH_TRAMP_BASE: usize = 0x3000_0000;
pub const BH_CODE_BASE: usize = 0x3000_1000;
pub const BH_TELE_BASE: usize = 0x3000_2000; // hart 0's window
pub const BH_TELE_STRIDE: usize = 0x100; // per-hart window stride
pub const BH_TELE_SLOTS: usize = 64;

// ---- generic memory-mapped access (just poke an address) ----
#[inline]
pub unsafe fn bh_rd32(addr: usize) -> u32 { read_volatile(addr as *const u32) }
#[inline]
pub unsafe fn bh_wr32(addr: usize, v: u32) { write_volatile(addr as *mut u32, v) }
#[inline]
pub unsafe fn bh_rd64(addr: usize) -> u64 { read_volatile(addr as *const u64) }
#[inline]
pub unsafe fn bh_wr64(addr: usize, v: u64) { write_volatile(addr as *mut u64, v) }

// ---- telemetry: bh_tele(slot, value) -> host reads it with `tele`. PER-HART: writes
// land in THIS hart's window (BH_TELE_BASE + hartid*stride), so harts never collide. ----
#[inline]
pub fn bh_tele_base() -> usize { BH_TELE_BASE + (bh_hartid() as usize) * BH_TELE_STRIDE }
#[inline]
pub unsafe fn bh_tele(slot: usize, v: u32) {
    write_volatile((bh_tele_base() as *mut u32).add(slot), v);
}

// ---- CSRs: per-hart control registers, readable only from hart code ----
#[inline]
pub fn bh_hartid() -> u32 {
    let v: usize;
    unsafe { core::arch::asm!("csrr {0}, mhartid", out(reg) v) };
    v as u32
}
#[inline]
pub fn bh_cycles() -> u64 {
    let v: u64;
    unsafe { core::arch::asm!("csrr {0}, mcycle", out(reg) v) };
    v
}
#[inline]
pub fn bh_instret() -> u64 {
    let v: u64;
    unsafe { core::arch::asm!("csrr {0}, minstret", out(reg) v) };
    v
}

// perf convention: surface cycle + retired counters to reserved slots 62/63 so the
// cockpit's Plot tab can chart throughput per hart (plot slot 63 in rate mode).
pub const BH_SLOT_CYCLES: usize = 62;
pub const BH_SLOT_RETIRED: usize = 63;
#[inline]
pub unsafe fn bh_perf() {
    bh_tele(BH_SLOT_CYCLES, bh_cycles() as u32);
    bh_tele(BH_SLOT_RETIRED, bh_instret() as u32);
}

// ---- arch-state dump: snapshot all 32 GPRs + key CSRs to this hart's DRAM block
// (0x30003000 + hartid*0x200) so the host's "Arch" tab can show the register file. ----
pub const BH_ARCH_BASE: usize = 0x3000_3000;
pub const BH_ARCH_STRIDE: usize = 0x200;
extern "C" { pub fn bh_dump_state(); }
core::arch::global_asm!(r#"
.globl bh_dump_state
bh_dump_state:
  csrw mscratch, t0
  csrw 0x350, t1
  li t0, 0x30003000
  csrr t1, mhartid
  slli t1, t1, 9
  add t0, t0, t1
  sd x0,0(t0)
  sd x1,8(t0)
  sd x2,16(t0)
  sd x3,24(t0)
  sd x4,32(t0)
  csrr t1, mscratch
  sd t1,40(t0)
  csrr t1,0x350
  sd t1,48(t0)
  sd x7,56(t0)
  sd x8,64(t0)
  sd x9,72(t0)
  sd x10,80(t0)
  sd x11,88(t0)
  sd x12,96(t0)
  sd x13,104(t0)
  sd x14,112(t0)
  sd x15,120(t0)
  sd x16,128(t0)
  sd x17,136(t0)
  sd x18,144(t0)
  sd x19,152(t0)
  sd x20,160(t0)
  sd x21,168(t0)
  sd x22,176(t0)
  sd x23,184(t0)
  sd x24,192(t0)
  sd x25,200(t0)
  sd x26,208(t0)
  sd x27,216(t0)
  sd x28,224(t0)
  sd x29,232(t0)
  sd x30,240(t0)
  sd x31,248(t0)
  csrr t1,mhartid
  sd t1,256(t0)
  csrr t1,mcycle
  sd t1,264(t0)
  csrr t1,minstret
  sd t1,272(t0)
  csrr t1,mstatus
  sd t1,280(t0)
  csrr t1,mepc
  sd t1,288(t0)
  csrr t1,mcause
  sd t1,296(t0)
  csrr t1,mtval
  sd t1,304(t0)
  csrr t1,0x351
  sd t1,312(t0)
  csrr t1,0x352
  sd t1,320(t0)
  auipc t1,0
  sd t1,328(t0)
  li t1,0x0D0DEAD0
  sd t1,336(t0)
  csrr t1,0x350
  csrr t0,mscratch
  ret
"#);

// ---- entry: set the stack, call kmain, park if it ever returns ----
core::arch::global_asm!(
    ".option norvc",
    ".section .text._start,\"ax\"",
    ".globl _start",
    "_start:",
    "  la sp, __stack_top",
    "  call kmain",
    "1:wfi",
    "  j 1b",
);

#[panic_handler]
fn bh_panic(_: &core::panic::PanicInfo) -> ! {
    loop { unsafe { core::arch::asm!("wfi") } }
}
