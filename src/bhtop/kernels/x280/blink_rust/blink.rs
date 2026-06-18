// blink.rs — minimal x280 bare-metal in Rust (no_std). Requires the rustup target
// (see README "Rust toolchain"):  rustup target add riscv64gc-unknown-none-elf
//   bhtop-l2cpu load 0 0 <path>/blink.rs
//   bhtop-l2cpu tele 0
#![no_std]
#![no_main]

use core::panic::PanicInfo;
use core::ptr::write_volatile;

const TELE: *mut u32 = 0x3000_2000 as *mut u32; // telemetry block

// crt0 (rt/crt0.s) is C-only; for Rust we provide our own _start in .text._start.
core::arch::global_asm!(
    ".option norvc",
    ".section .text._start,\"ax\"",
    ".globl _start",
    "_start:",
    "  la sp, __stack_top",
    "  call rust_main",
    "1:wfi",
    "  j 1b",
);

#[no_mangle]
extern "C" fn rust_main() -> ! {
    let mut hb: u32 = 0;
    loop {
        hb = hb.wrapping_add(1);
        unsafe {
            write_volatile(TELE.add(0), hb); // slot 0 = heartbeat
            write_volatile(TELE.add(1), hb.wrapping_mul(3)); // a metric
        }
    }
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("wfi") }
    }
}
