// hwinfo.rs — same as hwinfo.c, in Rust, using the harness. You write only kmain();
// bh.rs provides _start + the panic handler.
//
//   bhtop-l2cpu load 0 0 src/bhtop/l2cpu/examples/hwinfo.rs
//   bhtop-l2cpu tele 0
#![no_std]
#![no_main]
include!(concat!(env!("BH_RT"), "/bh.rs")); // pull in the harness (sets BH_RT for us)

#[no_mangle]
extern "C" fn kmain() -> ! {
    let mut hb: u32 = 0;
    unsafe {
        bh_tele(1, bh_hartid()); // which hart am I (0..3)
        bh_tele(5, 0x1F0C0DE5); // marker: this kernel is running
    }
    loop {
        hb = hb.wrapping_add(1);
        unsafe {
            bh_tele(0, hb); // slot 0 = heartbeat (liveness)
            bh_tele(2, bh_cycles() as u32); // cycle counter low32
            bh_tele(3, bh_instret() as u32); // instructions retired low32
        }
    }
}
