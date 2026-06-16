#!/usr/bin/env python3
# ============================================================================
#  l2cpu_xport_diag.py — READ-ONLY transport/coordinate diagnostic for L2CPU.
# ============================================================================
#
#  Why: the bringup attempt used pyluwen for L2CPU NoC access and every read of
#  (8,3) came back 0xFFFFFFFF (no-response), while tt-exalens reads the SAME
#  addresses fine. So pyluwen's NoC coordinate/transport differs from tt-exalens.
#  This script touches ONLY the (8,3) L2CPU tile + the ARC L2CPU_RESET register,
#  READ-ONLY, to pin the correct access before we write anything again.
#
#  Run AFTER `tt-smi -r 0` (the prior attempt likely wedged NoC0; this first
#  confirms recovery: tt-exalens should read sane values, ARC should read 0x0F).
#
#  It checks three things:
#   1. tt-exalens reads of the L2CPU peripheral regs (the KNOWN-GOOD reference),
#      plus the tile's coordinates in every coordinate system tt-exalens knows.
#   2. pyluwen axi_read32(L2CPU_RESET) — confirms the ARC path is healthy.
#   3. pyluwen noc_read32 of the same regs via (a) low x280-passthrough at (8,3)
#      and (b) the high alias that bypasses the x280 cores — to see which (if any)
#      matches the tt-exalens reference. Whichever matches is the correct pyluwen
#      NoC access; if none match, we stay on tt-exalens for NoC.
#
#  No writes. Safe to run. If it still reads 0xFFFFFFFF everywhere, the device is
#  still wedged — re-run `tt-smi -r 0`.
# ============================================================================

import sys

XY = (8, 3)                              # L2CPU tile, noc0
RESET_VEC   = 0x2001_0000
HART_STATUS = 0x2001_0400
SCRATCH     = 0x2001_0100
L2CPU_RESET = 0x80030014                 # ARC (via axi)
# high alias that bypasses the x280 cores: x280-phys + 0xFFFF_F7FE_DFF0_0000
HIGH = 0xFFFF_F7FE_DFF0_0000
COORD_SYSTEMS = ["noc0", "noc1", "translated", "virtual", "logical", "die"]


def main():
    print("L2CPU transport diagnostic — READ-ONLY, tile (8,3) + ARC L2CPU_RESET only.\n")

    # ---- 1. tt-exalens reference ----
    print("[1] tt-exalens (known-good reference)")
    ttx = {}
    try:
        from ttexalens import init_ttexalens
        from ttexalens.tt_exalens_lib import read_words_from_device
        ctx = init_ttexalens()
        dev = ctx.devices[0]
        loc = next((l for l in dev.get_block_locations("l2cpu") if tuple(l.to("noc0")) == XY), None)
        if loc is None:
            print("  no L2CPU tile at (8,3) — aborting"); return 1
        coords = {}
        for sysname in COORD_SYSTEMS:
            try:
                coords[sysname] = tuple(loc.to(sysname))
            except Exception:
                pass
        print(f"  tile coords: {coords}")
        for name, addr in (("reset_vec", RESET_VEC), ("status", HART_STATUS), ("scratch", SCRATCH)):
            try:
                v = read_words_from_device(loc, addr, word_count=1, context=ctx, noc_id=0, safe_mode=False)[0]
                ttx[addr] = v
                print(f"  ttx  0x{addr:08X} {name:9s} = 0x{v:08X}")
            except Exception as e:
                print(f"  ttx  0x{addr:08X} {name:9s} = ERROR {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  tt-exalens unavailable: {e}")

    # ---- 2 + 3. pyluwen ----
    print("\n[2] pyluwen ARC (axi)")
    chip = None
    try:
        from pyluwen import PciChip
        chip = PciChip(pci_interface=0)
        try:
            v = chip.axi_read32(L2CPU_RESET)
            print(f"  axi  0x{L2CPU_RESET:08X} L2CPU_RESET = 0x{v:08X}  "
                  f"({'healthy (expect 0x0F in reset)' if v != 0xFFFFFFFF else 'ALL-ONES = still wedged? re-run tt-smi -r 0'})")
        except Exception as e:
            print(f"  axi read ERROR {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  pyluwen unavailable: {e}")

    if chip is not None:
        print("\n[3] pyluwen NoC — low passthrough vs high alias at (8,3)")
        for name, addr in (("reset_vec", RESET_VEC), ("status", HART_STATUS)):
            ref = ttx.get(addr)
            for label, a in ((f"low  0x{addr:010X}", addr), (f"high 0x{addr + HIGH:012X}", addr + HIGH)):
                try:
                    v = chip.noc_read32(0, XY[0], XY[1], a)
                    match = "" if ref is None else ("  <== MATCHES tt-exalens" if v == ref else "  (differs)")
                    print(f"  noc {label} {name:9s} = 0x{v:08X}{match}")
                except Exception as e:
                    print(f"  noc {label} {name:9s} = ERROR {type(e).__name__}: {e}")

    print("\nVERDICT: use whichever NoC method MATCHES the tt-exalens reference for L2CPU writes.")
    print("If none match, keep L2CPU NoC on tt-exalens and use pyluwen only for ARC axi (PLL/reset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
