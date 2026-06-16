#!/usr/bin/env python3
# ============================================================================
#  l2cpu_dram_probe.py — is the L2CPU tile's local DRAM writable? (for code)
# ============================================================================
#
#  Stage 3 needs to place the redirect trampoline + new hart code in the tile's
#  local DRAM (scratch is only 64B and full). This confirms DRAM is writable on
#  tile (8,3) before we point a running hart at it. The harts' original reset
#  vector was 0xD000_0000 (uncached GDDR window), so that region is addressable;
#  this checks it's also WRITABLE (not harvested).
#
#  Write/restore test (leaves DRAM as found): for each candidate address, read
#  the original word, write a marker, read back, then restore the original.
#  tt-exalens transport (the proven L2CPU NoC path). READ-mostly + restore.
#  If anything wedges: tt-smi -r 0.
# ============================================================================

import sys

XY = (8, 3)
PASS_HI = 0x7FFF_FFFF_FFFF
MARKER = 0xC0DEF00D
# uncached GDDR window (0x3000_0000+) — where Stage 3 will put code — plus the
# original reset-vector address as a known-addressable reference.
CANDIDATES = [0x30000000, 0x30000040, 0x30000080, 0xD0000000]


def main():
    print(f"L2CPU local-DRAM writability probe — tile {XY}, write+restore.\n")
    from ttexalens import init_ttexalens
    from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device
    ctx = init_ttexalens()
    dev = ctx.devices[0]
    loc = next((l for l in dev.get_block_locations("l2cpu") if tuple(l.to("noc0")) == XY), None)
    if loc is None:
        print("no L2CPU tile at (8,3)"); return 1

    def rd(a):
        return read_words_from_device(loc, a, word_count=1, context=ctx, noc_id=0, safe_mode=False)[0]

    def wr(a, v):
        write_words_to_device(loc, a, [v], context=ctx, noc_id=0, safe_mode=False)

    good = []
    for a in CANDIDATES:
        if not (0 <= a <= PASS_HI):
            print(f"  0x{a:010X}: refused (outside passthrough window)"); continue
        try:
            orig = rd(a)
            wr(a, MARKER)
            back = rd(a)
            wr(a, orig)                       # restore
            restored = rd(a)
            ok = (back == MARKER) and (restored == orig)
            print(f"  0x{a:010X}: orig=0x{orig:08X} wrote=0x{MARKER:08X} read=0x{back:08X} "
                  f"restored=0x{restored:08X}  -> {'WRITABLE ✓' if ok else 'not writable ✗'}")
            if ok:
                good.append(a)
        except Exception as e:                # noqa: BLE001
            print(f"  0x{a:010X}: ERROR {type(e).__name__}: {e}  (possible wedge -> tt-smi -r 0)")
            return 2

    print()
    if good:
        print(f"DRAM is writable. Stage 3 can place code at: " + ", ".join(f"0x{a:08X}" for a in good))
        print(f"(l2cpu_redirect.py defaults to 0x{0x30000000:08X} for the trampoline + 0x{0x30000040:08X} for v2.)")
    else:
        print("No candidate DRAM address was writable — local DRAM may be harvested.")
        print("Stage 3 would then need a different code home; tell me and I'll adapt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
