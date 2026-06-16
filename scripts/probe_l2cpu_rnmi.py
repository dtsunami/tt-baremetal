#!/usr/bin/env python3
# ============================================================================
#  probe_l2cpu_rnmi.py  —  Stage-0 read-only probe of the Blackhole L2CPU
#                          RNMI "seize" path (x280 harts), with full telemetry.
# ============================================================================
#
#  WHY THIS EXISTS
#  ---------------
#  Each Blackhole L2CPU tile is a coherent cluster of four SiFive x280 RISC-V
#  harts. There is a documented hardware bug (tt-isa-documentation,
#  BlackholeA0/L2CPUTile/README.md, "Reset"): the harts can be brought OUT of
#  reset only ONCE. Putting them back into reset requires resetting the entire
#  ASIC (tt-smi -r). To adjust hart code on the fly without that, the docs
#  prescribe an RNMI ("resumable non-maskable interrupt") "seize" mechanism:
#  point a hart's RNMI trap handler at a small scratch stub that runs `fence.i`,
#  loads a per-hart reset-handler address, and jumps there. The host then
#  redirects a hart by (1) writing new code somewhere, (2) updating that hart's
#  reset-handler address word, (3) raising the hart's RNMI trigger bit.
#
#  This script does ONLY Stage 0 of that effort: it READS the relevant registers
#  so we can answer, before writing anything:
#    * Does touching the L2CPU peripheral window over the NoC actually work
#      (and not hang NoC0)?              -> the access-path question
#    * Are the RNMI trap-handler registers currently set, and to what?
#      Are the per-hart reset-handler addresses set? Any RNMI already pending?
#                                        -> current-state / liveness inference
#  An OPT-IN, default-OFF writability test (--writeback-test) additionally
#  answers the make-or-break unknown: are the RNMI handler registers still
#  WRITABLE now that the harts are already running? (The docs recommend setting
#  them before harts leave reset and not modifying them after — if they latched,
#  the whole seize approach needs one final ASIC reset to install the handler.)
#
#  SAFETY  (READ THIS)
#  -------------------
#  L2CPU register access is in the known NoC0-hang hazard zone. The PROVEN-FATAL
#  region is the tile's NIU config/status window at 0xFFFF_FFFF_FF00_0000. The
#  registers below live in a DIFFERENT window — the low 1:1 "passthrough to x280
#  physical address space" (NoC addr == x280 phys, range 0..0x7FFF_FFFF_FFFF,
#  per L2CPUTile/MemoryMap.md "as seen from the NoC"). That passthrough path is
#  the DOCUMENTED, INTENDED way for an external agent to reach these registers,
#  so it is *plausibly* safe — but it is UNVERIFIED on this card and sits right
#  next to the fatal region. Therefore this script:
#    * refuses any address outside the safe passthrough window (guard()),
#    * uses safe_mode=False  (tt-exalens's safe_mode is INVERTED for L2CPU: its
#      only "known" L2CPU address is the dangerous NIU region, so safe_mode would
#      reject these safe reads and bless the fatal one),
#    * passes an EXPLICIT noc_id  (this bypasses tt-exalens's auto NoC-failover,
#      which on a timeout would retry the same poke on the OTHER NoC and could
#      wedge both),
#    * starts with a single one-word "canary" read and ABORTS on the first
#      exception or suspiciously slow read, printing the recovery command.
#
#  IF THE NoC WEDGES:  run   tt-smi -r 0   on ttstar, then restart bhtop-web.
#
#  ADDRESS MAP  (x280 physical == NoC address; L2CPUTile/MemoryMap.md + RNMIs.md)
#  ----------------------------------------------------------------------------
#    0x2001_0000 + N*8   per-hart reset-handler address (initial pc), 8 B each
#    0x2001_0100         general-purpose scratch (64 B) — where the stub goes
#    0x2001_0414         all-harts RNMI trigger; bit N raises RNMI on hart N
#    0x2001_0418 + N*16  per-hart RNMI handlers: trap handler addr (8 B) +
#                        RNMI-exception handler addr (8 B); addresses are 47-bit
#
#  USAGE
#  -----
#    # default: read-only probe of every L2CPU tile over NoC0
#    .venv/bin/python scripts/probe_l2cpu_rnmi.py
#    # one tile, over NoC1
#    .venv/bin/python scripts/probe_l2cpu_rnmi.py --tile 8,3 --noc 1
#    # ALSO test register writability (writes a hart's RNMI handler reg to a
#    # safe value, reads it back, then RESTORES the original). Opt-in + confirm.
#    .venv/bin/python scripts/probe_l2cpu_rnmi.py --writeback-test
#
#  This script is intentionally standalone (only depends on tt-exalens) and does
#  NOTHING on import; all device access is inside main().
# ============================================================================

import argparse
import sys
import time

# --- safe address window (L2CPUTile/MemoryMap.md "as seen from the NoC") -----
PASS_LO, PASS_HI = 0x0, 0x7FFF_FFFF_FFFF      # 1:1 passthrough to x280 physical
NIU_DANGER = 0xFFFF_FFFF_FF00_0000            # proven-fatal NIU config/status window

# --- register map (x280 physical addresses) ---------------------------------
NUM_HARTS      = 4
RESET_HANDLERS = 0x2001_0000                  # +N*8
SCRATCH        = 0x2001_0100                  # 64 B
RNMI_TRIGGER   = 0x2001_0414                  # 1 word; bit N -> hart N
RNMI_HANDLERS  = 0x2001_0418                  # +N*16 (trap[8B] + exc[8B])
HART_STATUS    = 0x2001_0400                  # 2 B: all-harts cease/halt/wfi/debug status

SLOW_READ_S = 0.5                             # a read slower than this smells like a near-wedge


def guard(addr, nbytes):
    """Refuse anything outside the safe x280 passthrough window. This is the last
    line of defence against accidentally poking the NoC0-hang NIU region."""
    end = addr + nbytes - 1
    if not (PASS_LO <= addr and end <= PASS_HI):
        raise ValueError(f"address 0x{addr:08X}..0x{end:08X} is OUTSIDE the safe "
                         f"x280 passthrough window [0..0x{PASS_HI:X}] — refusing")
    if addr >= NIU_DANGER:
        raise ValueError(f"address 0x{addr:08X} is in the FATAL NIU window — refusing")


def a64(words, i):
    """Combine two little-endian 32-bit words into a 64-bit value."""
    return words[i] | (words[i + 1] << 32)


def classify(addr):
    """Human label for a handler/reset address, by L2CPU memory map region."""
    if addr == 0:
        return "unset (0)"
    if addr & 0xFFFF_FFFF_FFFF == 0xFFFF_FFFF_FFFF or addr == (1 << 64) - 1:
        return "unset (all-ones)"
    if 0x0800_0000 <= addr <= 0x081F_FFFF:
        return "L3 LIM scratchpad"
    if SCRATCH <= addr <= SCRATCH + 0x3F:
        return "RNMI scratch stub"
    if 0x2000_0000 <= addr <= 0x2FFF_FFFF:
        return "L2CPU external peripherals"
    if 0x3000_0000 <= addr <= 0x1_2FFF_FFFF:
        return "GDDR6 (uncached)"
    return f"0x{addr:X}"


class Hang(RuntimeError):
    pass


def timed_read(loc, addr, nwords, ctx, noc, dev_id):
    """Guarded, timed, explicit-noc, safe_mode=False read. Raises Hang on any
    device error so the caller can stop immediately rather than keep hammering a
    possibly-wedged NoC."""
    from ttexalens.tt_exalens_lib import read_words_from_device
    guard(addr, nwords * 4)
    t0 = time.monotonic()
    try:
        words = read_words_from_device(loc, addr, device_id=dev_id, word_count=nwords,
                                       context=ctx, noc_id=noc, safe_mode=False)
    except Exception as e:                       # noqa: BLE001 - want EVERYTHING here
        raise Hang(f"read @0x{addr:08X} (x{nwords}) failed: {type(e).__name__}: {e}") from e
    dt = time.monotonic() - t0
    return words, dt


def hexwords(words):
    return " ".join(f"{w:08X}" for w in words)


def banner(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def probe_tile(loc, n0, ctx, noc, dev_id):
    """Read + decode every RNMI/reset/scratch register for one L2CPU tile and
    print full telemetry. Returns a dict of decoded state for the summary."""
    banner(f"L2CPU tile noc0=({n0[0]},{n0[1]})   loc={loc}   via NoC{noc}")
    st = {"n0": n0}

    # ---- reset-handler addresses (per-hart initial pc) ----------------------
    ws, dt = timed_read(loc, RESET_HANDLERS, NUM_HARTS * 2, ctx, noc, dev_id)
    print(f"\n[reset handlers]  0x{RESET_HANDLERS:08X}  ({dt*1e3:.1f} ms)")
    print(f"  raw: {hexwords(ws)}")
    st["reset"] = []
    for j in range(NUM_HARTS):
        addr = a64(ws, j * 2)
        st["reset"].append(addr)
        print(f"  hart{j}: 0x{addr:016X}  [{classify(addr)}]")

    # ---- RNMI trap-handler table --------------------------------------------
    ws, dt = timed_read(loc, RNMI_HANDLERS, NUM_HARTS * 4, ctx, noc, dev_id)
    print(f"\n[RNMI handlers]   0x{RNMI_HANDLERS:08X}  ({dt*1e3:.1f} ms)")
    print(f"  raw: {hexwords(ws)}")
    st["rnmi"] = []
    for j in range(NUM_HARTS):
        trap = a64(ws, j * 4)
        exc = a64(ws, j * 4 + 2)
        st["rnmi"].append(trap)
        print(f"  hart{j}: trap=0x{trap:016X} [{classify(trap)}]   exc=0x{exc:016X} [{classify(exc)}]")

    # ---- RNMI trigger (pending NMIs) ----------------------------------------
    ws, dt = timed_read(loc, RNMI_TRIGGER, 1, ctx, noc, dev_id)
    trig = ws[0]
    st["trigger"] = trig
    pend = [j for j in range(NUM_HARTS) if trig & (1 << j)]
    print(f"\n[RNMI trigger]    0x{RNMI_TRIGGER:08X}  ({dt*1e3:.1f} ms)")
    print(f"  raw: {trig:08X}   pending harts: {pend or 'none'}")

    # ---- hart run/halt status (tells us if the harts are even out of reset) --
    ws, dt = timed_read(loc, HART_STATUS, 1, ctx, noc, dev_id)
    sv = ws[0] & 0xFFFF
    st["status"] = sv
    print(f"\n[hart status]     0x{HART_STATUS:08X}  ({dt*1e3:.1f} ms)")
    print(f"  raw16: {sv:04X}   {'all zero — harts likely still IN RESET (never released)' if sv == 0 else 'non-zero — some cease/halt/wfi/debug bits set'}")

    # ---- scratch (where the stub would live) --------------------------------
    ws, dt = timed_read(loc, SCRATCH, 16, ctx, noc, dev_id)
    nonzero = any(w for w in ws)
    st["scratch_nonzero"] = nonzero
    print(f"\n[scratch 64B]     0x{SCRATCH:08X}  ({dt*1e3:.1f} ms)   {'NON-EMPTY (code?)' if nonzero else 'empty'}")
    for r in range(4):
        chunk = ws[r * 4:r * 4 + 4]
        print(f"  +0x{r*16:02X}: {hexwords(chunk)}")

    return st


def writeback_test(loc, n0, ctx, noc, dev_id):
    """OPT-IN writability test for the make-or-break unknown: are the RNMI handler
    registers still writable after the harts are running?

    Method (idempotent / self-restoring): for hart 0's RNMI trap-handler reg,
    read the original 8 bytes, write a distinct SAFE test value (the scratch
    address — a valid handler target we never arm because we never set the
    trigger), read back, then RESTORE the original and confirm. We NEVER touch
    the RNMI trigger, so no RNMI actually fires during the test."""
    from ttexalens.tt_exalens_lib import write_words_to_device
    addr = RNMI_HANDLERS                          # hart 0 trap handler, 8 bytes
    guard(addr, 8)
    test_lo, test_hi = SCRATCH & 0xFFFFFFFF, (SCRATCH >> 32) & 0xFFFFFFFF

    banner(f"WRITABILITY TEST (writes!) hart0 RNMI handler @0x{addr:08X}  tile ({n0[0]},{n0[1]})")
    orig, _ = timed_read(loc, addr, 2, ctx, noc, dev_id)
    print(f"  original : {hexwords(orig)}")
    try:
        write_words_to_device(loc, addr, [test_lo, test_hi], device_id=dev_id,
                              context=ctx, noc_id=noc, safe_mode=False)
        back, _ = timed_read(loc, addr, 2, ctx, noc, dev_id)
        print(f"  wrote    : {test_lo:08X} {test_hi:08X}")
        print(f"  readback : {hexwords(back)}")
        took = (back[0] == test_lo and back[1] == test_hi)
        print(f"\n  => RNMI handler register is {'WRITABLE' if took else 'NOT writable (latched/ignored)'} "
              f"post-reset {'✓' if took else '✗'}")
        if not took:
            print("     (the seize approach then needs ONE tt-smi -r 0 to install the handler, "
                  "then never reset again)")
        return took
    finally:
        # always put it back exactly as found
        write_words_to_device(loc, addr, [orig[0], orig[1]], device_id=dev_id,
                              context=ctx, noc_id=noc, safe_mode=False)
        restored, _ = timed_read(loc, addr, 2, ctx, noc, dev_id)
        ok = restored[0] == orig[0] and restored[1] == orig[1]
        print(f"  restored : {hexwords(restored)}  {'(verified)' if ok else '!! RESTORE MISMATCH !!'}")


def main():
    ap = argparse.ArgumentParser(description="Stage-0 read-only probe of the L2CPU RNMI seize path.")
    ap.add_argument("--noc", type=int, default=0, choices=(0, 1),
                    help="explicit NoC to use (default 0). If it wedges, run tt-smi -r 0, try --noc 1.")
    ap.add_argument("--tile", default=None, help="limit to one L2CPU tile by noc0 coord, e.g. 8,3")
    ap.add_argument("--device", type=int, default=0, help="device id (default 0)")
    ap.add_argument("--writeback-test", action="store_true",
                    help="ALSO test RNMI-handler register writability (WRITES a safe value, then restores)")
    ap.add_argument("--yes", action="store_true", help="skip the writeback-test confirmation prompt")
    ap.add_argument("--dump-reset", type=int, nargs="?", const=16, default=0, metavar="N",
                    help="also hex-dump N words (default 16) at hart0's reset-vector address — read-only, "
                         "shows what the harts actually booted into before we ever consider seizing them")
    args = ap.parse_args()

    print("L2CPU RNMI probe — READ-ONLY" + ("  +WRITEBACK TEST" if args.writeback_test else ""))
    print(f"  NoC={args.noc}  device={args.device}  safe_mode=False (intentional)")
    print("  L2CPU is in the NoC0-hang hazard zone. If anything wedges: run  tt-smi -r 0  on ttstar.\n")

    from ttexalens import init_ttexalens
    ctx = init_ttexalens()
    dev = ctx.devices[args.device]

    locs = dev.get_block_locations("l2cpu")
    if not locs:
        print("no L2CPU tiles reported by the device — nothing to probe.")
        return 1
    tiles = [(loc, tuple(loc.to("noc0"))) for loc in locs]
    if args.tile:
        want = tuple(int(v) for v in args.tile.split(","))
        tiles = [(l, n) for (l, n) in tiles if n == want]
        if not tiles:
            print(f"no L2CPU tile at noc0 {args.tile}; available: {[n for _, n in [(l, tuple(l.to('noc0'))) for l in locs]]}")
            return 1
    print(f"L2CPU tiles to probe: {[n for _, n in tiles]}")

    # ---- canary: one tiny read before touching anything else ----------------
    loc0, n0_0 = tiles[0]
    print(f"\ncanary: reading 1 word @0x{RESET_HANDLERS:08X} on ({n0_0[0]},{n0_0[1]}) via NoC{args.noc} ...")
    try:
        w, dt = timed_read(loc0, RESET_HANDLERS, 1, ctx, args.noc, args.device)
    except Hang as e:
        print(f"\n!!! CANARY FAILED: {e}")
        print("!!! The L2CPU passthrough path did not respond. Possible NoC wedge.")
        print("!!! Recover with:  tt-smi -r 0   (then restart bhtop-web). Aborting.")
        return 2
    print(f"canary ok: {w[0]:08X}  ({dt*1e3:.1f} ms){'  [SLOW — caution]' if dt > SLOW_READ_S else ''}")

    states = []
    try:
        for loc, n0 in tiles:
            states.append(probe_tile(loc, n0, ctx, args.noc, args.device))
    except Hang as e:
        print(f"\n!!! STOP: {e}")
        print("!!! Aborting further reads. If the NoC is wedged, run:  tt-smi -r 0")
        return 2

    # ---- optional read-only dump of the reset-vector target -----------------
    if args.dump_reset:
        addr = states[0]["reset"][0]
        banner(f"reset-vector dump: {args.dump_reset} words @0x{addr:08X}  (hart0 of {tiles[0][1]})")
        try:
            ws, dt = timed_read(tiles[0][0], addr, args.dump_reset, ctx, args.noc, args.device)
        except (Hang, ValueError) as e:
            print(f"  cannot dump: {e}")
        else:
            print(f"  {'NON-ZERO (code present)' if any(ws) else 'all zero (no code at reset vector)'}  ({dt*1e3:.1f} ms)")
            for r in range((len(ws) + 3) // 4):
                print(f"  +0x{r*16:02X}: {hexwords(ws[r*4:r*4+4])}")

    # ---- optional writability test ------------------------------------------
    writable = None
    if args.writeback_test:
        if not args.yes:
            print("\n--writeback-test WRITES to an L2CPU RNMI handler register (then restores it).")
            if input("type 'yes' to proceed: ").strip().lower() != "yes":
                print("skipped writeback test.")
                args.writeback_test = False
        if args.writeback_test:
            try:
                writable = writeback_test(*tiles[0], ctx, args.noc, args.device)
            except Hang as e:
                print(f"\n!!! STOP during writeback test: {e}\n!!! Recover with:  tt-smi -r 0")
                return 2

    # ---- verdict ------------------------------------------------------------
    banner("VERDICT")
    print(f"  access path  : reads via NoC{args.noc} SUCCEEDED on {len(states)} tile(s) — "
          f"passthrough peripheral window is reachable, no hang observed.")
    any_rnmi = any(any(r and classify(r) != 'unset (0)' for r in s['rnmi']) for s in states)
    any_reset = any(any(r for r in s['reset']) for s in states)
    print(f"  RNMI handlers: {'SET on >=1 hart (firmware already wired a handler)' if any_rnmi else 'all unset (no seize handler installed yet)'}")
    print(f"  reset addrs  : {'set on >=1 hart' if any_reset else 'all zero'}")
    print(f"  pending RNMIs: {[s['trigger'] for s in states]}")
    if writable is not None:
        print(f"  WRITABILITY  : RNMI handler reg is {'WRITABLE now ✓ (seize installable in-place)' if writable else 'NOT writable ✗ (needs one ASIC reset to install)'}")
    else:
        print("  WRITABILITY  : not tested (re-run with --writeback-test to answer the make-or-break unknown)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
