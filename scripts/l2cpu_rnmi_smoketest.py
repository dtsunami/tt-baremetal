#!/usr/bin/env python3
# ============================================================================
#  l2cpu_rnmi_smoketest.py — Stage-1 RNMI "seize" smoke test for one L2CPU hart.
# ============================================================================
#
#  GOAL
#  ----
#  Prove the last open unknown from Stage 0 (probe_l2cpu_rnmi.py): are the x280
#  harts ALIVE and will they actually TAKE an RNMI? We do it the only way you can
#  — fire one and watch for a side effect — but with a self-verifying stub so the
#  result is unambiguous:
#
#    1. write a tiny machine-code stub into the L2CPU scratch (0x2001_0100) that
#       stores a sentinel word (0xC0FFEE) to 0x2001_0138 and then spins,
#    2. point the hart's RNMI trap handler (0x2001_0418 + hart*16) at the stub,
#    3. raise the hart's RNMI trigger bit (0x2001_0414),
#    4. poll 0x2001_0138 — if it reads 0xC0FFEE, the hart took the NMI, executed
#       OUR code, and is now parked in the stub's spin loop = SEIZED.
#
#  Stage-0 already established (on this card): the L2CPU passthrough window is
#  reachable over NoC0 without hanging; the RNMI handler registers are writable
#  post-reset; and the harts' reset vector (0xD000_0000) holds GARBAGE, not real
#  firmware — so seizing a hart interrupts nothing meaningful.
#
#  *** WHAT RUNNING THIS DOES — READ ***
#  With --fire this is the FIRST genuinely disruptive, *effectively irreversible*
#  step: once a hart takes the RNMI and enters the spin stub it stays parked
#  there until a full ASIC reset (tt-smi -r 0). That is the intended outcome (we
#  WANT control of the hart), but be deliberate. By default (no --fire) the
#  script only STAGES the stub into scratch (harmless data in empty scratch) and
#  prints the plan; it does NOT set the handler or pull the trigger.
#
#  Defaults to ONE hart on ONE tile so the other 15 harts stay pristine.
#
#  SAFETY: same rules as the probe — guarded addresses (safe x280 passthrough
#  window only), safe_mode=False (it's inverted for L2CPU), explicit noc_id (no
#  failover), abort on first device error. If anything wedges: tt-smi -r 0.
#
#  THE STUB  (RV64, hand-assembled; each word hand-decoded to verify its fields)
#  ----------------------------------------------------------------------------
#    0x200102B7  lui  t0, 0x20010      ; t0 = 0x2001_0000  (scratch/peripheral base)
#    0x00C10337  lui  t1, 0xC10        ; t1 = 0x00C1_0000
#    0xFEE30313  addi t1, t1, -18      ; t1 = 0x00C0_FFEE  (the sentinel)
#    0x1262AC23  sw   t1, 0x138(t0)    ; *(0x2001_0138) = 0x00C0_FFEE
#    0x0000006F  j    .                ; spin here forever (parked, mnstatus.NMIE stays 0)
#  Stub occupies 0x2001_0100..0x113 (20 B); sentinel slot 0x2001_0138 is clear of it.
#  No fence.i needed: the scratch/peripheral region is uncached, the hart never
#  executed here before, and we are not loading cached app code (that's Stage 2).
#
#  USAGE
#  -----
#    .venv/bin/python scripts/l2cpu_rnmi_smoketest.py                 # STAGE only (safe)
#    .venv/bin/python scripts/l2cpu_rnmi_smoketest.py --fire          # actually seize hart 0 of tile 8,3
#    .venv/bin/python scripts/l2cpu_rnmi_smoketest.py --tile 8,5 --hart 2 --fire
# ============================================================================

import argparse
import sys
import time

PASS_HI = 0x7FFF_FFFF_FFFF
NIU_DANGER = 0xFFFF_FFFF_FF00_0000

SCRATCH       = 0x2001_0100               # stub goes here; RNMI handler points here
SENTINEL_ADDR = 0x2001_0138               # stub stores the sentinel here (within scratch)
SENTINEL      = 0x00C0_FFEE
RNMI_TRIGGER  = 0x2001_0414               # bit N -> RNMI on hart N
RNMI_HANDLERS = 0x2001_0418               # +hart*16: trap handler (8B) + exc handler (8B)

STUB = [0x200102B7, 0x00C10337, 0xFEE30313, 0x1262AC23, 0x0000006F]


def guard(addr, nbytes):
    end = addr + nbytes - 1
    if not (0 <= addr and end <= PASS_HI):
        raise ValueError(f"addr 0x{addr:X}..0x{end:X} outside safe x280 passthrough window — refusing")
    if addr >= NIU_DANGER:
        raise ValueError(f"addr 0x{addr:X} in FATAL NIU window — refusing")


class Hang(RuntimeError):
    pass


def rd(loc, addr, n, ctx, noc, dev):
    from ttexalens.tt_exalens_lib import read_words_from_device
    guard(addr, n * 4)
    try:
        return read_words_from_device(loc, addr, device_id=dev, word_count=n,
                                      context=ctx, noc_id=noc, safe_mode=False)
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"read @0x{addr:X} x{n}: {type(e).__name__}: {e}") from e


def wr(loc, addr, words, ctx, noc, dev):
    from ttexalens.tt_exalens_lib import write_words_to_device
    guard(addr, len(words) * 4)
    try:
        write_words_to_device(loc, addr, list(words), device_id=dev,
                              context=ctx, noc_id=noc, safe_mode=False)
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"write @0x{addr:X} x{len(words)}: {type(e).__name__}: {e}") from e


def hexw(ws):
    return " ".join(f"{w:08X}" for w in ws)


def main():
    ap = argparse.ArgumentParser(description="Stage-1 RNMI seize smoke test for one L2CPU hart.")
    ap.add_argument("--tile", default="8,3", help="L2CPU tile noc0 coord (default 8,3)")
    ap.add_argument("--hart", type=int, default=0, choices=(0, 1, 2, 3), help="hart 0-3 (default 0)")
    ap.add_argument("--noc", type=int, default=0, choices=(0, 1), help="explicit NoC (default 0)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fire", action="store_true",
                    help="actually set the handler + pull the RNMI trigger (SEIZES the hart; irreversible "
                         "until tt-smi -r 0). Without this, only stages the stub.")
    ap.add_argument("--yes", action="store_true", help="skip the --fire confirmation prompt")
    args = ap.parse_args()

    want = tuple(int(v) for v in args.tile.split(","))
    hreg = RNMI_HANDLERS + args.hart * 16

    print(f"L2CPU RNMI smoke test — tile {want} hart {args.hart} via NoC{args.noc} "
          f"({'FIRE' if args.fire else 'STAGE only'})")
    print("  safe_mode=False, explicit noc_id. If anything wedges: tt-smi -r 0 on ttstar.\n")

    from ttexalens import init_ttexalens
    ctx = init_ttexalens()
    dev = ctx.devices[args.device]
    loc = next((l for l in dev.get_block_locations("l2cpu") if tuple(l.to("noc0")) == want), None)
    if loc is None:
        avail = [tuple(l.to("noc0")) for l in dev.get_block_locations("l2cpu")]
        print(f"no L2CPU tile at {want}; available: {avail}")
        return 1

    try:
        # ---- before snapshot ----
        h0 = rd(loc, hreg, 2, ctx, args.noc, args.device)
        trig0 = rd(loc, RNMI_TRIGGER, 1, ctx, args.noc, args.device)[0]
        sent0 = rd(loc, SENTINEL_ADDR, 1, ctx, args.noc, args.device)[0]
        print(f"before: handler[h{args.hart}]={hexw(h0)}  trigger={trig0:08X}  sentinel={sent0:08X}")

        # ---- stage the stub (harmless: data into empty scratch) ----
        wr(loc, SCRATCH, STUB, ctx, args.noc, args.device)
        wr(loc, SENTINEL_ADDR, [0], ctx, args.noc, args.device)        # clear sentinel
        back = rd(loc, SCRATCH, len(STUB), ctx, args.noc, args.device)
        ok = back == STUB
        print(f"staged stub @0x{SCRATCH:X}: {hexw(back)}  {'(verified)' if ok else '!! MISMATCH !!'}")
        if not ok:
            print("stub did not read back identically — aborting before doing anything live.")
            return 2

        if not args.fire:
            print("\nSTAGED ONLY. Stub is in scratch; handler NOT set, trigger NOT pulled — hart untouched.")
            print(f"Re-run with --fire to set handler[h{args.hart}]=0x{SCRATCH:X} and trigger the RNMI.")
            print("(That seizes the hart into the spin stub — irreversible until tt-smi -r 0.)")
            return 0

        # ---- fire ----
        if not args.yes:
            print(f"\n--fire will SEIZE hart {args.hart} of tile {want} (parks it until ASIC reset).")
            if input("type 'fire' to proceed: ").strip().lower() != "fire":
                print("aborted; left staged (handler not set, trigger not pulled).")
                return 0

        wr(loc, hreg, [SCRATCH & 0xFFFFFFFF, 0], ctx, args.noc, args.device)   # RNMI trap handler -> stub
        h1 = rd(loc, hreg, 2, ctx, args.noc, args.device)
        print(f"\nset handler[h{args.hart}] -> {hexw(h1)}")

        trig = rd(loc, RNMI_TRIGGER, 1, ctx, args.noc, args.device)[0]
        wr(loc, RNMI_TRIGGER, [trig | (1 << args.hart)], ctx, args.noc, args.device)  # FIRE
        print(f"pulled RNMI trigger bit {args.hart} (0x{trig:08X} -> 0x{trig | (1 << args.hart):08X})")

        # ---- poll for the sentinel ----
        fired = False
        for i in range(40):
            s = rd(loc, SENTINEL_ADDR, 1, ctx, args.noc, args.device)[0]
            if s == SENTINEL:
                fired = True
                print(f"  poll {i}: sentinel=0x{s:08X}  <-- RNMI FIRED, hart executed our stub")
                break
            time.sleep(0.025)
        if not fired:
            last = rd(loc, SENTINEL_ADDR, 1, ctx, args.noc, args.device)[0]
            print(f"  sentinel never set (last=0x{last:08X}) after ~1s")

        # ---- tidy: clear our trigger bit (hart already parked in spin/NMIE=0) ----
        tnow = rd(loc, RNMI_TRIGGER, 1, ctx, args.noc, args.device)[0]
        wr(loc, RNMI_TRIGGER, [tnow & ~(1 << args.hart)], ctx, args.noc, args.device)

        print("\n" + "=" * 60)
        if fired:
            print(f"  RESULT: hart {args.hart} is ALIVE, took the RNMI, and is now PARKED in our stub.")
            print(f"  => Stage 2 (full park+redirect stub for adjust-on-the-fly) is unblocked.")
        else:
            print(f"  RESULT: no sentinel. Hart {args.hart} did NOT visibly take the RNMI.")
            print(f"  Possible: hart wedged with NMIE=0, or not running. Try another hart/tile.")
        print("=" * 60)
        return 0

    except Hang as e:
        print(f"\n!!! STOP: {e}\n!!! Possible NoC wedge — recover with: tt-smi -r 0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
