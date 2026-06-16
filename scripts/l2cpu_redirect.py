#!/usr/bin/env python3
# ============================================================================
#  l2cpu_redirect.py — adjust a RUNNING L2CPU hart's code on the fly (RNMI).
# ============================================================================
#
#  This is the payoff: hart 0 of tile (8,3) is out of reset running our v1
#  heartbeat (counter climbing at 0x2001_0138). We redirect it — with NO reset —
#  to DIFFERENT code (v2, counting at 0x2001_0130) using the documented RNMI
#  "seize" mechanism, and show the behavior change live.
#
#  HOW (ISA L2CPUTile/{README,RNMIs}.md):
#    * trampoline in DRAM @0x3000_0000:  fence.i; ld t1,[reset-vec 0x2001_0000]; jr t1
#    * v2 code in DRAM    @0x3000_0040:  clears RNMI trigger; sets mnstatus.NMIE=1
#      (so the hart stays re-seizable); then counts at 0x2001_0130
#    * point hart0's RNMI trap handler (0x2001_0418) at the trampoline,
#      its RNMI *exception* handler (0x2001_0420) at a safe spin (so a fault
#      parks the hart instead of running into garbage),
#      and its reset-vector (0x2001_0000) at v2,
#    * pull the RNMI trigger bit (0x2001_0414 bit0): hart traps -> trampoline ->
#      fence.i -> loads reset-vec (=v2) -> jumps to v2. v1 counter freezes, v2
#      counter starts climbing.
#  Because v2 re-enables NMIE and clears the trigger, the redirect is REPEATABLE:
#  to load v3 later, write it, set reset-vec to it, pull the trigger again.
#
#  REQUIRES: bringup done (harts running) + DRAM writable (run l2cpu_dram_probe.py).
#  Transport: tt-exalens (proven L2CPU NoC path). Blobs are sfpi-assembled.
#  Default STAGES only (writes code + handlers, no trigger — hart keeps running
#  v1); --fire pulls the trigger. If anything wedges: tt-smi -r 0.
# ============================================================================

import argparse
import sys
import time

XY = (8, 3)
PASS_HI = 0x7FFF_FFFF_FFFF

# DRAM homes for the new code (confirm writable with l2cpu_dram_probe.py)
REDIR_ADDR = 0x30000000
V2_ADDR    = 0x30000040

# L2CPU peripheral registers (x280 phys, hart 0)
RESET_VEC = 0x20010000           # +0/+4 = low/high 32 of hart0 initial pc
RNMI_TRAP = 0x20010418           # hart0 RNMI trap handler addr
RNMI_EXC  = 0x20010420           # hart0 RNMI exception handler addr
TRIGGER   = 0x20010414           # bit0 -> RNMI on hart0
SPIN_ADDR = 0x20010120           # safe spin (installed by bringup) for the exc handler
V1_CTR    = 0x20010138           # v1 heartbeat counter (should FREEZE after redirect)
V2_CTR    = 0x20010130           # v2 heartbeat counter (should CLIMB after redirect)

# sfpi-assembled + disassembly-verified
REDIR = [0x0000100F, 0x200102B7, 0x0002B303, 0x00030067]
V2    = [0x200102B7, 0x4002AA23, 0x35346073, 0x00000313, 0x00130313, 0x1262A823, 0xFF9FF06F]


class Hang(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser(description="Redirect a running L2CPU hart to new code on the fly (RNMI).")
    ap.add_argument("--fire", action="store_true", help="pull the RNMI trigger (does the live redirect)")
    ap.add_argument("--yes", action="store_true", help="skip the --fire confirmation prompt")
    args = ap.parse_args()

    print(f"L2CPU on-the-fly redirect — tile {XY} hart 0 ({'FIRE' if args.fire else 'STAGE only'})")
    print("  v1 counter @0x{:08X} -> redirect -> v2 counter @0x{:08X}".format(V1_CTR, V2_CTR))
    print("  transport: tt-exalens. If anything wedges: tt-smi -r 0.\n")

    from ttexalens import init_ttexalens
    from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device
    ctx = init_ttexalens()
    dev = ctx.devices[0]
    loc = next((l for l in dev.get_block_locations("l2cpu") if tuple(l.to("noc0")) == XY), None)
    if loc is None:
        print("no L2CPU tile at (8,3)"); return 1

    def rd(a):
        if not (0 <= a <= PASS_HI):
            raise ValueError(f"addr 0x{a:X} outside passthrough window")
        try:
            return read_words_from_device(loc, a, word_count=1, context=ctx, noc_id=0, safe_mode=False)[0]
        except Exception as e:               # noqa: BLE001
            raise Hang(f"read 0x{a:X}: {type(e).__name__}: {e}") from e

    def wr(a, words):
        if not (0 <= a <= PASS_HI):
            raise ValueError(f"addr 0x{a:X} outside passthrough window")
        try:
            write_words_to_device(loc, a, list(words), context=ctx, noc_id=0, safe_mode=False)
        except Exception as e:               # noqa: BLE001
            raise Hang(f"write 0x{a:X}: {type(e).__name__}: {e}") from e

    def climbing(addr, dt=0.05):
        a = rd(addr); time.sleep(dt); b = rd(addr)
        return a, b, (a != b)

    try:
        # ---- canary: hart0 should be alive on v1 ----
        rv = rd(RESET_VEC)
        a, b, climb = climbing(V1_CTR)
        print(f"canary: reset-vec[h0]=0x{rv:08X}   v1 counter {a:#x} -> {b:#x}  "
              f"({'CLIMBING — hart0 alive on v1' if climb else 'NOT moving — hart0 not on v1?'})")
        if not climb:
            print("  v1 not climbing — bring up the harts first (l2cpu_bringup.py --release). Aborting.")
            return 1

        # ---- stage code in DRAM ----
        wr(REDIR_ADDR, REDIR)
        wr(V2_ADDR, V2)
        rb_r = [rd(REDIR_ADDR + i * 4) for i in range(len(REDIR))]
        rb_v = [rd(V2_ADDR + i * 4) for i in range(len(V2))]
        if rb_r != REDIR or rb_v != V2:
            print("DRAM code did not read back (run l2cpu_dram_probe.py). Aborting."); return 2
        print(f"staged trampoline @0x{REDIR_ADDR:08X} + v2 @0x{V2_ADDR:08X}  (verified)")

        # ---- point the RNMI handlers + reset-vector (harmless until trigger) ----
        wr(RNMI_TRAP, [REDIR_ADDR, 0])           # RNMI -> trampoline
        wr(RNMI_EXC, [SPIN_ADDR, 0])             # fault during RNMI -> safe spin
        wr(RESET_VEC, [V2_ADDR, 0])              # trampoline will jump here
        print(f"set RNMI trap[h0]=0x{rd(RNMI_TRAP):08X}  exc[h0]=0x{rd(RNMI_EXC):08X}  reset-vec[h0]=0x{rd(RESET_VEC):08X}")

        if not args.fire:
            print("\nSTAGED ONLY. Code + handlers set; trigger NOT pulled — hart0 still running v1.")
            print("Re-run with --fire to pull the RNMI trigger and redirect hart0 to v2.")
            return 0

        if not args.yes:
            print("\n--fire will SEIZE hart0 and redirect it from v1 to v2 (live, no reset).")
            if input("type 'fire' to proceed: ").strip().lower() != "fire":
                print("aborted; left staged (hart0 still on v1)."); return 0

        # ---- fire ----
        wr(V2_CTR, [0])                          # clear v2 counter so a climb is unambiguous
        v1_before = climbing(V1_CTR)
        wr(TRIGGER, [0x1])                       # RNMI on hart0  <-- the seize
        time.sleep(0.1)
        v1_after = climbing(V1_CTR)
        v2_after = climbing(V2_CTR)
        trig = rd(TRIGGER)

        v1_frozen = not v1_after[2]
        v2_climbing = v2_after[2]
        print(f"\nfired RNMI trigger bit0.")
        print(f"  v1 counter @0x{V1_CTR:08X}: {v1_after[0]:#x} -> {v1_after[1]:#x}  ({'FROZEN ✓' if v1_frozen else 'still moving ✗'})")
        print(f"  v2 counter @0x{V2_CTR:08X}: {v2_after[0]:#x} -> {v2_after[1]:#x}  ({'CLIMBING ✓' if v2_climbing else 'not moving ✗'})")
        print(f"  trigger now 0x{trig:08X}  (v2 clears it; expect 0)")

        print("\n" + "=" * 62)
        if v1_frozen and v2_climbing:
            print("  SUCCESS: hart0 redirected v1 -> v2 LIVE, with no reset.")
            print("  Repeatable: write v3 to DRAM, set reset-vec[h0] to it, pull the trigger again.")
        else:
            print("  Redirect did not cleanly take. Hart0 may have faulted into the spin (check v2).")
            print("  If hart0 is stuck, tt-smi -r 0 and re-bring-up.")
        print("=" * 62)
        return 0

    except Hang as e:
        print(f"\n!!! STOP: {e}\n!!! Possible wedge — recover with: tt-smi -r 0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
