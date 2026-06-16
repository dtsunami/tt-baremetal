#!/usr/bin/env python3
# ============================================================================
#  l2cpu_bringup.py — ONE-SHOT L2CPU (x280) reset-release + heartbeat bootstrap.
# ============================================================================
#
#  WHAT / WHY
#  ----------
#  The L2CPU harts boot held in reset (probe showed RNMI never fires; hart-status
#  0x2001_0400 reads 0). To run ANY hart code — and to make the RNMI "adjust on
#  the fly" seize meaningful — they must be brought out of reset ONCE. This script
#  does that on one L2CPU tile and boots a tiny verified heartbeat so we can SEE
#  the harts run our code: hart 0 increments a counter at 0x2001_0138; harts 1-3
#  spin. The host polls the counter — if it climbs, the harts are alive on OUR code.
#
#  TRANSPORT (learned the hard way — see below)
#  --------------------------------------------
#  L2CPU NoC access goes through **tt-exalens** (init_ttexalens + read/write_words_
#  from/to_device on the tile's OnChipCoordinate). This is the path we PROVED reads
#  and writes the tile correctly. ARC access (PLL + reset) goes through **pyluwen
#  axi_*** — a separate transport from NoC-to-ARC, which is the SAFE ARC path
#  (NoC-to-ARC is the NoC0-hang hazard). tt-exalens is initialized FIRST (it sets
#  up NoC access); pyluwen is used ONLY for ARC.
#
#  (An earlier version used pyluwen for the NoC too; standalone pyluwen hadn't
#  initialized NoC access, its reads returned 0xFFFFFFFF, and writing into that
#  state wedged NoC0. Hence: a mandatory canary read before any write, and treating
#  an all-ones register read as an ERROR, never as a valid value.)
#
#  *** IRREVERSIBLE ***  Silicon bug: harts leave reset only ONCE; re-resetting
#  needs a full ASIC reset (tt-smi -r 0). --release is one-shot. By DEFAULT this
#  only STAGES (writes bootstrap + reset vectors while harts are in reset —
#  harmless); --release does the clock glide + reset flip + heartbeat poll.
#
#  Mirrors tenstorrent/tt-bh-linux boot.py + clock.py: clock->200MHz BEFORE the
#  flip (ISA hard requirement), L2CPU_RESET=ARC 0x8003_0014 bit(idx+4) RMW 0->1,
#  clock->1750MHz after. Bootstraps assembler-verified (sfpi riscv-tt-elf-as).
#
#  USAGE
#    .venv/bin/python scripts/l2cpu_bringup.py               # STAGE + show (safe)
#    .venv/bin/python scripts/l2cpu_bringup.py --release     # one-shot bringup of tile 8,3
#    .venv/bin/python scripts/l2cpu_bringup.py --tile 1 --release   # tile idx 1 = (8,9)
# ============================================================================

import argparse
import ctypes
import sys
import time

# ---- ARC tile (via pyluwen axi_*) ------------------------------------------
PLL4_BASE   = 0x80020500
PLL_CNTL_1  = 0x4
PLL_CNTL_5  = 0x14
L2CPU_RESET = 0x80030014
ARC_ALLOW   = {PLL4_BASE + PLL_CNTL_1, PLL4_BASE + PLL_CNTL_5, L2CPU_RESET}
SOL = {200: [128, [15, 15, 15, 15]], 1750: [140, [1, 1, 1, 1]]}     # clock.py verbatim

# ---- L2CPU tiles: index -> (noc0 coord, L2CPU_RESET bit) (ISA README.md) ----
TILES = {0: ((8, 3), 4), 1: ((8, 9), 5), 2: ((8, 5), 6), 3: ((8, 7), 7)}

# ---- L2CPU peripheral addresses (x280 phys == NoC addr) --------------------
PASS_HI    = 0x7FFF_FFFF_FFFF
RESET_VEC  = 0x2001_0000
SCRATCH    = 0x2001_0100
SPIN_ADDR  = 0x2001_0120
COUNTER    = 0x2001_0138
HART_STATUS = 0x2001_0400

# assembler-verified (sfpi riscv-tt-elf-as -march=rv32i)
HEARTBEAT = [0x200102B7, 0x00000313, 0x00130313, 0x1262AC23, 0xFF9FF06F]  # ++counter @0x20010138
SPIN      = [0x0000006F]                                                   # j .


class Hang(RuntimeError):
    pass


# ---- PLL glide (ported verbatim from tt-bh-linux clock.py) ------------------
class PLLCNTL5(ctypes.LittleEndianStructure):
    _fields_ = [("postdiv", 4 * ctypes.c_uint8)]

    def step(self, chip, target, field):
        one = max(min(target - self.postdiv[field], 1), -1)
        while self.postdiv[field] != target:
            self.postdiv[field] += one
            chip.axi_write(PLL4_BASE + PLL_CNTL_5, bytearray(self))
            time.sleep(1e-9)


class PLLCNTL1(ctypes.LittleEndianStructure):
    _fields_ = [("refdiv", ctypes.c_uint8), ("postdiv", ctypes.c_uint8), ("fbdiv", ctypes.c_uint16)]

    def step_fbdiv(self, chip, target):
        one = max(min(target - self.fbdiv, 1), -1)
        while self.fbdiv != target:
            self.fbdiv += one
            chip.axi_write(PLL4_BASE + PLL_CNTL_1, bytearray(self))
            time.sleep(1e-9)


def set_l2cpu_pll(chip, mhz):
    fb, pds = SOL[mhz]
    b5 = bytearray(4); chip.axi_read(PLL4_BASE + PLL_CNTL_5, b5); p5 = PLLCNTL5.from_buffer(b5)
    b1 = bytearray(4); chip.axi_read(PLL4_BASE + PLL_CNTL_1, b1); p1 = PLLCNTL1.from_buffer(b1)
    for i, t in [(i, t) for i, t in enumerate(pds) if t > p5.postdiv[i]]:
        p5.step(chip, t, i)
    p1.step_fbdiv(chip, fb)
    for i, t in [(i, t) for i, t in enumerate(pds) if t < p5.postdiv[i]]:
        p5.step(chip, t, i)
    print(f"  L2SYS clock -> {mhz} MHz")


# ---- L2CPU NoC via tt-exalens (the proven path) ----------------------------
def noc_r(ctx, loc, addr):
    from ttexalens.tt_exalens_lib import read_words_from_device
    if not (0 <= addr <= PASS_HI):
        raise ValueError(f"NoC addr 0x{addr:X} outside safe passthrough window — refusing")
    try:
        return read_words_from_device(loc, addr, word_count=1, context=ctx, noc_id=0, safe_mode=False)[0]
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"noc read 0x{addr:X}: {type(e).__name__}: {e}") from e


def noc_w(ctx, loc, addr, words):
    from ttexalens.tt_exalens_lib import write_words_to_device
    if not (0 <= addr <= PASS_HI):
        raise ValueError(f"NoC addr 0x{addr:X} outside safe passthrough window — refusing")
    try:
        write_words_to_device(loc, addr, list(words), context=ctx, noc_id=0, safe_mode=False)
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"noc write 0x{addr:X}: {type(e).__name__}: {e}") from e


# ---- ARC via pyluwen axi (allowlisted) -------------------------------------
def axi_r(chip, addr):
    if addr not in ARC_ALLOW:
        raise ValueError(f"ARC addr 0x{addr:08X} not in allowlist — refusing")
    try:
        return chip.axi_read32(addr)
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"axi_read32 0x{addr:08X}: {type(e).__name__}: {e}") from e


def axi_w(chip, addr, val):
    if addr not in ARC_ALLOW:
        raise ValueError(f"ARC addr 0x{addr:08X} not in allowlist — refusing")
    try:
        chip.axi_write32(addr, val)
    except Exception as e:                       # noqa: BLE001
        raise Hang(f"axi_write32 0x{addr:08X}: {type(e).__name__}: {e}") from e


def main():
    ap = argparse.ArgumentParser(description="One-shot L2CPU reset-release + heartbeat bootstrap.")
    ap.add_argument("--tile", type=int, default=0, choices=(0, 1, 2, 3),
                    help="L2CPU tile index (0=8,3  1=8,9  2=8,5  3=8,7). Default 0.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--release", action="store_true",
                    help="ONE-SHOT: glide clock to 200MHz, flip L2CPU_RESET (irreversible), raise to 1750, "
                         "poll heartbeat. Without this, only stages the bootstrap (safe).")
    ap.add_argument("--yes", action="store_true", help="skip the --release confirmation prompt")
    args = ap.parse_args()

    xy, bit = TILES[args.tile]
    print(f"L2CPU bringup — tile index {args.tile} = noc0 {xy}, L2CPU_RESET bit {bit} "
          f"({'RELEASE' if args.release else 'STAGE only'})")
    print("  NoC via tt-exalens, ARC via pyluwen axi. If anything wedges: tt-smi -r 0.\n")

    # tt-exalens FIRST (it initializes NoC access), then pyluwen for ARC only.
    from ttexalens import init_ttexalens
    ctx = init_ttexalens()
    dev = ctx.devices[args.device]
    loc = next((l for l in dev.get_block_locations("l2cpu") if tuple(l.to("noc0")) == xy), None)
    if loc is None:
        print(f"no L2CPU tile at {xy}"); return 1
    from pyluwen import PciChip
    chip = PciChip(pci_interface=args.device)

    try:
        # ---- CANARY: confirm NoC reaches the tile before ANY write ----
        canary = noc_r(ctx, loc, RESET_VEC)
        if canary == 0xFFFFFFFF:
            print(f"!!! canary read of 0x{RESET_VEC:08X} = 0xFFFFFFFF (no response). NoC not reaching tile.")
            print("!!! Do NOT proceed. Recover with tt-smi -r 0 and retry. Aborting."); return 2
        rst = axi_r(chip, L2CPU_RESET)
        if rst == 0xFFFFFFFF:
            print("!!! ARC L2CPU_RESET read = 0xFFFFFFFF (bus error/wedge). Recover with tt-smi -r 0."); return 2
        already = bool(rst & (1 << bit))
        status = noc_r(ctx, loc, HART_STATUS) & 0xFFFF
        print(f"canary reset-vec[h0]=0x{canary:08X} (sane)   hart status=0x{status:04X}")
        print(f"L2CPU_RESET=0x{rst:08X}  (tile bit {bit} = {'SET — already released' if already else 'clear — in reset'})")
        if already:
            print("\n!!! This tile's harts are ALREADY out of reset. Cannot re-release (reset-once bug).")
            print("!!! To start over: tt-smi -r 0, then re-run. Aborting."); return 1

        # ---- stage bootstraps + reset vectors (harmless while in reset) ----
        noc_w(ctx, loc, SCRATCH, HEARTBEAT)
        noc_w(ctx, loc, SPIN_ADDR, SPIN)
        noc_w(ctx, loc, COUNTER, [0])
        for h in range(4):
            tgt = SCRATCH if h == 0 else SPIN_ADDR
            noc_w(ctx, loc, RESET_VEC + h * 8, [tgt, 0])     # low32, high32
        back = [noc_r(ctx, loc, SCRATCH + i * 4) for i in range(len(HEARTBEAT))]
        vecs = [noc_r(ctx, loc, RESET_VEC + h * 8) for h in range(4)]
        ok = back == HEARTBEAT
        print(f"\nstaged heartbeat @0x{SCRATCH:X}: {' '.join(f'{w:08X}' for w in back)}  {'(verified)' if ok else '!! MISMATCH !!'}")
        print("reset vectors: " + "  ".join(f"h{h}=0x{v:08X}" for h, v in enumerate(vecs)))
        if not ok:
            print("bootstrap did not read back — aborting before touching reset."); return 2

        if not args.release:
            print("\nSTAGED ONLY. Bootstrap + reset vectors set; clock/reset untouched, harts still in reset.")
            print("Re-run with --release to glide the clock and flip L2CPU_RESET (ONE-SHOT, irreversible).")
            return 0

        # ---- release (one-shot) ----
        if not args.yes:
            print(f"\n--release will bring tile {xy} harts OUT OF RESET — IRREVERSIBLE until tt-smi -r 0.")
            if input("type 'release' to proceed: ").strip().lower() != "release":
                print("aborted; left staged (harts still in reset)."); return 0

        print("\nreleasing:")
        set_l2cpu_pll(chip, 200)                              # clock LOW first (required)
        v = axi_r(chip, L2CPU_RESET)
        if v == 0xFFFFFFFF or (v & (1 << bit)):
            print(f"  L2CPU_RESET=0x{v:08X} unexpected — aborting before the flip."); return 2
        axi_w(chip, L2CPU_RESET, v | (1 << bit))             # THE FLIP (0->1)
        rb = axi_r(chip, L2CPU_RESET)
        print(f"  L2CPU_RESET -> 0x{rb:08X}  (bit {bit} {'set ✓' if rb & (1 << bit) else 'NOT set ✗'})")
        set_l2cpu_pll(chip, 1750)                             # raise to run speed

        # ---- poll the heartbeat ----
        print(f"\npolling heartbeat @0x{COUNTER:X} ...")
        seen = []
        for _ in range(30):
            seen.append(noc_r(ctx, loc, COUNTER))
            if seen[-1] != seen[0]:
                print(f"  counter moving: {seen[0]:#x} -> {seen[-1]:#x}  <-- harts ALIVE, running our bootstrap")
                break
            time.sleep(0.05)
        moved = len(set(seen)) > 1
        print("\n" + "=" * 60)
        if moved:
            print(f"  SUCCESS: tile {xy} hart 0 is OUT OF RESET executing our heartbeat.")
            print("  => Stage 3 (real bootstrap: RNMI park + on-the-fly reload) is unblocked.")
        else:
            print(f"  Harts released but heartbeat did not advance (counter={seen[-1]:#x}).")
            print("  Hart may not fetch from scratch (try DRAM-resident code) or clock issue. Harts are now")
            print("  OUT OF RESET (irreversible) — tt-smi -r 0 to retry a different bootstrap.")
        print("=" * 60)
        return 0

    except Hang as e:
        print(f"\n!!! STOP: {e}\n!!! Possible wedge — recover with: tt-smi -r 0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
