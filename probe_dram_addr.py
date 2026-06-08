"""Resolve L1-vs-off-chip and test address patterns that fill / don't fill GDDR.

Map (BlackholeDramBlock): dram_bank = 4 GiB @ noc 0x0 ; l1 = 128 KB @ noc 0x2000000000.
=> a NoC write to 0x10000 targets OFF-CHIP dram_bank, not L1.

Decisive proof: write at a 1 GiB offset (can't alias a 128 KB L1) and read it back.
All writes save->write->read->restore. Safe: dram tiles only, never gddr_xbar2/gddr_phy.
"""
import time
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device
from bhtop import floorplan

ctx = init_ttexalens()
dev = ctx.devices[0]
fp = floorplan.build(ctx)

L1_NOC = 0x2000000000
PAT = [0xDEADBE00, 0xCAFE0001, 0x5A5A5A5A, 0x0BADF00D]


def rd(t, addr, n):
    return read_words_from_device(t.coord, addr, word_count=n, noc_id=0, context=ctx, safe_mode=False)


def wr(t, addr, vals):
    write_words_to_device(t.coord, addr, vals, noc_id=0, context=ctx, safe_mode=False)


def rb_test(t, addr):
    """save->write PAT->read->restore; return (ok, readback)."""
    orig = rd(t, addr, 4)
    wr(t, addr, PAT)
    back = rd(t, addr, 4)
    wr(t, addr, orig)
    return back == PAT, back


# --- find a working off-chip write tile per controller (decisive 1 GiB offset) ---
print("read-back @ 0x40000000 (1 GiB -- far beyond 128 KB L1 => real off-chip if MATCH):")
working = None
for ctrl in sorted(fp.dram_ctrl):
    for t in fp.dram_ctrl[ctrl]:
        try:
            ok, back = rb_test(t, 0x40000000)
            print(f"  d{ctrl} {t.label:6} {'MATCH' if ok else 'mismatch'}  {[hex(x) for x in back]}")
            if ok and working is None:
                working = t
        except Exception as e:
            print(f"  d{ctrl} {t.label:6} ERR {type(e).__name__}")
    if working and ctrl >= 2:
        break

if working is None:
    print("\nno off-chip write tile found; aborting.")
    raise SystemExit

t = working
print(f"\nusing tile {t.label} (ctrl d{t.dram_ctrl}) for address-pattern tests\n")

# --- address-pattern matrix: which addresses 'fill DRAM' vs stay in L1 ---
targets = {
    "off-chip 0x00010000 (inject tgt)": 0x00010000,
    "off-chip 0x00100000 (1 MiB)":      0x00100000,
    "off-chip 0x40000000 (1 GiB)":      0x40000000,
    "off-chip 0x80000000 (2 GiB)":      0x80000000,
    "L1 slice 0x2000001000":            L1_NOC + 0x1000,
}
for name, addr in targets.items():
    try:
        ok, back = rb_test(t, addr)
        print(f"  {name:34} {'persists' if ok else 'NO/aliased':10} back={[hex(x) for x in back]}")
    except Exception as e:
        print(f"  {name:34} ERR {type(e).__name__}: {e}")

# --- alias check: distinct values at 1 MiB and 1 GiB simultaneously ---
a1, a2 = 0x00100000, 0x40000000
o1, o2 = rd(t, a1, 2), rd(t, a2, 2)
wr(t, a1, [0x11111111, 0x22222222]); wr(t, a2, [0x33333333, 0x44444444])
r1, r2 = rd(t, a1, 2), rd(t, a2, 2)
wr(t, a1, o1); wr(t, a2, o2)
print(f"\nalias check: @1MiB={[hex(x) for x in r1]}  @1GiB={[hex(x) for x in r2]}  "
      f"=> {'DISTINCT (real multi-GiB DRAM)' if r1 != r2 else 'ALIASED (small buffer = L1)'}")

# --- gddr_mc full 64 KB scan under Dan's live off-chip injection ---
def mc(words):
    return read_words_from_device(t.coord, 0xFC100000, word_count=words, noc_id=0, context=ctx, safe_mode=False)


W = 0x10000 // 4
m0 = mc(W); time.sleep(0.6); m1 = mc(W)
movers = [(i, m0[i], m1[i]) for i in range(W) if m0[i] != m1[i]]
print(f"\ngddr_mc full 64 KB scan under live inject: {len(movers)} movers"
      f"  ({'counters LIVE' if movers else 'counters DISABLED'})")
for i, a, b in movers[:16]:
    print(f"   +0x{i*4:04x}: {a:#010x} -> {b:#010x}  d={(b - a) & 0xFFFFFFFF}")
print("\ndone (read-only except save/restore'd scratch writes).")
