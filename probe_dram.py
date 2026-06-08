"""Read-only DRAM-core counter probe (runs alongside a live bhtop-inject).

Goal: under live injection, see which counters actually move:
  #2 NIU/router flits  (source t2,2 inject  + DRAM dest RX_SLAVE)  -- NoC->L1
  #1 baby RISC-V perf counters on a DRAM slice (INSTRN / L1 / WALL_CLOCK)
  #3 gddr_mc_regs off-chip MC counters (expect ~0 movers = disabled)

SAFE: only tensix/dram tiles; never reads gddr_xbar2(0xFC303000)/gddr_phy(0xFC400000).
"""
import time
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device
from bhtop import floorplan
from bhtop import noc_counters as nc

MASTER_OUT = [8, 9]        # 0x200: master write-out  (source-out bytes)
SLAVE_IN   = nc.RX_SLAVE_IN  # [56,57]: write-data flits landed (sink-in)

ctx = init_ttexalens()
dev = ctx.devices[0]
fp = floorplan.build(ctx)


def niu(t, off, n, noc=0):
    base = 0xFFB20000 if noc == 0 else 0xFFB30000
    return read_words_from_device(t.coord, base + off, word_count=n, noc_id=noc,
                                  context=ctx, safe_mode=False)


def counts(t, noc=0):
    w = niu(t, 0x200, 64, noc)          # 64 words so [56,57] are in range
    return sum(w[i] for i in MASTER_OUT), sum(w[i] for i in SLAVE_IN)


# scan all SAFE endpoints; the hotspot ranks itself (no label guessing)
watch = [t for t in fp.placed if t.kind in ("tensix", "dram")]
base = {t.noc0: (counts(t, 0), counts(t, 1)) for t in watch}
time.sleep(0.6)
print("#2 NIU flits over ~0.6s  (out = master write-out, in = slave landed):")
rows = []
for t in watch:
    (o0a, i0a), (o1a, i1a) = base[t.noc0]
    o0, i0 = counts(t, 0); o1, i1 = counts(t, 1)
    out = ((o0 - o0a) & 0xFFFFFFFF) + ((o1 - o1a) & 0xFFFFFFFF)
    inn = ((i0 - i0a) & 0xFFFFFFFF) + ((i1 - i1a) & 0xFFFFFFFF)
    rows.append((out + inn, out, inn, t))
rows.sort(key=lambda r: r[0], reverse=True)
for tot, out, inn, t in rows[:10]:
    tag = f"  ctrl d{t.dram_ctrl}" if t.kind == "dram" else f"  {t.kind} noc0={t.noc0}"
    print(f"  {t.label:8} out={out*64/1e6:8.1f}MB  in={inn*64/1e6:8.1f}MB{tag}")

# busiest DRAM slice that is actually landing data
dram_rows = [r for r in rows if r[3].kind == "dram" and r[2] > 0]
if not dram_rows:
    print("\n(no DRAM slice showing landed flits this window)")
hot = (dram_rows[0][3] if dram_rows else
       [t for t in fp.placed if t.kind == "dram"][0])
print(f"\nDeep-probe DRAM slice: {hot.label} (ctrl d{hot.dram_ctrl}) coord={hot.coord}")

# ---- #1 baby RISC-V perf counters on the hot DRAM slice -------------------
rs = dev.get_block(hot.coord).get_register_store(0)
perf = ["RISCV_DEBUG_REG_WALL_CLOCK_0",
        "RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0",
        "RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1",
        "RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2",
        "RISCV_DEBUG_REG_PERF_CNT_L1_0",
        "RISCV_DEBUG_REG_PERF_CNT_L1_1",
        "RISCV_DEBUG_REG_PERF_CNT_L1_2",
        "RISCV_DEBUG_REG_PERF_CNT_ALL"]


def snap():
    o = {}
    for n in perf:
        try:
            o[n] = rs.read_register(n)
        except Exception as e:
            o[n] = f"ERR:{type(e).__name__}"
    return o


p0 = snap(); time.sleep(0.5); p1 = snap()
print("\n#1 baby RISC-V (drisc) perf counters  [before -> after 0.5s]:")
for n in perf:
    d = ""
    if isinstance(p0[n], int) and isinstance(p1[n], int):
        d = f"   d={(p1[n]-p0[n]) & 0xFFFFFFFF}"
    print(f"  {n:42} {str(p0[n]):>12} -> {str(p1[n]):>12}{d}")

for n in ["RISC_CTRL_REG_RESET_PC_0", "RISCV_DEBUG_REG_SOFT_RESET_0"]:
    try:
        print(f"  {n:42} = {hex(rs.read_register(n))}")
    except Exception as e:
        print(f"  {n:42} = ERR:{type(e).__name__}")

# ---- #3 gddr_mc_regs movers under load (first 4KB, twice) -----------------
def mc(t, words=1024):
    return read_words_from_device(t.coord, 0xFC100000, word_count=words, noc_id=0,
                                  context=ctx, safe_mode=False)


try:
    m0 = mc(hot); time.sleep(0.5); m1 = mc(hot)
    movers = [(i, m0[i], m1[i]) for i in range(len(m0)) if m0[i] != m1[i]]
    print(f"\n#3 gddr_mc_regs (0xFC100000, first {len(m0)} words) movers under load: {len(movers)}")
    for i, a, b in movers[:12]:
        print(f"    +0x{i*4:04x}: {a:#010x} -> {b:#010x}  (d={(b-a) & 0xFFFFFFFF})")
except Exception as e:
    print(f"\n#3 gddr_mc_regs read ERR: {type(e).__name__}: {e}")

print("\nprobe done (read-only, no mgmt tiles, no unsafe regions).")
