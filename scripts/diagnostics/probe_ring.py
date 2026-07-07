"""Isolate the render-ring mechanism. Operands can be garbage — we only watch whether all 3 render threads
WAKE (HB bumps, DBG_M advances). Four snapshots:
  (a) HOST NoC-write ring, NO conductor         -> baseline (test_het_loop proves this works)
  (b) HOST NoC-write ring, idle conductor present -> does the conductor's mere PRESENCE wedge a thread?
  (c) CONDUCTOR local-store ring (flag=1)        -> does math wake on a LOCAL-store ring?
DBG_M=0 => math never entered its stage loop (stuck at wait_ring). HB bumps => a full ring completed."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu
from bhtop.tensix.loader import worker_coord
from bhtop.tensix import llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord

WX, WY, HUB = 11, 2, (8, 3)
FLAG, ACK = 0x30005C00, 0x30005D00
OPBASE, GINORCH = 0x30080000, 0x300C0000
PHI_ST, PHI2T_ST, GT_ST = 0x8000, 0xC000, 0x10000
CFG = 0x3200

ctx = init_ttexalens(); coord = worker_coord(ctx, WX, WY); dev = L2cpu(ctx=ctx)
try: dev.bringup(0)
except Exception: pass
R = lambda a: rds(coord, a, word_count=1, context=ctx)[0]
def snap(tag):
    print(f"  [{tag:38s}] HB={R(0x16020)} DONE={R(0x16010)} DB={R(0x16000)} | "
          f"DBG_U={R(0x16030):#06x} DBG_M={R(0x16040):#06x} DBG_P={R(0x16050):#06x}", flush=True)
def host_ring():
    r = R(0x16010) + 1; wr(coord, 0x16000, [r], context=ctx); time.sleep(0.15)

llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
              formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
boot_resident("resident_train_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
time.sleep(0.3)
snap("render booted, no ring")

# (a) HOST ring, no conductor
host_ring(); snap("after HOST ring (no conductor)")
host_ring(); snap("after 2nd HOST ring (no conductor)")

# (b) idle conductor present, host ring
dev.wr(0, FLAG, [0]); dev.wr(0, ACK, [0])
wr(coord, CFG, [bm_coord(*HUB), 0, OPBASE, GINORCH, FLAG, ACK, PHI_ST, PHI2T_ST, GT_ST, 8], context=ctx)
BareMetal(WX, WY, ctx=ctx, risc="brisc").run(BareMetal.build("conductor")); time.sleep(0.2)
snap("idle conductor booted")
host_ring(); snap("after HOST ring (idle conductor present)")

# (c) conductor's own LOCAL-store ring
dev.wr(0, FLAG, [1]); time.sleep(0.4)
snap("after CONDUCTOR local-store ring")
print(f"  conductor dbg(flag,last,nflag,g,rdone,ring,ack,DBwr)={list(rds(coord, 0x2100, word_count=8, context=ctx))}", flush=True)
