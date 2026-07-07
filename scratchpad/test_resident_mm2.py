"""R2 mechanism: RESIDENT inter-stage dataflow — C2 = (A@B) @ D as TWO matmuls in one doorbell ring,
with the stage-1 result staged in L1 and re-consumed by stage 2 (pack->unpack sync on-device). If this
is bit-exact across rings with no reload, the fused render's 11-stage chaining is proven mechanism.

bf16 throughout (intermediate C1 packed bf16 so stage-2 unpacks the same format the matmul reads — the
render passes bf16 between stages too). Small operands keep everything bf16-exact.
"""
import sys, os, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import load_elf, read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run

DB, DONE, HB, DBG_U, DBG_M, DBG_P, S1 = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050, 0x16060
A_ADDR, B_ADDR, C1_ADDR, OUT_ADDR, D_ADDR = 0x21000, 0x31000, 0x41000, 0x51000, 0x61000
POISON = 0xBADF00D5
TH = {"UNPACK": "trisc0", "MATH": "trisc1", "PACK": "trisc2"}


def enc(m):  # tilize + bf16 pack
    return MM.pack_bf16_words([float(x) for x in MM.tilize32(m)])


def operands(seed):
    a = [((i + k + seed) % 3) for i in range(32) for k in range(32)]          # {0,1,2}
    b = [((k * j + 1 + seed) % 3) for k in range(32) for j in range(32)]      # {0,1,2}
    d = [(1 if j == (k + 1) % 32 else 0) for k in range(32) for j in range(32)]  # cyclic permutation
    return a, b, d


def main():
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    b = llk_run.build("resident_mm2_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False, formats=None)
    assert b["ok"], b["log"]
    out = os.path.expanduser("~/bhtop/kernels/tensix/llk/_build/resident_mm2_perf")
    dev = ctx.devices[0]; block = dev.get_block(coord)
    rdbg = {c: block.get_risc_debug(TH[c]) for c in TH}
    for c in TH: rdbg[c].set_reset_signal(True)
    for c in TH: load_elf(elf_file=os.path.join(out, c + ".elf"), location=coord, risc_name=TH[c], device_id=0, context=ctx, verify_write=False)
    wr(coord, 0x20000, [0] * 64, context=ctx)
    wr(coord, 0x20000, [1, 1, 1, 1, 1, 128, 128, 0, 4, 4], context=ctx)   # bf16 tsize=128
    wr(coord, 0x16000, [0] * 32, context=ctx)                             # clear DB..S1
    wr(coord, 0x1FFB8, [0xA3] * 3, context=ctx)
    rdbg["UNPACK"].set_reset_signal(False)
    time.sleep(0.3)

    n_ok = 0
    for ring in range(1, 4):
        a, bb, d = operands(ring)
        wr(coord, A_ADDR, enc(a), context=ctx)
        wr(coord, B_ADDR, enc(bb), context=ctx)
        wr(coord, D_ADDR, enc(d), context=ctx)
        wr(coord, C1_ADDR, [POISON] * 512, context=ctx)
        wr(coord, OUT_ADDR, [POISON] * 512, context=ctx)
        wr(coord, DB, [ring], context=ctx)
        t0 = time.time(); done = None
        while time.time() - t0 < 5.0:
            done = rd(coord, DONE, context=ctx)
            if done == ring: break
            time.sleep(0.005)
        outw = rds(coord, OUT_ADDR, word_count=512, context=ctx)
        c2 = MM.untilize32(MM.unpack_bf16_words(outw))
        gold = MM.matmul_golden(MM.matmul_golden(a, bb), d)
        mism = sum(1 for i in range(1024) if float(gold[i]) != c2[i])
        ok = (done == ring and mism == 0)
        n_ok += ok
        print(f"[mm2] ring {ring}: done={done} hb={rd(coord,HB,context=ctx)} "
              f"U={hex(rd(coord,DBG_U,context=ctx))} M={hex(rd(coord,DBG_M,context=ctx))} "
              f"P={hex(rd(coord,DBG_P,context=ctx))} S1={rd(coord,S1,context=ctx)} | "
              f"C2 vs (A@B)@D mism={mism} corner g={gold[0]} d={c2[0]} -> {'OK' if ok else 'CHECK'}")
    for c in TH: rdbg[c].set_reset_signal(True)
    print(f"\n[mm2] {n_ok}/3 rings bit-exact")
    print(f"[mm2] RESIDENT INTER-STAGE DATAFLOW (chained matmul, on-device L1 restage): "
          f"{'PASS' if n_ok == 3 else 'CHECK'}")
    return n_ok == 3


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
