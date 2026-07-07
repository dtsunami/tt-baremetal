"""De-risk the RESIDENT cb_io: boot once on a worker's BRISC, then ring it via a doorbell (no reload) to
NoC-read operands from x280 GDDR into L1. Verify it fires, lands the data, and re-fires on the next ring
(proving residency). This replaces the 1.8s-per-step BareMetal.run reload."""
import sys, struct, time
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu
from bhtop.tensix.loader import worker_coord
from bhtop.tensix.baremetal import BareMetal, bm_coord

WX, WY = 11, 2
IO_DB, IO_DONE, IO_CFG = 0x3000, 0x3010, 0x3020
H_psi = 0x22000
OPBASE = 0x30080000
HUB = (8, 3)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); coord = worker_coord(ctx, WX, WY); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass

    # zero the doorbell region, then boot cb_io resident on BRISC
    wr(coord, IO_DB, [0, 0, 0, 0, 0, 0, 0, 0], context=ctx)
    bm = BareMetal(WX, WY, ctx=ctx, risc="brisc")
    bm.run(BareMetal.build("cb_io"))                      # boots into the doorbell loop (resident)
    time.sleep(0.2)
    pr(f"[boot] cb_io resident on BRISC ({WX},{WY})")

    def ring_read(pattern_word):
        # stage 6 operand tiles in x280 GDDR with a recognizable pattern
        for i in range(6):
            dev.wr(0, OPBASE + i * 0x800, [(pattern_word + i) & 0xFFFFFFFF] * 512)
        wr(coord, H_psi, [0xBADF00D5] * 512, context=ctx)  # poison worker L1 psi
        wr(coord, IO_CFG, [bm_coord(*HUB), OPBASE, 0], context=ctx)   # cfg: coord, base, mode=read
        r = rd(coord, IO_DONE, context=ctx) + 1
        t0 = time.time(); wr(coord, IO_DB, [r], context=ctx)
        while time.time() - t0 < 2.0 and rd(coord, IO_DONE, context=ctx) != r: time.sleep(0.001)
        ms = (time.time() - t0) * 1e3
        got = [w & 0xFFFFFFFF for w in rds(coord, H_psi, word_count=512, context=ctx)]
        ok = all(g == (pattern_word & 0xFFFFFFFF) for g in got)      # psi tile = OPBASE+0 pattern
        return ok, ms, rd(coord, IO_DONE, context=ctx)

    ok1, ms1, d1 = ring_read(0x11110000)
    ok2, ms2, d2 = ring_read(0x22220000)                  # re-ring with new data (no reload)
    pr(f"[ring 1] read {'OK' if ok1 else 'FAIL'}  done={d1}  {ms1:.2f} ms")
    pr(f"[ring 2] read {'OK' if ok2 else 'FAIL'}  done={d2}  {ms2:.2f} ms  (resident re-fire)")
    print("CB_IO_RESIDENT_OK — doorbell-driven NoC read, no reload" if (ok1 and ok2 and d2 == 2)
          else "CB_IO_RESIDENT_FAIL")


if __name__ == "__main__":
    main()
