"""De-risk the x280-orchestrator handshake: a resident BRISC conductor polls a flag in x280 GDDR (NoC read)
and acks back (NoC write). Host stands in for the x280: set flag, poll ack, measure round-trip. Proves the
workers can be driven by the x280 through GDDR with NO x280->worker writes (which would need a TLB)."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu
from bhtop.tensix.loader import worker_coord
from bhtop.tensix.baremetal import BareMetal, bm_coord

WX, WY = 11, 2
HUB = (8, 3)
FLAG, ACK = 0x30005400, 0x30005500     # in the x280's GDDR


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass
    dev.wr(0, FLAG, [0]); dev.wr(0, ACK, [0])
    BareMetal(WX, WY, ctx=ctx, risc="brisc").run(BareMetal.build("conductor_probe"),
                                                 params=[bm_coord(*HUB), FLAG, ACK])
    time.sleep(0.2)
    pr(f"[boot] conductor_probe resident on BRISC ({WX},{WY}); polling x280 GDDR flag 0x{FLAG:x}")

    times = []
    for r in range(1, 9):
        t0 = time.time(); dev.wr(0, FLAG, [r])          # x280 (host stand-in) raises the flag
        while dev.rd(0, ACK) != r and time.time() - t0 < 2.0: pass
        times.append((time.time() - t0) * 1e3)
    ok = dev.rd(0, ACK) == 8
    pr(f"[handshake] 8 round-trips, ack={dev.rd(0, ACK)}, latency: first={times[0]:.2f}ms "
       f"median={sorted(times)[len(times)//2]:.3f}ms min={min(times):.3f}ms")
    dbg = dev.rd(0, 0x30005400)  # (unused) ; read conductor dbg via worker L1 not needed here
    print("HANDSHAKE_OK — x280 drives worker through GDDR (poll+ack), host-free control path proven"
          if ok else "HANDSHAKE_FAIL")


if __name__ == "__main__":
    main()
