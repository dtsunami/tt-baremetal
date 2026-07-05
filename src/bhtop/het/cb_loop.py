"""Het DRAM circular buffer (M2): x280 PRODUCER ⇄ Tensix CONSUMER through a GDDR ring with
produced/acked backpressure. Both engines run concurrently on one shared exalens ctx; the host only
launches them and reads the result — no data relayed through the host."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.baremetal import BareMetal, bm_coord

T, N = 12, 4
ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)

# init shared state (P, A, PDONE) in uncached x280 GDDR
dev.wr(0, 0x30002000, [0]); dev.wr(0, 0x30002010, [0]); dev.wr(0, 0x30002020, [0])

# launch x280 producer (hart 0)
pw = tc.compile_source("/home/starboy/bhtop/src/bhtop/kernels/x280/het/cb_producer.c", base=CODE_ADDR, march="rv64gc")
for _ in range(6):
    dev.load(0, 0, pw); time.sleep(0.25)
    if dev.rd(0, 0x30002000) > 0 or dev.rd(0, 0x30002020) == 0xD09E: break   # producing or done
print("producer launched; produced so far:", dev.rd(0, 0x30002000), "(fills ring then waits on ack)")

# launch Tensix consumer (worker (1,2))
bm = BareMetal(1, 2, ctx=ctx)
bm.run(bm.build("cb_consumer"), params=[bm_coord(8, 3), T, N])

# wait for consumer done
deadline = time.time() + 8
done = 0
while time.time() < deadline:
    done = bm.result(1)[0]
    if done == (0xC0DE0000 | T): break
    time.sleep(0.1)

sums = bm.rd(0x2100 + 0x40, T)                      # per-item checksums (BM_DBG+0x40)
polls = bm.rd(0x2100, 1)[0]                         # consumer P-polls
pdone = dev.rd(0, 0x30002020)                       # producer done flag
exp = [1600 * i + 120 for i in range(T)]            # sum(i*100+w, w=0..15)
ok = (done == (0xC0DE0000 | T)) and (sums == exp) and (pdone == 0xD09E)
print(f"consumer done marker: {hex(done)} (want {hex(0xC0DE0000|T)})")
print(f"producer done: {hex(pdone)}   consumer P-polls: {polls} (> T ⇒ waited on producer)")
print(f"checksums match expected: {sums == exp}")
print(f"  sums[:4]={sums[:4]}  exp[:4]={exp[:4]}")
print(f"\nCB LOOP {'PASS' if ok else 'FAIL'} — {T} items streamed x280→ring(N={N})→Tensix, "
      f"ring wrapped {T-N}× (backpressure held: producer never overwrote an un-acked slot)")
