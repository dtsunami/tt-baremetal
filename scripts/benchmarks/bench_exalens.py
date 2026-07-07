"""Quantify the exalens host<->device transport: time write_words_to_device / read_words_from_device across
sizes to separate PER-CALL latency (fixed overhead/transaction) from PER-WORD bandwidth. This decomposes the
99.9%-host-relay bottleneck: if it's latency-bound, the fix is fewer+bigger transfers / on-device streaming;
if bandwidth-bound, the fix is DMA."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import write_words_to_device as wr, read_words_from_device as rds
from bhtop.tensix.loader import worker_coords

SIZES = [1, 64, 512, 2048, 16384, 65536, 262144]   # words (×4 bytes)
ADDR = 0x100000                                      # L1 scratch on a worker
REPS = 20


def timed(fn, reps):
    best = 1e9
    for _ in range(reps):
        t0 = time.time(); fn(); dt = time.time() - t0
        best = min(best, dt)
    return best


def main():
    ctx = init_ttexalens()
    coord = worker_coords(ctx)[0]
    print(f"[bench] exalens write/read to worker {coord} @0x{ADDR:x}\n")
    print(f"{'words':>8} {'KB':>8} | {'wr ms':>9} {'wr MB/s':>9} | {'rd ms':>9} {'rd MB/s':>9}")
    wr_rows, rd_rows = [], []
    for n in SIZES:
        payload = list(range(n))
        reps = max(3, REPS if n <= 16384 else 3)
        wms = timed(lambda: wr(coord, ADDR, payload, context=ctx), reps) * 1e3
        rms = timed(lambda: rds(coord, ADDR, word_count=n, context=ctx), reps) * 1e3
        kb = n * 4 / 1024
        wbw = (n * 4 / 1e6) / (wms / 1e3) if wms else 0
        rbw = (n * 4 / 1e6) / (rms / 1e3) if rms else 0
        print(f"{n:>8} {kb:>8.1f} | {wms:>9.3f} {wbw:>9.1f} | {rms:>9.3f} {rbw:>9.1f}")
        wr_rows.append((n, wms)); rd_rows.append((n, rms))

    # linear decompose: t = latency + words*per_word  (use smallest vs largest)
    def decomp(rows):
        (n0, t0), (n1, t1) = rows[0], rows[-1]
        per_word = (t1 - t0) / (n1 - n0)            # ms/word
        latency = t0 - n0 * per_word                # ms fixed
        bw = 4 / (per_word / 1e3) / 1e6 if per_word > 0 else 0   # MB/s asymptotic
        return latency, bw
    wl, wbw = decomp(wr_rows); rl, rbw = decomp(rd_rows)
    print(f"\n[decomp] WRITE per-call latency = {wl:.3f} ms  |  asymptotic BW = {wbw:.0f} MB/s")
    print(f"[decomp] READ  per-call latency = {rl:.3f} ms  |  asymptotic BW = {rbw:.0f} MB/s")
    # what the flow actually does: 256px render_readback = 10240 reads of 512 words
    per_read_512 = next(t for n, t in rd_rows if n == 512)
    print(f"\n[flow] a 512-word read costs {per_read_512:.3f} ms; the 256px step did 10,240 such reads = "
          f"{per_read_512*10240/1e3:.1f} s of readback alone (measured 10.3 s). Batching 5 reads/ring into 1, "
          f"and all tiles into few big transfers, collapses the per-call-latency tax.")


if __name__ == "__main__":
    main()
