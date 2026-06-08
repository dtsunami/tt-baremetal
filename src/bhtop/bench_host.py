#!/usr/bin/env python3
"""
Host-driven NoC microbenchmark + counter validation.

Pushes a known volume of bytes over a chosen NoC to a target tile's L1, then
checks that the tile's NIU write-data counter advanced by exactly bytes/64
flits. Same for reads against SLV_RD_DATA_WORD_SENT. This validates the entire
counter -> bandwidth pipeline against ground truth, with no tt-metal needed.

(Traffic here is host->PCIe-tile->target. A Tensix-initiated kernel benchmark
is the deeper follow-up once tt-metal is built.)
"""
import argparse
import time

from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import (
    read_words_from_device, write_to_device,
)
from . import noc_counters as nc

L1_ADDR = 0x40000          # safe scratch offset in Tensix L1


def read_counters(loc, noc_id, ctx):
    return read_words_from_device(loc, nc.counter_base(noc_id),
                                  word_count=nc.COUNTER_ARRAY_LEN, noc_id=noc_id, context=ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="2,2", help="target tensix tile (default 2,2)")
    ap.add_argument("--noc", type=int, default=None, choices=(0, 1),
                    help="NoC to test (default: both, with a per-NoC comparison)")
    ap.add_argument("--kb", type=int, default=64, help="bytes per transfer (KiB)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--bulk", action="store_true",
                    help="use burst (64B-flit) access instead of 4B-mode")
    args = ap.parse_args()
    use_4B = not args.bulk

    ctx = init_ttexalens()
    loc = args.tile
    nbytes = args.kb * 1024
    payload = bytes(nbytes)                      # zeros; only volume matters
    payload_total = nbytes * args.iters
    nocs = (0, 1) if args.noc is None else (args.noc,)

    WR = nc.RX_SLAVE_IN          # [56,57] write-data flits landed at target
    RD = nc.TX_SLAVE_OUT         # [51]    read-data flits served by target

    def report(tag, noc, flits, payload_total, secs, direction):
        wire = flits * nc.FLIT_BYTES                 # link occupancy (full flit slots)
        eff = payload_total / wire if wire else 0     # flit packing efficiency
        print(f"=== {tag}  tile {loc}  NoC{noc}  {args.kb} KiB x {args.iters}  "
              f"({'bulk' if args.bulk else '4B-mode'}) ===")
        print(f"  payload moved      : {payload_total/1e6:10.2f} MB")
        print(f"  wire (flits x 64)  : {wire/1e6:10.2f} MB  ({flits} flits)")
        print(f"  flit efficiency    : {eff*100:6.1f}%   ({payload_total/flits:.1f} payload B/flit)")
        print(f"  {direction:18s}: {wire/secs/1e9:6.2f} GB/s wire  "
              f"({payload_total/secs/1e9:.2f} GB/s payload, {secs:.2f}s)\n")
        return wire / secs if secs else 0.0          # wire GB/s for the comparison

    def run_noc(noc):
        # ---- WRITE test ----
        c0 = read_counters(loc, noc, ctx)
        t0 = time.monotonic()
        for _ in range(args.iters):
            write_to_device(loc, L1_ADDR, payload, noc_id=noc, context=ctx, use_4B_mode=use_4B)
        t1 = time.monotonic()
        c1 = read_counters(loc, noc, ctx)
        wr_flits = sum((c1[i] - c0[i]) & nc.COUNTER_MASK for i in WR)
        wr_bw = report("WRITE", noc, wr_flits, payload_total, t1 - t0, "host->NoC write")

        # ---- READ test ----
        c0 = read_counters(loc, noc, ctx)
        t0 = time.monotonic()
        for _ in range(args.iters):
            read_words_from_device(loc, L1_ADDR, word_count=nbytes // 4,
                                   noc_id=noc, context=ctx, use_4B_mode=use_4B)
        t1 = time.monotonic()
        c1 = read_counters(loc, noc, ctx)
        rd_flits = sum((c1[i] - c0[i]) & nc.COUNTER_MASK for i in RD)
        rd_bw = report("READ", noc, rd_flits, payload_total, t1 - t0, "NoC->host read")
        return wr_bw, rd_bw

    results = {noc: run_noc(noc) for noc in nocs}

    if len(nocs) > 1:
        print(f"=== BW over NoCs  tile {loc}  ({'bulk' if args.bulk else '4B-mode'}, wire GB/s) ===")
        print(f"  {'':6} {'NoC0':>10} {'NoC1':>10}")
        print(f"  {'write':6} {results[0][0]:>9.2f}  {results[1][0]:>9.2f}")
        print(f"  {'read':6} {results[0][1]:>9.2f}  {results[1][1]:>9.2f}")
        print("  (host-driven point transfer; per-link saturation needs on-chip kernels)")


if __name__ == "__main__":
    main()
