"""
Blackhole NoC NIU counter decode.

Single source of truth for the per-tile NIU hardware performance counters.
Reference: tenstorrent/tt-isa-documentation, BlackholeA0/NoC/{Counters,MemoryMap}.md

Each tile (Tensix / DRAM / Ethernet) contains TWO NIUs (NoC interface units):
  NIU #0 -> NoC0 router, registers based at 0xFFB2_0000
  NIU #1 -> NoC1 router, registers based at 0xFFB3_0000

Each NIU exposes a 62-entry array of 32-bit counters at NIU_BASE + 0x0200.
A NoC "data word" is one 512-bit (64-byte) flit, so:  bytes = count * 64.

Counters are free-running and wrap at 2**32; bandwidth is derived from the
delta between two samples over wall-clock dt.
"""

FLIT_BYTES = 64                  # 512-bit NoC flit
COUNTER_MASK = 0xFFFFFFFF        # 32-bit free-running counters
COUNTER_ARRAY_OFF = 0x0200       # NIU_BASE + 0x200
COUNTER_ARRAY_LEN = 62           # words to read in one burst

# NIU register bases (Tensix / Ethernet tiles; DRAM tiles mirror the low 32 bits)
NIU_BASE = {
    0: 0xFFB20000,   # NoC0
    1: 0xFFB30000,   # NoC1
}

def counter_base(noc_id: int) -> int:
    return NIU_BASE[noc_id] + COUNTER_ARRAY_OFF

# index -> canonical name (from Counters.md)
COUNTERS = {
    0:  "MST_ATOMIC_RESP_RECEIVED",
    1:  "MST_WR_ACK_RECEIVED",
    2:  "MST_RD_RESP_RECEIVED",
    3:  "MST_RD_DATA_WORD_RECEIVED",          # read data flits pulled IN  (this tile = requester)
    4:  "MST_CMD_ACCEPTED",
    5:  "MST_RD_REQ_SENT",
    6:  "MST_NONPOSTED_ATOMIC_SENT",
    7:  "MST_POSTED_ATOMIC_SENT",
    8:  "MST_NONPOSTED_WR_DATA_WORD_SENT",    # write data flits pushed OUT (nonposted)
    9:  "MST_POSTED_WR_DATA_WORD_SENT",       # write data flits pushed OUT (posted)
    10: "MST_NONPOSTED_WR_REQ_SENT",
    11: "MST_POSTED_WR_REQ_SENT",
    12: "MST_NONPOSTED_WR_REQ_STARTED",
    13: "MST_POSTED_WR_REQ_STARTED",
    14: "MST_RD_REQ_STARTED",
    15: "MST_NONPOSTED_ATOMIC_STARTED",
    # 16..31 NIU_MST_REQS_OUTSTANDING_ID(0..15)      - 8-bit, in-flight depth
    # 32..47 NIU_MST_WRITE_REQS_OUTGOING_ID(0..15)   - 8-bit, write drain depth
    48: "SLV_ATOMIC_RESP_SENT",
    49: "SLV_WR_ACK_SENT",
    50: "SLV_RD_RESP_SENT",
    51: "SLV_RD_DATA_WORD_SENT",              # read data flits SERVED out (this tile = target, e.g. DRAM)
    52: "SLV_REQ_ACCEPTED",
    53: "SLV_RD_REQ_RECEIVED",
    54: "SLV_NONPOSTED_ATOMIC_RECEIVED",
    55: "SLV_POSTED_ATOMIC_RECEIVED",
    56: "SLV_NONPOSTED_WR_DATA_WORD_RECEIVED",# write data flits LANDED here (nonposted)
    57: "SLV_POSTED_WR_DATA_WORD_RECEIVED",   # write data flits LANDED here (posted)
    58: "SLV_NONPOSTED_WR_REQ_RECEIVED",
    59: "SLV_POSTED_WR_REQ_RECEIVED",
    60: "SLV_NONPOSTED_WR_REQ_STARTED",
    61: "SLV_POSTED_WR_REQ_STARTED",
}
NAME_TO_IDX = {v: k for k, v in COUNTERS.items()}

# The four flit-counting indices that constitute throughput, split by direction.
# "initiated" = traffic this tile originated (master side)
# "served"    = traffic this tile sourced/sank on behalf of others (slave side)
TX_MASTER_OUT = [8, 9]    # write data flits this tile sent out
RX_MASTER_IN  = [3]       # read data flits this tile pulled in
TX_SLAVE_OUT  = [51]      # read data flits this tile served (DRAM reads land here)
RX_SLAVE_IN   = [56, 57]  # write data flits that landed on this tile


def _delta(now: int, prev: int) -> int:
    """32-bit wrap-safe counter delta."""
    return (now - prev) & COUNTER_MASK


def flit_bandwidth(words_now, words_prev, indices, dt: float) -> float:
    """Bytes/sec across the given counter indices, over dt seconds."""
    if dt <= 0:
        return 0.0
    flits = sum(_delta(words_now[i], words_prev[i]) for i in indices)
    return flits * FLIT_BYTES / dt


def tile_bandwidths(words_now, words_prev, dt: float) -> dict:
    """All four directional bandwidths (bytes/s) for one tile+NoC sample pair."""
    return {
        "tx_master": flit_bandwidth(words_now, words_prev, TX_MASTER_OUT, dt),
        "rx_master": flit_bandwidth(words_now, words_prev, RX_MASTER_IN, dt),
        "tx_slave":  flit_bandwidth(words_now, words_prev, TX_SLAVE_OUT, dt),
        "rx_slave":  flit_bandwidth(words_now, words_prev, RX_SLAVE_IN, dt),
    }


METRICS = ["total", "tx", "rx", "master", "slave"]

def metric_scalar(bw: dict, metric: str) -> float:
    """Collapse a 4-direction bandwidth dict to one scalar (bytes/s) for a metric."""
    if not bw:
        return 0.0
    if metric == "total":
        return sum(bw.values())
    if metric == "tx":
        return bw["tx_master"] + bw["tx_slave"]
    if metric == "rx":
        return bw["rx_master"] + bw["rx_slave"]
    if metric == "master":
        return bw["tx_master"] + bw["rx_master"]
    if metric == "slave":
        return bw["tx_slave"] + bw["rx_slave"]
    return 0.0
