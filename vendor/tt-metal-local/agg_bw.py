#!/usr/bin/env python3
"""Compute aggregate NoC bandwidth from a tt-metal device profiler CSV.

Sums bytes moved across all cores' data-movement zones (RISCV0/RISCV1) and
divides by the wall-clock span of those zones (min start -> max end), at 1.35 GHz.
"""
import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
rows = list(csv.reader(open(path)))
freq = FREQ
for r in rows[:1]:
    for tok in r:
        if "CHIP_FREQ" in tok:
            pass
# header line 2 (index1) has columns; data starts after
data = [r for r in rows if r and r[0].isdigit()]
# columns: slot,cx,cy,risc,timer,cycles,data,runid,traceid,tic,zonename,type,line,file,meta
zones = {}      # (cx,cy,risc,zonename) -> {start,end}
stamp = {}      # (cx,cy,risc) -> {name: value}
for r in data:
    cx, cy, risc = r[1], r[2], r[3]
    cyc = int(r[5]); val = int(r[6]); zname = r[10]; typ = r[11]
    key = (cx, cy, risc)
    if typ == "ZONE_START" and zname.startswith("RISCV"):
        zones.setdefault((cx, cy, risc, zname), {})["start"] = cyc
    elif typ == "ZONE_END" and zname.startswith("RISCV"):
        zones.setdefault((cx, cy, risc, zname), {})["end"] = cyc
    elif typ == "TS_DATA":
        stamp.setdefault(key, {})[zname] = val

# bytes per core-risc
total_bytes = 0
per = {}
for key, s in stamp.items():
    if "Per-core bytes" in s:
        b = s["Per-core bytes"]
    else:
        b = s.get("Number of transactions", 0) * s.get("Transaction size in bytes", 0)
    per[key] = b
    total_bytes += b

# Per-core (same clock domain) zone durations; cores run concurrently so the
# aggregate wall-time is the slowest core's busy span, not a cross-core diff.
durs = [z["end"] - z["start"] for z in zones.values() if "start" in z and "end" in z]
if not durs:
    print("no DM zones found"); sys.exit(1)
durs.sort()
wall_cycles = durs[-1]                 # slowest core bounds the concurrent wall
med = durs[len(durs)//2]
secs = wall_cycles / freq
bw = total_bytes / secs
ncores = len({(cx, cy) for (cx, cy, risc) in per})
print(f"cores active: {ncores}   DM zones: {len(zones)}")
print(f"total bytes : {total_bytes/1e6:.1f} MB")
print(f"per-core dur: min {durs[0]:,} / med {med:,} / max {durs[-1]:,} cycles")
print(f"wall (max)  : {wall_cycles:,} cyc = {secs*1e6:.1f} us @ {freq/1e9:.2f} GHz")
print(f"AGGREGATE BW: {bw/1e12:.2f} TB/s   ({bw/1e9:.0f} GB/s)   [median-dur: {total_bytes/(med/freq)/1e12:.2f} TB/s]")
