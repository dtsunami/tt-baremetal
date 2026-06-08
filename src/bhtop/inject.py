"""
NoC traffic injection + path tracing for Blackhole (reverse-engineering engine).

Programs an arbitrary tile's NoC0 NIU request-initiator registers from the host so
that *that tile* sources a posted write to a destination tile, then reads the router
port counters (0x500) to project the route the NoC actually chose.

SAFETY (learned the hard way -- see bh-noc-hang-hazard memory):
  * Only ever touch Tensix / DRAM / Eth tiles. Touching ARC/Security/PCIe/L2CPU
    register space hangs NoC0 (recover with `tt-smi -r 0`).
  * Single large auto-split transfers (one fire) keep observer traffic negligible.

Decoded NoC0 router port words (0x500, 8 ports x 8 words / 4 VCs):
  words 0-6   = South transit     words 32-38 = East transit
  words 44-46 = local eject       words 60-62 = local inject
NoC0 routes east+south only, so a route is fully captured by East+South transit.
"""
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from . import noc_counters as nc

NIU0 = 0xFFB20000
NIU = {0: 0xFFB20000, 1: 0xFFB30000}     # NoC0 / NoC1 NIU register bases
COUNTERS = NIU0 + 0x200
ROUTER = NIU0 + 0x500

# request-initiator #0 register offsets (BlackholeA0/NoC/MemoryMap.md)
TARG_LO, TARG_MID, TARG_HI = 0x00, 0x04, 0x08
RET_LO, RET_MID, RET_HI = 0x0C, 0x10, 0x14
PKT_TAG, CTRL, AT_LEN_BE, CMD_CTRL = 0x18, 0x1C, 0x20, 0x40
NOC_CMD_WR = 0x2

SAFE_KINDS = {"tensix", "dram", "eth"}
L1_SIZE = 0x180000            # 1.5 MiB Tensix L1
SRC_ADDR, DST_ADDR = 0x20000, 0x10000
MAX_LEN = L1_SIZE - SRC_ADDR  # keep source read inside L1 (else initiators wedge!)
TRANSIT_WORDS = [0, 2, 4, 6, 32, 34, 36, 38]   # East + South transit (NoC0)
INJECT_WORDS = [60, 62]                          # local NIU inject (marks source)


def _hi(x, y):
    return (x & 0x3F) | ((y & 0x3F) << 6)


class HangError(RuntimeError):
    pass


class Injector:
    def __init__(self, fp, ctx):
        self.fp = fp
        self.ctx = ctx
        self.cells, self.cols, self.rows = fp.grid("noc0")
        # only tensix routers are scanned for transit (safest, and the fabric we route through)
        self.scan = [t for t in fp.placed if t.kind == "tensix"]

    def _rd(self, tile, addr, n):
        return read_words_from_device(tile.coord, addr, word_count=n, noc_id=0,
                                      context=self.ctx, safe_mode=False)

    def _w(self, tile, off, val):
        write_words_to_device(tile.coord, NIU0 + off, [val], noc_id=0,
                              context=self.ctx, safe_mode=False)

    def _transit(self, tile):
        try:
            w = self._rd(tile, ROUTER, 64)
        except Exception:
            return None
        return sum(w[i] for i in TRANSIT_WORDS)

    def inject_and_trace(self, src, dst, length=0x80000, fires=8):
        """Posted-write src->dst; return (route:{tile.key->transit}, src_bw, dst_bw)."""
        if src.kind not in SAFE_KINDS or dst.kind not in SAFE_KINDS:
            raise ValueError("src/dst must be tensix/dram/eth (others hang the NoC)")
        length = min(length, MAX_LEN)        # keep source read inside L1
        sx, sy = src.noc0
        dx, dy = dst.noc0
        # program src initiator: posted write, data A:0x20000 -> B:0x10000
        for off, val in [(TARG_LO, 0x20000), (TARG_MID, 0), (TARG_HI, _hi(sx, sy)),
                         (RET_LO, 0x10000), (RET_MID, 0), (RET_HI, _hi(dx, dy)),
                         (PKT_TAG, 0), (CTRL, NOC_CMD_WR), (AT_LEN_BE, length)]:
            self._w(src, off, val)

        base = {t.key: self._transit(t) for t in self.scan}
        s0 = self._rd(src, COUNTERS, 62)
        d0 = self._rd(dst, COUNTERS, 62)
        for _ in range(fires):
            self._w(src, CMD_CTRL, 1)
            for _ in range(20000):
                if self._rd(src, NIU0 + CMD_CTRL, 1)[0] == 0:
                    break
        s1 = self._rd(src, COUNTERS, 62)
        d1 = self._rd(dst, COUNTERS, 62)

        route = {}
        for t in self.scan:
            now = self._transit(t)
            b = base[t.key]
            route[t.key] = None if (now is None or b is None) else (now - b) & 0xFFFFFFFF
        # endpoint bandwidth (bytes): src master write-out, dst slave write-in
        sent = (((s1[8] - s0[8]) & 0xFFFFFFFF) + ((s1[9] - s0[9]) & 0xFFFFFFFF)) * 64
        landed = (((d1[56] - d0[56]) & 0xFFFFFFFF) + ((d1[57] - d0[57]) & 0xFFFFFFFF)) * 64
        return route, sent, landed

    # ---- dual-NoC register access -----------------------------------------
    def _wn(self, tile, off, val, noc):
        write_words_to_device(tile.coord, NIU[noc] + off, [val], noc_id=noc,
                              context=self.ctx, safe_mode=False)

    def _rdn(self, tile, off, n, noc):
        return read_words_from_device(tile.coord, NIU[noc] + off, word_count=n,
                                      noc_id=noc, context=self.ctx, safe_mode=False)

    def _pick_noc(self, src, dst):
        """Pick the NoC that reaches dst with fewer forced hops. NoC0 only goes
        east+south, NoC1 only west+north (both on a torus), so the shorter-routing
        NoC depends on the destination's direction."""
        W, H = self.cols, self.rows
        sx, sy = src.noc0; dx, dy = dst.noc0
        es = (dx - sx) % W + (dy - sy) % H      # NoC0: east + south
        wn = (sx - dx) % W + (sy - dy) % H      # NoC1: west + north
        return 0 if es <= wn else 1

    def _route(self, src, dst, noc):
        """Deterministic dimension-ordered path (noc0 coords, torus-wrapped): NoC0
        goes east-then-south, NoC1 north-then-west. Returns the traversed tiles."""
        W, H = self.cols, self.rows
        sx, sy = src.noc0; dx, dy = dst.noc0
        path = []
        if noc == 0:
            x = sx
            while x != dx: x = (x + 1) % W; path.append((x, sy))
            y = sy
            while y != dy: y = (y + 1) % H; path.append((dx, y))
        else:
            y = sy
            while y != dy: y = (y - 1) % H; path.append((sx, y))
            x = sx
            while x != dx: x = (x - 1) % W; path.append((x, dy))
        return path

    def run_dual(self, pairs, length=0x40000, fires=1):
        """Drive each pair on whichever NoC routes it shortest (NoC0 east+south /
        NoC1 west+north, verified: NIU1 uses noc0 coords). Returns:
          foot  : {0: {noc0_xy -> hits}, 1: {...}}  computed route per NoC,
          total : bytes written,  secs : wall time.
        Routes are computed (deterministic dimension-ordered); endpoint landings are
        measured separately (read_dram_counters / slave counters)."""
        import time
        length = min(length, MAX_LEN)
        pairs = [(s, d) for s, d in pairs
                 if s.kind in SAFE_KINDS and d.kind in SAFE_KINDS]
        foot = {0: {}, 1: {}}
        total = 0
        t0 = time.monotonic()
        for _ in range(fires):
            for s, d in pairs:
                noc = self._pick_noc(s, d)
                sx, sy = s.noc0; dx, dy = d.noc0
                for off, val in [(TARG_LO, 0x20000), (TARG_MID, 0), (TARG_HI, _hi(sx, sy)),
                                 (RET_LO, 0x10000), (RET_MID, 0), (RET_HI, _hi(dx, dy)),
                                 (PKT_TAG, 0), (CTRL, NOC_CMD_WR), (AT_LEN_BE, length)]:
                    self._wn(s, off, val, noc)
                c0 = self._rdn(s, 0x200, 62, noc)
                self._wn(s, CMD_CTRL, 1, noc)
                for _ in range(50000):
                    if self._rdn(s, CMD_CTRL, 1, noc)[0] == 0:
                        break
                total += self._wr_flits(self._rdn(s, 0x200, 62, noc), c0) * 64
                for p in self._route(s, d, noc):
                    foot[noc][p] = foot[noc].get(p, 0) + 1
        return foot, total, time.monotonic() - t0

    def dram_access_tiles(self):
        """One *working* write-access tile per GDDR6 controller. A controller exposes
        3 endpoint tiles but not all accept NoC writes (the row-0/1 ones don't), so we
        probe each with a tiny write and keep the first whose slave counter advances.
        Cached. Safe (tensix source -> DRAM, 4KB)."""
        if getattr(self, "_dram_access", None):
            return self._dram_access
        src = self.scan[0]
        out = {}
        for ctrl, tiles in sorted(self.fp.dram_ctrl.items()):
            chosen = tiles[0]
            for t in tiles:
                noc = self._pick_noc(src, t)
                b = sum(self._rdn(t, 0x200, 62, noc)[i] for i in nc.RX_SLAVE_IN)
                sx, sy = src.noc0; dx, dy = t.noc0
                for off, val in [(TARG_LO, 0x20000), (TARG_MID, 0), (TARG_HI, _hi(sx, sy)),
                                 (RET_LO, 0x10000), (RET_MID, 0), (RET_HI, _hi(dx, dy)),
                                 (PKT_TAG, 0), (CTRL, NOC_CMD_WR), (AT_LEN_BE, 0x1000)]:
                    self._wn(src, off, val, noc)
                self._wn(src, CMD_CTRL, 1, noc)
                for _ in range(50000):
                    if self._rdn(src, CMD_CTRL, 1, noc)[0] == 0:
                        break
                a = sum(self._rdn(t, 0x200, 62, noc)[i] for i in nc.RX_SLAVE_IN)
                if (a - b) & 0xFFFFFFFF:
                    chosen = t
                    break
            out[ctrl] = chosen
        self._dram_access = out
        return out

    def read_dram_counters(self):
        """Per-GDDR6-controller writes-landed flits, split by NoC the write arrived on
        (NoC0 writes land on the tile's NIU0 slave counters, NoC1 on NIU1): summed over
        the controller's 3 endpoint tiles -> {ctrl: {0: flits_noc0, 1: flits_noc1}}.
        DRAM tiles are safe to touch."""
        out = {}
        for ctrl, tiles in self.fp.dram_ctrl.items():
            d = {0: 0, 1: 0}
            for t in tiles:
                for noc in (0, 1):
                    try:
                        w = self._rdn(t, 0x200, 62, noc)
                    except Exception:
                        continue
                    d[noc] += sum(w[i] for i in nc.RX_SLAVE_IN)   # SLV_*_WR_DATA_WORD_RECEIVED
            out[ctrl] = d
        return out

    def _wr_flits(self, c1, c0):
        return (((c1[8] - c0[8]) & 0xFFFFFFFF) + ((c1[9] - c0[9]) & 0xFFFFFFFF))

    def run_pattern(self, pairs, length=0x40000, fires=3):
        """Run a whole data-movement pattern (list of (src_tile,dst_tile)).

        Returns (footprint:{(x,y)->transit}, total_bytes, seconds). footprint is the
        decoded directional transit per Tensix router -- which nodes the pattern's
        traffic actually flowed through.

        Each pair is PROGRAMMED + FIRED + measured individually: scatter patterns
        (one src -> many dst) share a source tile and reuse initiator #0, so the
        per-pair program must immediately precede its fire (else only the last dst
        would be written). Slower (serial) but correct for any src/dst mix."""
        import time
        length = min(length, MAX_LEN)        # keep source read inside L1
        pairs = [(s, d) for s, d in pairs
                 if s.kind in SAFE_KINDS and d.kind in SAFE_KINDS]
        base_t = {t.key: self._transit(t) for t in self.scan}
        total = 0
        t0 = time.monotonic()
        for _ in range(fires):
            for s, d in pairs:
                sx, sy = s.noc0; dx, dy = d.noc0
                for off, val in [(TARG_LO, 0x20000), (TARG_MID, 0), (TARG_HI, _hi(sx, sy)),
                                 (RET_LO, 0x10000), (RET_MID, 0), (RET_HI, _hi(dx, dy)),
                                 (PKT_TAG, 0), (CTRL, NOC_CMD_WR), (AT_LEN_BE, length)]:
                    self._w(s, off, val)
                c0 = self._rd(s, COUNTERS, 62)
                self._w(s, CMD_CTRL, 1)
                for _ in range(50000):
                    if self._rd(s, NIU0 + CMD_CTRL, 1)[0] == 0:
                        break
                total += self._wr_flits(self._rd(s, COUNTERS, 62), c0) * 64
        secs = time.monotonic() - t0
        foot = {}
        for t in self.scan:
            now = self._transit(t); b = base_t[t.key]
            foot[t.noc0] = 0 if (now is None or b is None) else (now - b) & 0xFFFFFFFF
        return foot, total, secs


if __name__ == "__main__":
    import sys
    from ttexalens import init_ttexalens
    from bhtop.floorplan import build
    ctx = init_ttexalens(); fp = build(ctx)
    cells, _, _ = fp.grid("noc0")
    a = tuple(int(v) for v in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1, 3)
    b = tuple(int(v) for v in sys.argv[2].split(",")) if len(sys.argv) > 2 else (6, 4)
    inj = Injector(fp, ctx)
    route, sent, landed = inj.inject_and_trace(cells[a], cells[b])
    print(f"inject {a} -> {b}:  src sent {sent/1e3:.0f} kB,  dst landed {landed/1e3:.0f} kB\n")
    mx = max((v for v in route.values() if v), default=1)
    for y in range(fp.grid('noc0')[2]):
        line = f" y={y:2d} "
        for x in range(fp.grid('noc0')[1]):
            t = cells.get((x, y))
            if t is None or t.kind != "tensix":
                line += " ·" if t is None else f" {t.glyph}"
                continue
            v = route.get(t.key, 0) or 0
            mark = "##" if v > mx*0.4 else "▓▓" if v > mx*0.1 else "··" if v > 0 else "  "
            if (x, y) == a: mark = "SS"
            elif (x, y) == b: mark = "DD"
            line += mark
        print(line)
    print("\nroute nodes (transit>0):", sorted([k for k,v in route.items() if v and (k!=a and k!=b)]))
