"""
tensix.loader — read and POKE a Tensix kernel's runtime args directly in L1 over the NoC
(tt-exalens), and re-trigger the firmware go — the on-the-fly editing path. Sibling of the x280
L2CPU loader: there we own crt0 + a mailbox; here we ride tt-metal's resident firmware launch
protocol (see tensix.abi + TENSIX_ABI.md).

HYBRID MODEL (v1): tt-metal opens the device + JIT-builds + runs the program once — that loads the
firmware, the kernel binary, and writes the launch_msg + RTAs into L1. From then on bhtop can:
  * READ the live launch ring / kernel_config / RTAs (always safe — Tensix L1 reads do NOT touch
    the ARC/PCIe/L2CPU hang-hazard surface),
  * POKE new runtime-arg values into L1 (get_arg_val<i> is just *(rta_base+i*4)),
  * re-issue the go signal to run again with the new args — NO recompile, NO host rebuild.

Compile-time args (get_compile_time_arg_val) are baked into the kernel binary at JIT and can't be
poked — only runtime args. Re-go semantics are cleanest in slow-dispatch mode
(TT_METAL_SLOW_DISPATCH_MODE=1), where the host owns the launch ring rather than a dispatch core;
in fast dispatch the dispatcher also drives go, so treat go() as experimental there.

All I/O is bounded to the worker's L1 [0, MEM_L1_SIZE). Device access only — keep it off the
device thread the same way the rest of bhtop does.
"""
from . import abi


class TensixLauncher:
    """Drive one Tensix worker core's launch mailbox over the NoC.

    `coord` is anything tt-exalens accepts as a location: an OnChipCoordinate (what bhtop's
    floorplan hands out as `tile.coord`) or a coord string. Share the DeviceManager's `ctx` so the
    chip never sees a second owner."""

    def __init__(self, coord, ctx=None, device_id=0, noc_id=0):
        self.coord = coord
        self.device_id = device_id
        self.noc_id = noc_id
        if ctx is None:
            from ttexalens import init_ttexalens
            ctx = init_ttexalens()
        self.ctx = ctx

    @classmethod
    def at(cls, x, y, ctx=None, device_id=0, noc_id=0):
        """Construct from a worker's noc0 (x, y) — the coordinate space the rest of bhtop uses."""
        if ctx is None:
            from ttexalens import init_ttexalens
            ctx = init_ttexalens()
        return cls(worker_coord(ctx, x, y, device_id), ctx=ctx, device_id=device_id, noc_id=noc_id)

    # ---- raw L1 I/O (guarded) --------------------------------------------------------
    def _guard(self, addr, nbytes):
        if not (abi.MEM_L1_BASE <= addr and addr + nbytes <= abi.MEM_L1_SIZE):
            raise ValueError(f"L1 access [{addr:#x}, +{nbytes}) outside [0, {abi.MEM_L1_SIZE:#x})")

    def rd(self, addr, nwords=1):
        from ttexalens.tt_exalens_lib import read_words_from_device
        self._guard(addr, nwords * 4)
        return read_words_from_device(self.coord, addr, device_id=self.device_id,
                                      word_count=nwords, context=self.ctx, noc_id=self.noc_id)

    def rd_bytes(self, addr, nbytes):
        return abi.words_to_bytes(self.rd(addr, (nbytes + 3) // 4))[:nbytes]

    def wr(self, addr, words):
        from ttexalens.tt_exalens_lib import write_words_to_device
        words = list(words)
        self._guard(addr, len(words) * 4)
        write_words_to_device(self.coord, addr, words, device_id=self.device_id,
                              context=self.ctx, noc_id=self.noc_id)

    # ---- launch ring + kernel config -------------------------------------------------
    def read_rd_ptr(self):
        return self.rd(abi.launch_rd_ptr_addr(), 1)[0]

    def read_launch(self, idx):
        """Decode launch[idx].kernel_config."""
        buf = self.rd_bytes(abi.launch_addr(idx), abi.LAUNCH_STRIDE)
        return abi.decode_kernel_config(buf)

    def read_ring(self):
        """Decode all launch entries + the read pointer. `active` is the best-effort index of the
        program the firmware last ran (dispatcher pre-increments rd_ptr, so the live entry is
        usually rd_ptr-1); verify against host_assigned_id for a known run."""
        rd_ptr = self.read_rd_ptr()
        entries = [self.read_launch(i) for i in range(abi.LAUNCH_ENTRIES)]
        return {"rd_ptr": rd_ptr, "active": self._active_index(rd_ptr, entries), "entries": entries}

    def _read_watcher_ids(self, idx):
        """watcher_kernel_ids[5] for launch entry `idx` (one small L1 read)."""
        import struct
        b = self.rd_bytes(abi.launch_addr(idx) + abi.KCFG_WATCHER_IDS_OFF, abi.MAX_PROCS * 2)
        return list(struct.unpack(f"<{abi.MAX_PROCS}H", b))

    def brief(self, kernels=None):
        """A cheap (~5 NoC reads) liveness probe for scanning every worker: which launch entry is
        active, whether it has a resident program (enables != 0), its program id, and the go signal.
        If `kernels` (a watcher_kernel_id -> kernel-info map, from the Inspector) is given, also
        resolves the kernel NAME(s) on this core. Avoids decoding all 8 entries so a scan stays fast."""
        rd = self.read_rd_ptr()
        idx, en = rd % abi.LAUNCH_ENTRIES, 0
        for cand in ((rd - 1) % abi.LAUNCH_ENTRIES, rd % abi.LAUNCH_ENTRIES):
            e = self.rd(abi.launch_addr(cand) + abi.KCFG_ENABLES_OFF, 1)[0]
            if e:
                idx, en = cand, e
                break
        host_id = self.rd(abi.launch_addr(idx) + abi.KCFG_HOST_ID_OFF, 1)[0]
        go = self.read_go()
        out = {"rd_ptr": rd, "active": idx, "enables": en, "resident": bool(en),
               "host_id": host_id, "signal": go["signal_name"]}
        if en:
            wids = self._read_watcher_ids(idx)
            out["watcher_kernel_ids"] = [wids[p] for p in range(abi.MAX_PROCS) if en & (1 << p)]
            if kernels is not None:
                ks = _resolve_kernels(wids, en, kernels)
                out["kernels"] = ks
                out["kernel_names"] = _dedup_names(ks)
                out["user_kernel"] = any(not k.get("infra") for k in ks)   # vs dispatch infra
        return out

    @staticmethod
    def _active_index(rd_ptr, entries):
        order = [(rd_ptr - 1) % abi.LAUNCH_ENTRIES, rd_ptr % abi.LAUNCH_ENTRIES]
        order += [i for i in range(abi.LAUNCH_ENTRIES) if i not in order]
        for i in order:
            if entries[i]["enables"]:
                return i
        return rd_ptr % abi.LAUNCH_ENTRIES

    def _kcfg(self, index=None):
        if index is None:
            index = self.read_ring()["active"]
        return self.read_launch(index), index

    # ---- runtime args (the pokeable knob) --------------------------------------------
    def rta_addr(self, proc, index=None):
        kcfg, _ = self._kcfg(index)
        return abi.rta_l1_addr(kcfg, proc)

    def read_rta(self, proc, nwords, index=None, common=False):
        kcfg, _ = self._kcfg(index)
        base = abi.crta_l1_addr(kcfg, proc) if common else abi.rta_l1_addr(kcfg, proc)
        return self.rd(base, nwords)

    def write_rta(self, proc, values, index=None, arg_offset=0, common=False):
        """POKE runtime-arg words for `proc` starting at arg index `arg_offset`. `values` are raw
        u32s (already host-encoded — e.g. a packed coord or an address). Returns the L1 address
        written. Pair with go() to re-run with the new args."""
        kcfg, _ = self._kcfg(index)
        base = abi.crta_l1_addr(kcfg, proc) if common else abi.rta_l1_addr(kcfg, proc)
        addr = base + arg_offset * 4
        self.wr(addr, values)
        return addr

    # ---- go signal -------------------------------------------------------------------
    def read_go(self):
        gi = self.rd(abi.go_index_addr(), 1)[0]
        word = self.rd(abi.go_addr(gi), 1)[0]
        return {"go_index": gi, "raw": word, **abi.decode_go(word)}

    def go(self, signal=abi.RUN_MSG_GO):
        """Set the active go message's signal byte (default GO=0x80). Re-runs the resident kernel
        with whatever RTAs are currently in L1. Experimental under fast dispatch — see module doc."""
        gi = self.rd(abi.go_index_addr(), 1)[0]
        addr = abi.go_addr(gi)
        word = self.rd(addr, 1)[0]
        self.wr(addr, [abi.with_signal(word, signal)])
        return {"go_index": gi, "addr": addr, "signal": signal,
                "signal_name": abi.SIGNAL_NAME.get(signal, hex(signal))}

    # ---- human-readable snapshot -----------------------------------------------------
    def snapshot(self, index=None, kernels=None):
        """Decode the active launch entry. If `kernels` (watcher_kernel_id -> Inspector info) is
        given, each enabled processor row carries the KERNEL running on it (name/source/hash)."""
        ring = self.read_ring()
        idx = ring["active"] if index is None else index
        kcfg = ring["entries"][idx]
        wids = kcfg.get("watcher_kernel_ids", [0] * abi.MAX_PROCS)
        procs = []
        for p in kcfg["enabled_procs"]:
            row = {"proc": abi.PROC_NAME.get(p, p), "rta_addr": hex(abi.rta_l1_addr(kcfg, p)),
                   "crta_addr": hex(abi.crta_l1_addr(kcfg, p)), "watcher_kernel_id": wids[p]}
            if kernels is not None:
                k = kernels.get(wids[p])
                if k:
                    row["kernel"] = {"name": k.get("name"), "source": k.get("source"),
                                     "hash": k.get("hash"), "program_id": k.get("program_id"),
                                     "infra": bool(k.get("infra"))}
            procs.append(row)
        out = {"coord": str(self.coord), "rd_ptr": ring["rd_ptr"], "active_index": idx,
               "host_assigned_id": kcfg["host_assigned_id"], "enables": hex(kcfg["enables"]),
               "mode": "HOST" if kcfg["mode"] else "DEV",
               "kernel_config_base": [hex(b) for b in kcfg["kernel_config_base"]],
               "procs": procs, "go": self.read_go()}
        if kernels is not None:
            ks = _resolve_kernels(wids, kcfg["enables"], kernels)
            out["kernels"] = ks
            out["kernel_names"] = _dedup_names(ks)
            out["user_kernel"] = any(not k.get("infra") for k in ks)
        return out


# ---- kernel-identity resolution (pure: joins watcher ids with an Inspector map) ----
def _resolve_kernels(watcher_ids, enables, kernels):
    """[{proc, name, source, hash, program_id}] for the enabled processors whose watcher_kernel_id
    resolves in the Inspector map. `kernels` = {watcher_kernel_id: info}."""
    out = []
    for p in range(abi.MAX_PROCS):
        if not (enables & (1 << p)):
            continue
        k = kernels.get(watcher_ids[p])
        if k:
            out.append({"proc": abi.PROC_NAME.get(p, p), "name": k.get("name"),
                        "source": k.get("source"), "hash": k.get("hash"),
                        "program_id": k.get("program_id"), "infra": bool(k.get("infra"))})
    return out


def _dedup_names(ks):
    """Unique kernel names in order (a program's reader/writer/compute often differ)."""
    seen, names = set(), []
    for k in ks:
        n = k.get("name")
        if n and n not in seen:
            seen.add(n); names.append(n)
    return names


def worker_coords(ctx=None, device_id=0):
    """Convenience: the OnChipCoordinates of the device's Tensix worker cores (for CLI/exploration)."""
    if ctx is None:
        from ttexalens import init_ttexalens
        ctx = init_ttexalens()
    dev = ctx.devices[device_id]
    for name in ("functional_workers", "tensix", "worker"):
        try:
            locs = dev.get_block_locations(name)
            if locs:
                return locs
        except Exception:
            continue
    return []


def worker_coord(ctx, x, y, device_id=0):
    """Resolve a worker's noc0 (x, y) to its OnChipCoordinate (raises if it isn't a worker)."""
    for c in worker_coords(ctx, device_id):
        try:
            if tuple(c.to("noc0")) == (x, y):
                return c
        except Exception:
            continue
    raise ValueError(f"no Tensix worker at noc0 ({x},{y})")
