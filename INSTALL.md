# TT Blackhole Stack — Clean Install Guide

Rebuild the full Tenstorrent Blackhole software stack on a fresh box so it integrates
cleanly with **bhtop** (NoC telemetry + bare-metal het) and **tt-splat** (on-device 3DGS
training). Written against the working configuration on `ttstar` as of 2026-07-11.

## Reference configuration (what this box actually ran)

| Component | Version / source | Notes |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | kernel `6.8.0-124-generic` |
| Python | 3.10.12 (system `/usr/bin/python3`) | tt-smi + bhtop use this; tt-metal builds its own venv |
| **kmd** | `tenstorrent` **2.8.0** (DKMS) | kernel driver, `/dev/tenstorrent/0` |
| **umd** | **fork `dtsunami/tt-umd`** (base 0.9.8) | **the keystone** — adds the Blackhole PCIe **DMA path** (~9.5 GB/s vs stock ~2.6 MB/s) |
| **smi** | `tt-smi` **4.1.2** (`~/.local`) | reset + telemetry CLI |
| **metal** | `tt-metal` **v0.74.0-dev20260621** (`8e6f2bab553`, `main`) | builds `python_env` with **ttnn** + torch |
| **nn** | **ttnn** — built *inside* tt-metal | not a separate clone (see §5b) |
| bhtop | `dtsunami/tt.git` | own `.venv`; deps `tt-exalens>=0.3.21`, `pyluwen` |
| tt-splat | `dtsunami/tt-splat.git` | installs `-e` **into tt-metal's `python_env`** |
| hardware | Blackhole p150a, **PCIe Gen4 slot** | Gen4, *not* a marginal Gen5 link (see §10) |

**Dependency / build order:** `kmd → umd(fork) → smi → metal(+ttnn) → { bhtop, tt-splat }`

---

## 1. System prerequisites

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build git git-lfs \
  python3.10 python3.10-venv python3.10-dev python3-pip \
  clang cmake-format libhwloc-dev libhugetlbfs-dev \
  dkms linux-headers-$(uname -r) pciutils
```

Add yourself to the `sudo` group and log out/in. Confirm the card is on the bus:

```bash
lspci -nn | grep -i 1e52      # expect: Processing accelerators [1e52:b140]  (Blackhole)
```

> **Fast path:** Tenstorrent's `tt-installer` (one-liner from their GitHub) sets up kmd +
> hugepages + tt-smi automatically. This guide does it by repo so the **umd fork** and the
> bhtop/tt-splat integration are explicit. Use the installer for §2/§4 if you prefer, then
> jump to §3.

---

## 2. KMD — kernel-mode driver (`tenstorrent` 2.8.0)

```bash
git clone https://github.com/tenstorrent/tt-kmd.git ~/tt-kmd
cd ~/tt-kmd
git checkout ttkmd-2.8.0            # match the known-good version
sudo dkms add .
sudo dkms install tenstorrent/2.8.0
sudo modprobe tenstorrent
modinfo tenstorrent | grep ^version   # -> 2.8.0
```

**Hugepages** (TT device open requires 1 GB hugepages). Install the systemd unit from
`tt-system-tools`, or set manually and reboot:

```bash
git clone https://github.com/tenstorrent/tt-system-tools.git ~/tt-system-tools
cd ~/tt-system-tools && sudo ./hugepages-setup.sh
sudo reboot
```

After reboot: `ls /dev/tenstorrent/` should show `0`, and
`grep HugePages_Total /proc/meminfo` should be non-zero.

---

## 3. UMD fork — the DMA keystone (`dtsunami/tt-umd`)

This is the **most important non-standard step**. The stock UMD chops every device
transfer into 4-byte register accesses (~2.6 MB/s). The fork adds a Blackhole D2H/H2D **DMA**
path (`BlackholeDmaTransfer`, ~9.5 GB/s copy / ~21 GB/s zero-copy), bit-exact on p150a. The
entire het perf story (param readback, gt upload, resident training) depends on it.

```bash
git clone https://github.com/dtsunami/tt-umd.git ~/tt-umd
cd ~/tt-umd
# The DMA patches touch: device/.../pcie_dma/blackhole_dma_transfer.{hpp,cpp},
#   device/.../pcie_protocol.{hpp,cpp}, tests/microbenchmark/.../test_pcie_dma.cpp
# (see the tt-umd branch/commit that carries them — §PRESERVE below)
```

Build the self-contained Python bindings `.so` (nanobind). See
`~/tt-umd/docs/CMAKE_INSTALL_BUILD.md` for the exact CMake invocation; the artifact this box
used was `tt_umd-096-selfcontained.cpython-310-x86_64-linux-gnu.so`. Keep that `.so` — it is
**swapped into every run env** in §8.

> The fork also renamed `set_power_state → set_clock_state` and reworked
> `TopologyDiscoveryOptions` (dropped `no_wait_for_eth_training`). That's the source of the
> `tt-smi -r 0` reinit warning — see §4.

---

## 4. SMI — `tt-smi` 4.1.2

```bash
python3 -m pip install --user tt-smi==4.1.2     # lands in ~/.local/bin + ~/.local/lib
tt-smi --version    # -> 4.1.2
```

**Fork caveat (known-good workaround):** `tt-smi -r 0` completes the warm reset but then
throws a *harmless* `AttributeError: 'TopologyDiscoveryOptions' object has no attribute
'no_wait_for_eth_training'` in its post-reset re-scan (tt-smi expects the old UMD API; the
fork dropped that field). The reset **succeeded** — only the cosmetic rescan failed. Always
reset with:

```bash
tt-smi -r 0 --no_reinit      # reset happens, skips the broken rescan, exits 0
```

(Optional permanent fix: guard `tt_smi_reset.py:187` with
`if hasattr(options, "no_wait_for_eth_training"):`.) On a single p150a the flag is a no-op
anyway — it only governs *inter-chip* Ethernet training.

---

## 5. METAL — `tt-metal` (+ ttnn)

```bash
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
git checkout 8e6f2bab553           # v0.74.0-dev20260621 (known-good)
git submodule update --init --recursive
export ARCH_NAME=blackhole TT_METAL_HOME=$HOME/tt-metal
./build_metal.sh                   # builds tt-metal + ttnn (long)
./create_venv.sh                   # -> python_env/  (ttnn + torch, imported by tt-splat)
```

Verify: `source python_env/bin/activate && python -c "import ttnn; print('ttnn ok')"`.

### 5b. "nn" = ttnn (not a separate repo)

ttnn ships **inside** tt-metal — `import ttnn` from the `python_env` above. tt-splat's device
path uses it (see [tt-splat-llk-dram-arch]). There is no separate `tt-nn` clone to make.
*(If by "nn" you meant a different repo — e.g. tt-forge / tt-mlir / a models repo — say so and
I'll add a section; nothing on this box used one.)*

### 5c. Local tt-metal additions (restore from preservation)

This box carried **untracked** local work under `tt_metal/programming_examples/contributed/`
and `tests/.../data_movement/` — the resident Tensix **bootloader**, the **gather_scatter_3hop**
kernel, **jit** examples, `agg_bw.py`, and a patched `tt_metal/hw/inc/dataflow_api.h`. These are
**not upstream** and only partly mirrored in bhtop. Restore them from the preservation bundle
(§PRESERVE) after the clone.

---

## 6. bhtop — NoC telemetry + bare-metal het

```bash
git clone https://github.com/dtsunami/tt.git ~/bhtop
cd ~/bhtop
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .                   # tt-exalens, pyluwen, rich, textual, fastapi
```

**Swap the fork `.so` into this venv** (§8). Then:

```bash
bhtop            # live NoC NIU-counter telemetry (sanity check the device is reachable)
```

CLI entry points: `bhtop`, `bhtop-inject`, `bhtop-metal`, `bhtop-l2cpu`, `bhtop-tensix`,
`bhtop-web`, `bhtop-kern`.

---

## 7. tt-splat — on-device 3DGS training

tt-splat installs **into tt-metal's `python_env`** (it reuses tt-metal's torch/ttnn — a plain
`pip install -e .` there adds only light deps and never rebuilds ttnn):

```bash
git clone https://github.com/dtsunami/tt-splat.git ~/tt-splat
cd ~/tt-splat
source ~/tt-metal/python_env/bin/activate
pip install -e .                   # typer, pillow, fastapi, etc. — torch/ttnn already present
```

**Swap the fork `.so` into `python_env` too** (§8). Smoke test:

```bash
cd ~/tt-splat
ttgs blackhole work/nerf_data/<scene>          # ttnn path
TT_DEVICE_BAREMETAL=1 ttgs blackhole <scene>   # bare-metal het path (bhtop grid_engine)
```

Bare-metal defaults are set in `server/baremetal_resident.py`: `TT_BM_NOCPACE=1`,
`TT_BM_PERF_BUSY=1` (AICLK OC), `TT_BM_AUTORECOVER=1`. See [het-pipeline-noc-wedge] and
[tt-splat-baremetal-campaign] for the tuning knobs.

---

## 8. The `.so` swap (do this in EVERY run env)

The fork's DMA lives in a compiled binding. Every environment that opens the device must load
the fork `.so`, not the pip-installed stock one:

```bash
FORK_SO=~/tt-umd/tt_umd-096-selfcontained.cpython-310-x86_64-linux-gnu.so   # your fork build

# bhtop venv:
cp "$FORK_SO" ~/bhtop/.venv/lib/python3.10/site-packages/tt_umd/tt_umd.cpython-310-x86_64-linux-gnu.so
# tt-metal python_env (used by tt-splat):
cp "$FORK_SO" ~/tt-metal/python_env/lib/python3.10/site-packages/tt_umd/tt_umd.cpython-310-x86_64-linux-gnu.so
```

DMA readback is gated in `grid_engine.py` by `TT_DMA_READBACK=1` (default), which sets
`ctx.use_4B_mode=False`. If the fork `.so` is missing, it falls back to the slow register path
safely — so a broken swap shows up as *slow*, not *crashed*. See [bh-exalens-dma-wall].

---

## 9. Verification checklist

```bash
modinfo tenstorrent | grep ^version                 # 2.8.0
lspci -nn | grep 1e52                                # b140 present
tt-smi -s                                            # telemetry snapshot
python -c "import tt_umd, ttnn; print('umd+ttnn ok')" # inside python_env
bhtop                                                # NoC counters render
TT_DEVICE_BAREMETAL=1 ttgs blackhole <scene>         # het step runs
```

---

## 10. Gotchas index (hard-won — read before you burn a day)

- **PCIe slot:** run the card in a **Gen4** slot. A marginal Gen5 x16 link throws correctable
  RxErrors under heavy het traffic → **hard host freeze** (no catchable exception). The step is
  device-bound, so Gen4 bandwidth costs ~nothing. [het-pipeline-noc-wedge]
- **`tt-smi -r 0` →** always `--no_reinit` with the umd fork (§4).
- **`.so` swap** must be redone in **every** venv after any `pip install`/reinstall of tt_umd
  (§8), else you silently drop to 2.6 MB/s.
- **Hugepages** must be up *before* the first device open, or UMD init fails.
- **het NoC wedge:** the 4-hart NIU race is silicon-intrinsic; survive it with
  `TT_BM_AUTORECOVER=1` + checkpoint every ~500–1000 steps; each `tt-smi -r 0` slightly wears
  the card, so a very long campaign benefits from a periodic process restart. [het-pipeline-noc-wedge]
- **NoC0-hang hazard:** never poke ARC/Security/PCIe/L2CPU register windows over NoC0 — it hangs
  the tile. Recover with `tt-smi -r 0`. [bh-noc-hang-hazard]

---

## PRESERVE — before wiping this box

Uncommitted/at-risk work that must be pushed or bundled first (see the session report):

1. **tt-umd fork DMA patches** — 5 files, ~337 lines, committed *nowhere*. Push to
   `dtsunami/tt-umd`. **Highest priority** (this whole guide's §3 depends on it).
2. **tt-metal local additions** — bootloader / gather_scatter_3hop / jit / `agg_bw.py` /
   `dataflow_api.h`. `origin` is upstream tenstorrent (can't push) and they're only partly in
   bhtop → bundle (`git format-patch` or copy into a dtsunami repo) before wipe.
3. **comfy/tt-metal** — `models/tt_dit/layers/lora.py` (untracked, minor).
4. bhtop, tt-splat, arcgs — already clean & pushed to `dtsunami/*`. Safe.
