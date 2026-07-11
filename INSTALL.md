# TT Blackhole Stack — Clean Install Guide

Rebuild the full Tenstorrent Blackhole software stack on a fresh box so it integrates
cleanly with **bhtop** (NoC telemetry + bare-metal het) and **tt-splat** (on-device 3DGS
training). **Target dev env: Ubuntu 24.04 LTS + Python 3.12** (chosen for upstreaming — the
stack supports it end-to-end). First proven on `ttstar` (22.04.5 / 3.10) as of 2026-07-11, which
remains the documented fallback.

## Reference configuration (what this box actually ran)

| Component | Version / source | Notes |
|---|---|---|
| OS | **Ubuntu 24.04 LTS** (primary) · 22.04.5 (fallback) | kernel 6.8.x; 24.04 ships cmake 3.28 in-repo |
| Python | **3.12** on 24.04 · 3.10 on 22.04 (system `python3`) | **one shared `~/.ttvenv`**; stack supports 3.10–3.13 |
| **kmd** | `tenstorrent` **2.8.0** (DKMS) | kernel driver, `/dev/tenstorrent/0` |
| **umd** | **fork `dtsunami/tt-umd`** = upstream + one commit `4039b93e` **"D2H DMA unlock"** | **the keystone.** Adds the Blackhole **D2H read** DMA (~9.5 GB/s vs stock ~2.6 MB/s). **Two consumers:** (a) Python `tt_umd`, `pip install -e ~/tt-umd` into `~/.ttvenv` → tt-smi, bhtop; (b) tt-metal's **C++ umd submodule**, repointed to fork branch `d2h-on-metal-pin` → ttnn (§5) |
| **smi** | `tt-smi` **5.3.1**, `pip install -e` from git into `~/.ttvenv` | reset + telemetry CLI |
| **metal** | **fork of `tenstorrent/tt-metal`** (current `main`), umd submodule → `dtsunami/tt-umd@d2h-on-metal-pin` | builds ttnn + torch **into `~/.ttvenv`** (build already targets it); needs **cmake ≥3.24 + clang-20** (§1) |
| **nn** | **ttnn** — built *inside* tt-metal | not a separate clone (see §5c) |
| bhtop | **`dtsunami/tt-baremetal.git`** (was `dtsunami/tt.git`) | `pip install -e` into `~/.ttvenv`; deps `tt-exalens>=0.3.21`, `pyluwen` |
| tt-splat | `dtsunami/tt-splat.git` | `pip install -e` **into `~/.ttvenv`** |
| hardware | Blackhole p150a, **PCIe Gen4 slot** | Gen4, *not* a marginal Gen5 link (see §10) |

**Dependency / build order:** `kmd → umd(fork) → smi → metal(+ttnn) → { bhtop, tt-splat }`

**Environment model:** every Python component lives in a **single `~/.ttvenv`** — activate it once and `pip install -e` each repo into it, so a fork edit + reinstall is all it takes to iterate. `kmd` is a kernel module (DKMS), not in the venv. Because everything shares one env, the fork's DMA binding is compiled straight into `.ttvenv` by the editable install, which **retires the per-env `.so` swap** the old multi-venv layout needed (§8).
> ⚠️ **Single-env tradeoff:** tt-metal pins heavy deps (a specific torch, plus `rich`/`textual` versions that also matter to tt-smi). Forcing them into `.ttvenv` can trigger pip upgrades/downgrades that disturb tt-smi's TUI. If that happens, the escape hatch is to let tt-metal keep its own `python_env` (drop the `PYTHON_ENV_DIR` override in §5) and swap the fork `.so` into it per §8-legacy.

---

## 1. System prerequisites

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build git git-lfs wget curl \
  python3 python3-venv python3-dev python3-pip \
  cmake-format libhwloc-dev libhugetlbfs-dev \
  dkms linux-headers-$(uname -r) pciutils
```

*(Generic `python3` → 3.12 on 24.04, 3.10 on 22.04. `cmake` here is 3.28 on 24.04 — new enough;
on 22.04 it's 3.22 and gets shadowed below.)*

**Toolchain — clang-20 always; cmake pip-upgrade only on 22.04.** current tt-metal requires
**cmake ≥3.24** and **clang-20** (`cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake`).

```bash
# clang-20 — via LLVM's official apt script (BOTH 22.04 and 24.04; not in either distro's repos)
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 20

# cmake — 22.04 ONLY (its 3.22 is too old). 24.04's apt cmake (3.28) is already new enough.
# Install into the active .ttvenv after §4 creates it:
python3 -m pip install -U "cmake<4"        # 3.31.x; shadows system 3.22
```

> **OS choice — 24.04 LTS + Python 3.12 is the primary dev-env target.** Verified the whole stack
> supports it: tt-metal's `create_venv.sh` **auto-selects 3.12 on 24.04** and its
> `install_dependencies.sh` handles g++-14 / MPI-ULFM; tt-umd builds `cp312`/`cp313`; tt-exalens
> ships a `cp312` wheel; pyluwen's `cp311-abi3` wheel is **forward-compatible with 3.12+** (abi3 =
> stable ABI, so no cp312-specific wheel is needed). The only real chore is clang-20 — identical on
> both OSes. **22.04.5 + Python 3.10** is the documented fallback (first-proven config), identical
> except you must `pip install cmake` (its 3.22 is too old).
> **One 3.12 belt-and-suspenders step:** `pip install -U setuptools` in the venv, covering any dep
> that still reaches for the `distutils` removed in 3.12. (None found in tt-metal, but cheap insurance.)

Add yourself to the `sudo` group and log out/in. Confirm the card is on the bus:

```bash
lspci -nn | grep -i 1e52      # expect: Processing accelerators [1e52:b140]  (Blackhole)
```

> **Fast path:** Tenstorrent's `tt-installer` (one-liner from their GitHub) sets up kmd +
> hugepages + tt-smi automatically. This guide does it by repo so the **umd fork** and the
> bhtop/tt-splat integration are explicit. The installer is fine for **§2 (kmd + hugepages)**,
> then jump to §3 — but do **not** use it for tt-smi: §4 wants the **editable git** tt-smi in
> `~/.ttvenv` so you can fork it, not the installer's system-wide wheel.

---

## 1b. Remote access — set it up FIRST (the part that actually hurts on a wipe)

Do this right after the base install so everything else can be driven headless from your laptop.

**SSH:**

```bash
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
# from your laptop, push your key, then disable password auth:
#   ssh-copy-id starboy@ttstar
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

**VSCode tunnel** (persistent, survives reboot, no inbound port / port-forward needed):

```bash
# headless VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz && sudo mv code /usr/local/bin/

# one-time device-code login (prints a github.com/login/device code), stable tunnel name:
code tunnel --accept-server-license-terms --name ttstar

# then persist it across reboots (Ctrl-C the above first):
code tunnel service install --accept-server-license-terms --name ttstar
```

Connect from anywhere via **`https://vscode.dev/tunnel/ttstar`** or the *Remote - Tunnels*
extension. The tunnel name `ttstar` is stable across reinstalls, so the *next* wipe only needs the
one device-code re-login — same URL, no reconfig.

> The device-code login is one-time; the systemd service reconnects automatically on boot.

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
cd ~
git clone https://github.com/tenstorrent/tt-system-tools.git
cd ~/tt-system-tools/hugepages-setup

# 1. put the script where the service looks for it
sudo mkdir -p /opt/tenstorrent/bin
sudo cp hugepages-setup.sh /opt/tenstorrent/bin/
sudo chmod +x /opt/tenstorrent/bin/hugepages-setup.sh

# 2. install both systemd units (keep the escaped mount filename EXACTLY)
sudo cp tenstorrent-hugepages.service /etc/systemd/system/
sudo cp 'dev-hugepages\x2d1G.mount' /etc/systemd/system/

# 3. reload + enable + start now (no reboot needed to test)
sudo systemctl daemon-reload
sudo systemctl enable --now tenstorrent-hugepages.service 'dev-hugepages\x2d1G.mount'
```

```bash
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages   # expect 4
mount | grep hugepages-1G                                        # expect /dev/hugepages-1G mounted
systemctl is-enabled tenstorrent-hugepages.service 'dev-hugepages\x2d1G.mount'  # both enabled
```

---

## 3. UMD fork — the DMA keystone (`dtsunami/tt-umd`)

This is the **most important non-standard step**. Stock UMD without DMA chops every device
transfer into 4-byte register accesses (~2.6 MB/s). Current upstream already carries the
Blackhole H2D/write DMA (`blackhole_dma_transfer.cpp`), but **the D2H *read* unlock is not
upstream** — it's the fork's single delta commit `4039b93e "D2H DMA unlock"` (+337 lines across
`blackhole_dma_transfer.{hpp,cpp}`, `pcie_protocol.{hpp,cpp}`, `test_pcie_dma.cpp`). It brings
D2H readback to ~9.5 GB/s (~21 GB/s zero-copy), bit-exact on p150a. The entire het perf story
(param readback, gt upload, resident training) depends on it.

**Two consumers need the D2H commit** — they're wired independently:
- **Python `tt_umd`** (tt-smi, bhtop): `pip install -e ~/tt-umd` into `~/.ttvenv` — §4.
- **tt-metal's C++ umd** (ttnn): a *separate* vendored submodule; the fork is grafted in via a
  cherry-pick onto tt-metal's exact umd pin — §5. Installing the Python fork does **not** affect
  ttnn's device path.

```bash
git clone https://github.com/dtsunami/tt-umd.git ~/tt-umd
cd ~/tt-umd
# The DMA patches touch: device/.../pcie_dma/blackhole_dma_transfer.{hpp,cpp},
#   device/.../pcie_protocol.{hpp,cpp}, tests/microbenchmark/.../test_pcie_dma.cpp
# (see the tt-umd branch/commit that carries them — §PRESERVE below)
```

**Install** happens in §4 as `pip install -e ~/tt-umd` into the shared `~/.ttvenv` — that
editable build *compiles the fork's own binding* (with the DMA path) straight into the venv,
so there is nothing to swap. Just clone here; don't build anything standalone yet.

> **Legacy (multi-venv only):** the old flow built a self-contained nanobind `.so`
> (`tt_umd-096-selfcontained.cpython-310-x86_64-linux-gnu.so`, see
> `~/tt-umd/docs/CMAKE_INSTALL_BUILD.md`) and copied it over the *stock* pip wheel in each env
> (§8-legacy). With a single `.ttvenv` + editable install, that step is retired — **verify the
> DMA benchmark in §2-check before deleting §8** to be sure the editable build carries the DMA
> path and not just telemetry.

> The fork also renamed `set_power_state → set_clock_state` and reworked
> `TopologyDiscoveryOptions` (dropped `no_wait_for_eth_training`). That was the source of the
> *historical* `tt-smi 4.1.2 -r 0` reinit warning — resolved on 5.3.1, see §4.

---

## 4. SMI — `tt-smi` from git repo

Create the **one shared venv** and install both tt-smi and the umd fork editable into it.
Order matters: install tt-smi first (it pulls the stock `tt-umd==0.9.5` wheel it pins), then
`-e` the fork **over** it (upgrades to 0.9.8 and swaps in the editable, DMA-carrying binding).

```bash
python3 -m venv ~/.ttvenv
source ~/.ttvenv/bin/activate
git clone https://github.com/tenstorrent/tt-smi.git ~/tt-smi
# ~/tt-umd already cloned in §3
pip install -e ~/tt-smi     # pulls stock tt-umd 0.9.5 as a dep
pip install -e ~/tt-umd     # -> fork 0.9.8; pip prints a "tt-smi requires tt-umd==0.9.5"
                            #    conflict WARNING — cosmetic, install succeeds & imports fine
```

Verify — should report the version and see the Blackhole card, link speed/width, telemetry:

```bash
tt-smi --version    # -> 5.3.1
tt-smi -ls          # list boards: UMD chip id, PCI BDF, /dev/tenstorrent/<n>
tt-smi              # full TUI
```

**Reset:** on this stack (`tt-smi 5.3.1` + umd fork `0.9.8`) `tt-smi -r 0` performs the warm
reset **and** the post-reset re-init cleanly — topology discovery completes and it reports the
firmware bundle (e.g. `19.11.0`), exit 0. No workaround needed:

```bash
tt-smi -r 0         # warm reset + reinit, works clean on 5.3.1
```

> **Historical:** the old `tt-smi 4.1.2` + fork combo threw a *harmless*
> `AttributeError: 'TopologyDiscoveryOptions' object has no attribute 'no_wait_for_eth_training'`
> in the post-reset rescan (4.1.2 expected the old UMD API the fork had dropped), and the
> workaround was `tt-smi -r 0 --no_reinit`. **5.3.1 no longer needs this** — the current smi and
> the fork's UMD API line up. If you ever pin back to 4.1.2, `--no_reinit` is the escape hatch
> (the flag only governs *inter-chip* Ethernet training, a no-op on a single p150a).

---

## 5. METAL — `tt-metal` (+ ttnn)

> **Prereqs:** cmake ≥3.24 + clang-20 from §1 must be on PATH, or `build_metal.sh` fails at
> `cmake_minimum_required` / compiler detection. Verify: `cmake --version` ≥3.24, `clang-20 --version`.
>
> **Let tt-metal install its own build deps** — it's OS-aware (g++-14 + MPI-ULFM on 24.04,
> g++-12 on 22.04) so you don't hand-maintain the list: `cd ~/tt-metal && sudo ./install_dependencies.sh`.
> On **24.04** its `create_venv.sh` auto-selects **Python 3.12**; on 22.04, 3.10 (override with
> `./create_venv.sh --python-version 3.12` if ever needed).

tt-metal builds umd **from source** via its `tt_metal/third_party/umd` submodule (it
`add_subdirectory`s it and links `umd::tt-umd`). ttnn therefore uses this **C++** umd, *not* the
Python `tt_umd` in `.ttvenv` — so to get D2H into ttnn we point that submodule at the fork.

### 5a. Graft the D2H fork into tt-metal (fork-both, reproducible)

**(i) Build the umd fork branch = tt-metal's pinned umd + your D2H commit.** tt-metal pins a
specific umd revision; cherry-picking the single D2H commit onto *that* revision keeps metal's
tested base and adds only the read-DMA delta (the patch applies clean):

```bash
cd ~/tt-umd
UMD_PIN=$(git -C ~/tt-metal/tt_metal/third_party/umd rev-parse HEAD)   # metal's pinned umd sha
git fetch ~/tt-metal/tt_metal/third_party/umd "$UMD_PIN"
git branch d2h-on-metal-pin "$UMD_PIN"
git switch d2h-on-metal-pin
git cherry-pick -n 4039b93e                       # the "D2H DMA unlock" commit
git reset -q -- '*.so' && git checkout -- '*.so' 2>/dev/null  # drop the prebuilt artifact
git commit -m "D2H DMA unlock (cherry-pick 4039b93e onto tt-metal umd pin)"
git switch main
git push origin d2h-on-metal-pin                  # push to your dtsunami/tt-umd fork
```

**(ii) Fork `tenstorrent/tt-metal` → `dtsunami/tt-metal`** (`gh repo fork tenstorrent/tt-metal`
or the GitHub UI), then repoint its umd submodule to the branch above and commit the pointer so
a fresh `--recursive` clone carries it:

```bash
git clone https://github.com/dtsunami/tt-metal.git ~/tt-metal    # your fork
cd ~/tt-metal
git submodule set-url tt_metal/third_party/umd https://github.com/dtsunami/tt-umd.git
git submodule update --init --recursive
git -C tt_metal/third_party/umd fetch origin d2h-on-metal-pin
git -C tt_metal/third_party/umd checkout d2h-on-metal-pin
git add .gitmodules tt_metal/third_party/umd
git commit -m "Point umd submodule at dtsunami/tt-umd d2h-on-metal-pin (D2H DMA)"
git push
```

> **Quick local alternative (no push, no metal fork):** in a plain upstream clone, just
> `cd ~/tt-metal/tt_metal/third_party/umd && git remote add fork https://github.com/dtsunami/tt-umd.git
> && git fetch fork && git cherry-pick 4039b93e` — builds now, but a fresh clone loses it.
> **Maintenance:** when tt-metal bumps its umd pin, re-run 5a(i) against the new pin.

### 5b. Build tt-metal + ttnn into `~/.ttvenv`

```bash
cd ~/tt-metal
git rev-parse --short HEAD          # RECORD this — bleeding-edge main is a moving target
export ARCH_NAME=blackhole TT_METAL_HOME=$HOME/tt-metal
export PYTHON_ENV_DIR=$HOME/.ttvenv # build ttnn + torch INTO the shared venv
./build_metal.sh                    # builds tt-metal + ttnn (with D2H umd) — long
./create_venv.sh                    # installs ttnn + torch into ~/.ttvenv
source ~/.ttvenv/bin/activate && python -c "import ttnn; print('ttnn ok')"
```

`build_metal.sh` already configures against `~/.ttvenv` (its cmake log shows
`-DPython3_EXECUTABLE=~/.ttvenv/bin/python3`), and `create_venv.sh` honours `PYTHON_ENV_DIR`
(default `$TT_METAL_HOME/python_env`) — so ttnn lands in the shared env, not a separate one.

> ⚠️ Watch the `create_venv.sh` pip output: if it **downgrades `rich`/`textual`/`pydantic`**
> (which tt-smi also uses) you may see tt-smi's TUI break. If so, either re-`pip install -e ~/tt-smi`
> to reconcile, or fall back to a separate `python_env` (unset `PYTHON_ENV_DIR`) + §8-legacy `.so` swap.

### 5c. "nn" = ttnn (not a separate repo)

ttnn ships **inside** tt-metal — after §5 it's importable from the shared `~/.ttvenv`. tt-splat's
device path uses it (see [tt-splat-llk-dram-arch]). There is no separate `tt-nn` clone to make.
*(If by "nn" you meant a different repo — e.g. tt-forge / tt-mlir / a models repo — say so and
I'll add a section; nothing on this box used one.)*

### 5d. Local tt-metal additions (restore from preservation)

This box carried **untracked** local work under `tt_metal/programming_examples/contributed/`
and `tests/.../data_movement/` — the resident Tensix **bootloader**, the **gather_scatter_3hop**
kernel, **jit** examples, `agg_bw.py`, and a patched `tt_metal/hw/inc/dataflow_api.h`. These are
**not upstream** and only partly mirrored in bhtop. Restore them from the preservation bundle
(§PRESERVE) after the clone.

---

## 6. bhtop — NoC telemetry + bare-metal het

The bhtop repo was **renamed `dtsunami/tt` → `dtsunami/tt-baremetal`** (this INSTALL.md lives
inside it). Install it editable into the **shared `~/.ttvenv`** — no separate venv, no `.so` swap:

```bash
git clone https://github.com/dtsunami/tt-baremetal.git ~/tt-baremetal   # if not already cloned
cd ~/tt-baremetal
source ~/.ttvenv/bin/activate
pip install -e .                   # tt-exalens, pyluwen, rich, textual, fastapi
```

The fork's DMA binding is already live in `~/.ttvenv` from §4 (no per-venv `.so` copy). Then:

```bash
bhtop            # live NoC NIU-counter telemetry (sanity check the device is reachable)
```

CLI entry points: `bhtop`, `bhtop-inject`, `bhtop-metal`, `bhtop-l2cpu`, `bhtop-tensix`,
`bhtop-web`, `bhtop-kern`.

---

## 7. tt-splat — on-device 3DGS training

tt-splat installs into the **shared `~/.ttvenv`** (torch/ttnn already there from §5 — a plain
`pip install -e .` adds only light deps and never rebuilds ttnn):

```bash
git clone https://github.com/dtsunami/tt-splat.git ~/tt-splat
cd ~/tt-splat
source ~/.ttvenv/bin/activate
pip install -e .                   # typer, pillow, fastapi, etc. — torch/ttnn already present
```

The fork DMA binding is already in `~/.ttvenv` (§4) — no `.so` swap. Smoke test:

```bash
cd ~/tt-splat
ttgs blackhole work/nerf_data/<scene>          # ttnn path
TT_DEVICE_BAREMETAL=1 ttgs blackhole <scene>   # bare-metal het path (bhtop grid_engine)
```

Bare-metal defaults are set in `server/baremetal_resident.py`: `TT_BM_NOCPACE=1`,
`TT_BM_PERF_BUSY=1` (AICLK OC), `TT_BM_AUTORECOVER=1`. See [het-pipeline-noc-wedge] and
[tt-splat-baremetal-campaign] for the tuning knobs.

---

## 8. The `.so` swap — RETIRED in the single-`.ttvenv` layout

With one shared `~/.ttvenv` and `pip install -e ~/tt-umd`, the fork's DMA binding is compiled
**directly into the venv** — the **Python** consumers (tt-smi, bhtop, tt-splat's Python paths)
import the same `tt_umd` from `~/.ttvenv`, so there is nothing to copy. Skip this section unless
you fell back to the multi-venv layout.

> **ttnn is the exception:** ttnn/tt-metal do **not** import the `.ttvenv` `tt_umd` — they link
> the **C++** umd compiled from tt-metal's submodule. ttnn gets D2H only via the §5a submodule
> graft, never from this `.ttvenv` package. Two separate umd builds, two separate wiring steps.

**Verify the DMA path is actually live** (this is the check that lets you trust §8 is gone):

```bash
source ~/.ttvenv/bin/activate
python -c "import tt_umd, os; print(tt_umd.__file__)"   # -> under ~/.ttvenv/.../site-packages
# then run the fork's PCIe DMA microbenchmark / TT_DMA_READBACK=1 path and confirm ~9.5 GB/s,
# NOT ~2.6 MB/s. Slow => the editable build didn't carry the DMA patch; rebuild ~/tt-umd.
```

DMA readback is gated in `grid_engine.py` by `TT_DMA_READBACK=1` (default), which sets
`ctx.use_4B_mode=False`. If the DMA binding is missing it falls back to the slow register path
safely — so a problem shows up as *slow*, not *crashed*. See [bh-exalens-dma-wall].

### §8-legacy — manual `.so` swap (multi-venv fallback only)

Only if you kept tt-metal on its own `python_env` (§5 escape hatch) and bhtop on its own
`.venv`: build the self-contained nanobind `.so` (§3 legacy) and copy it over the stock wheel
in each env that opens the device:

```bash
FORK_SO=~/tt-umd/tt_umd-096-selfcontained.cpython-310-x86_64-linux-gnu.so   # your fork build
cp "$FORK_SO" <env>/lib/python3.10/site-packages/tt_umd/tt_umd.cpython-310-x86_64-linux-gnu.so
# repeat for each separate venv (bhtop/.venv, tt-metal/python_env, ...)
```

---

## 9. Verification checklist

```bash
modinfo tenstorrent | grep ^version                 # 2.8.0
lspci -nn | grep 1e52                                # b140 present
source ~/.ttvenv/bin/activate                        # the one env for everything
tt-smi --version                                     # 5.3.1
tt-smi -s                                            # telemetry snapshot
python -c "import tt_umd, ttnn; print('umd+ttnn ok')" # both in ~/.ttvenv
bhtop                                                # NoC counters render
TT_DEVICE_BAREMETAL=1 ttgs blackhole <scene>         # het step runs
```

---

## 10. Gotchas index (hard-won — read before you burn a day)

- **PCIe slot:** run the card in a **Gen4** slot. A marginal Gen5 x16 link throws correctable
  RxErrors under heavy het traffic → **hard host freeze** (no catchable exception). The step is
  device-bound, so Gen4 bandwidth costs ~nothing. [het-pipeline-noc-wedge]
- **`tt-smi -r 0` →** works clean on **5.3.1** (reset + reinit, exit 0). The `--no_reinit`
  workaround was only for the old 4.1.2 combo (§4).
- **fork DMA binding:** with the single `~/.ttvenv` + editable `tt-umd`, it's compiled into the
  venv — no `.so` swap. After any `pip install`/reinstall that pulls **stock** `tt_umd`, re-run
  `pip install -e ~/tt-umd` to restore the fork, else you silently drop to 2.6 MB/s (§8).
- **Hugepages:** needed before **tt-metal's** device open (1 GB pool), or UMD init fails there.
  Note **tt-smi telemetry + `-r 0` reset work *without* hugepages** — don't take SMI succeeding
  as proof the pool is up; check `nr_hugepages` explicitly (§2).
- **het NoC wedge:** the 4-hart NIU race is silicon-intrinsic; survive it with
  `TT_BM_AUTORECOVER=1` + checkpoint every ~500–1000 steps; each `tt-smi -r 0` slightly wears
  the card, so a very long campaign benefits from a periodic process restart. [het-pipeline-noc-wedge]
- **NoC0-hang hazard:** never poke ARC/Security/PCIe/L2CPU register windows over NoC0 — it hangs
  the tile. Recover with `tt-smi -r 0`. [bh-noc-hang-hazard]

---

## PRESERVE — before wiping this box

Uncommitted/at-risk work that must be pushed or bundled first (see the session report):

1. **tt-umd D2H DMA** — commit `4039b93e` is on `dtsunami/tt-umd` **main** (pushed). Two
   follow-ups for the tt-metal path: **push the `d2h-on-metal-pin` branch** (`git push origin
   d2h-on-metal-pin`) and **fork tt-metal** with its umd submodule repointed at it (§5a). Until
   both are pushed, ttnn's D2H lives only in this box's local git. **Highest priority.**
2. **tt-metal local additions** — bootloader / gather_scatter_3hop / jit / `agg_bw.py` /
   `dataflow_api.h`. `origin` is upstream tenstorrent (can't push) and they're only partly in
   bhtop → bundle (`git format-patch` or copy into a dtsunami repo) before wipe.
3. **comfy/tt-metal** — `models/tt_dit/layers/lora.py` (untracked, minor).
4. bhtop, tt-splat, arcgs — already clean & pushed to `dtsunami/*`. Safe.
