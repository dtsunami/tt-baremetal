# L2CPU Copilot — plan & handoff

Live bring-up + code loader for the Blackhole **L2CPU (SiFive x280)** harts, built into bhtop.
Goal: compile asm/C/Rust and run it on any L2CPU hart, **iterate live with no reset**, with easy
telemetry — plus a path to full Linux + SSH. This doc is the handoff for a fresh session.

Related memory: `[[l2cpu-bootstrap]]` (full play-by-play + every address), `[[bh-noc-hang-hazard]]`,
`[[project-tt-noc-top]]`, `[[bhtop-next-session]]`. Host: **ttstar**, repo `~/bhtop`, venv `~/bhtop/.venv`,
node `~/.local/node/bin`, tt-metal `~/tt-metal`.

---

## Status — what works (all proven on silicon unless noted)

1. **Bring-up** ✅ — release a tile's 4 harts from reset (ARC PLL glide → `L2CPU_RESET` flip → raise clock),
   the tt-bh-linux sequence. Ran tile (8,3): harts out of reset executing our bare-metal.
2. **Live code swap** ✅ — RNMI "seize": redirect a *running* hart to new DRAM code with no reset,
   repeatable (self-re-arming trampoline). Demoed v1→v2 (counter froze, new counter climbed).
3. **Real loader package** ✅ (built + **verified end-to-end on silicon through the CLI**, 2026-06-14)
   — `src/bhtop/l2cpu/`: compile asm/C/Rust → load → redirect on any tile/hart, interactive CLI, telemetry.
   Proven on tile 0: `bringup` → `load counter.c` → live-redirect to a 2nd kernel (old counter froze, new
   `0xDEADBEEF`/`0xCAFEF00D` signatures appeared, heartbeat kept climbing, no reset) → multi-hart (hart 0 +
   hart 1 running distinct kernels concurrently). **Found + fixed a redirect bug in the process — see below.**
4. **Linux + SSH** ✅ verified feasible (tt-bh-linux `make boot` → `make ssh`, out of the box). See
   `src/bhtop/l2cpu/LINUX.md`.

Current device state is whatever the last `tt-smi -r 0` left — assume harts in reset until a `bringup`.

---

## The package: `src/bhtop/l2cpu/`

| file | role |
|---|---|
| `__init__.py` | `L2cpu` controller — bringup / load / redirect / telemetry; tt-exalens (NoC) + pyluwen (ARC); guards + canary |
| `toolchain.py` | compile asm/C/Rust → flat u32 words at a base (sfpi gcc + `rt/link.ld` + `rt/crt0.s` + objcopy) |
| `cli.py` | `bhtop-l2cpu` — REPL + one-shot: tiles/bringup/load/tele/status/peek/poke/disasm/examples |
| `rt/link.ld`, `rt/crt0.s` | flat-image link + C startup (so a C kernel is just `int main()`) |
| `include/tele.h` | telemetry: `TELE[0]=++hb;` host reads with `tele` |
| `examples/{heartbeat.s,counter.c,blink.rs}` | asm / C / Rust starters |
| `README.md`, `LINUX.md` | usage + doc links; Linux/SSH writeup |

Run: `python -m bhtop.l2cpu …` now, or `pip install -e .` once → `bhtop-l2cpu`. (pyluwen already a dep.)

### Quickstart
```sh
tt-smi -r 0                                  # clean slate (optional)
python -m bhtop.l2cpu bringup 0              # one-shot: release tile 0, install trampoline, park harts
python -m bhtop.l2cpu load 0 0 src/bhtop/l2cpu/examples/counter.c   # compile+load+run, hart 0
python -m bhtop.l2cpu tele 0                 # watch telemetry (slot 0 = heartbeat)
python -m bhtop.l2cpu load 0 0 my.c          # edit & reload — redirected live, no reset
```

### Standalone scripts (proven the mechanism step by step; superseded by the package but kept)
`scripts/{probe_l2cpu_rnmi.py, l2cpu_xport_diag.py, l2cpu_dram_probe.py, l2cpu_bringup.py, l2cpu_redirect.py}`.

---

## Hardware facts (don't re-derive these)

- **Tiles** (index → noc0 coord → `L2CPU_RESET` bit): `0→(8,3) bit4`, `1→(8,9) bit5`, `2→(8,5) bit6`, `3→(8,7) bit7`. 4 harts each.
- **Transport (critical):** L2CPU NoC via **tt-exalens** (`read/write_words_from/to_device`, `safe_mode=False`,
  explicit `noc_id=0`); ARC via **pyluwen** `axi_*` — a *separate* transport that avoids the NoC0-hang hazard.
  Init tt-exalens FIRST (it sets up NoC; standalone pyluwen NoC returned all-ones and a write wedged NoC0).
  pyluwen `(8,3)` == tt-exalens noc0 (8,3) (translated==noc0 here); the earlier wedge was uninitialized NoC, not coords.
- **L2CPU regs** (x280 phys, hart N): reset-vec `0x20010000+N*8`; RNMI trap `0x20010418+N*16`, exc `0x20010420+N*16`;
  RNMI trigger `0x20010414` bit N; hart status `0x20010400`; scratch `0x20010100` (64B, executable).
- **DRAM** (uncached, per tile, writable — probed): trampoline `0x30000000`, code `0x30001000`, telemetry `0x30002000`.
- **ARC**: PLL4 `0x80020500` (CNTL_1 `+0x4`, CNTL_5 `+0x14`); `L2CPU_RESET` `0x80030014`. PLL solutions
  `{200:[128,[15]*4], 1750:[140,[1]*4]}`; glide = raise postdivs, step fbdiv, lower postdivs (±1, 1ns).
- **Self-re-arming trampoline** (sfpi-assembled): `lui t0,0x20010; sw zero,0x414(t0); csrsi 0x353,8(NMIE); fence.i;
  ld t1,0(t0); jr t1` = `[0x200102B7,0x4002AA23,0x35346073,0x0000100F,0x0002B303,0x00030067]`.
- **Toolchain:** sfpi `~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-*` (GCC 15.1.0) for asm/C (present).
  Rust needs `rustup target add riscv64gc-unknown-none-elf` (rustup not installed). No capstone/objdump on PATH;
  use sfpi objdump.

### SAFETY (hard rules)
- **`bringup` / reset-release is ONE-SHOT** (reset-once silicon bug): re-do requires `tt-smi -r 0`.
- **L2CPU is in the NoC0-hang zone.** Always: address-guard (passthrough window only / ARC allowlist),
  `safe_mode=False` + explicit `noc_id`, **canary read before any write**, abort on all-ones reads, never
  interpret all-ones as a valid value. If wedged: `tt-smi -r 0`.

---

## Linux + SSH (verified, see LINUX.md)
tt-bh-linux boots Debian on the harts; `console/tt-bh-linux` host runner emulates virtio over PCIe/NoC
(console = OpenSBI virtual UART `console=hvc0 earlycon=sbi`; net = virtio-net + **libslirp userspace NAT**,
no root). Guest DHCPs `10.0.2.15`; host forwards `localhost:(2222+l2cpu+4*ttdev)→22`; default rootfs has
`openssh-server` + empty-pw `debian` user. So **`make boot` → `make ssh` works out of the box** (full rootfs,
not the cloud-init image). `bhtop-l2cpu` = bare-metal iterate-live lane; tt-bh-linux = full-shell lane; same
ARC/reset mechanism, coexist on different tiles.

---

## Bug found + fixed during end-to-end bring-up (2026-06-14)
**Symptom:** `load` reported "running ✓" but the redirect silently did nothing — the hart kept running the
*previous* kernel. Root cause: `redirect` assumed the tile's trampoline was OUR self-re-arming one, but it
was only installed by `bringup`. A tile brought up by a *standalone script* (or any non-package path) has a
trampoline that does **not** restore `mnstatus.NMIE`. The first package redirect bounces through it into the
loaded code with NMIE=0, which **masks all further RNMIs** + leaves stale code in the i-cache; every later
redirect is ignored, yet "running ✓" stayed green because the stale kernel keeps incrementing the heartbeat.
**Fix** (`src/bhtop/l2cpu/__init__.py`): (a) `redirect` now `_install_trampoline()` (idempotent re-stamp of
OUR trampoline + RNMI handlers) every time, so a load never depends on how the tile was brought up; (b) real
**seize verification** — our trampoline clears the trigger bit on seize, so `redirect` polls it and raises a
clear `tt-smi -r 0` error if the bit stays set instead of falsely reporting success; (c) CLI prints
"loaded + seized hart N ✓" only on a verified seize. Also write the trigger as a bare `1<<hart` (matches the
proven standalone path) instead of read-modify-OR. **Lesson: heartbeat-climbing ≠ your new code is running.**

## Next / open tasks (pick up here)
1. ✅ **DONE — package verified end-to-end on chip** (`bringup 0` → `load counter.c` → live-redirect → tele);
   bug above found + fixed in the process. (Standalone scripts proved the HW; the package now proven via CLI.)
2. ✅ **DONE — Rust path verified on silicon** (rustup + `riscv64gc-unknown-none-elf` installed 2026-06-14):
   `load 0 2 examples/hwinfo.rs` compiled, seized hart 2, read `bh_hartid()`→2 + cycle CSRs via telemetry.
3. ✅ **DONE — multi-hart + multi-tile proven**: hart 0 + hart 1 distinct kernels on tile 0; tile 1 (noc0 (8,9))
   brought up and ran the harness independently (`regs 1` confirms). Tiles 2/3 untouched (bringup each once when needed).

### Newbie harness + docs (added 2026-06-14) — "register maps etc, in the bhtop spirit"
- **`regmap.py`** — ONE canonical model of the L2CPU env (regions, registers, CSRs, tiles, ARC). The loader,
  the harness, the docs, and the `map`/`regs` CLI all read it (no second copy to drift). `__init__.py` now
  imports its scalar addresses from here.
- **Harness in every language**: `include/bh.h` (C), `rt/bh.rs` (Rust, owns `_start`+panic so you write only
  `kmain`), `include/bh.inc` (asm). Named registers + `bh_hartid()`/`bh_cycles()`/`bh_rd32`/`TELE[]`. Rust pulls
  it in via `include!(concat!(env!("BH_RT"), "/bh.rs"))` — toolchain.py sets `BH_RT`; asm via `-Wa,-I` + `.include`.
- **CLI inspector**: `bhtop-l2cpu map` (static annotated map, no chip) + `regs <t> [hart]` (live, decoded
  HART_STATUS/TRIGGER/per-hart vectors). The "see your hardware live" bit.
- **Docs**: `HARDWARE.md` — newcomer's tour of Blackhole physical arch → L2CPU tile → memory/register map →
  how a kernel runs (RNMI) → safety. README updated. Examples `hwinfo.c`/`hwinfo.rs` read real CPU registers.
4. **Telemetry → Kernel Lab web UI**: surface the L2CPU telemetry block + load/redirect controls in the bhtop
   web cockpit (`src/bhtop/web/` + `frontend/`), alongside the NoC heatmap.
5. **C ergonomics**: optional libc-lite (memcpy/memset) + a bigger example (e.g. walk DRAM, compute, report);
   consider cached-GDDR code (`0x4000_3000_0000`) for perf (trampoline already uses `ld` for 64-bit addrs).
6. **Linux convenience**: a bhtop wrapper to `make boot`/`make ssh` a tile, and copy the user's SSH key in.
7. **Hardening**: confirm whether anything reset the harts between runs earlier (likely a manual `tt-smi -r 0`);
   the `l2cpu_dram_probe.py` touches `0xD0000000` — keep an eye on it.

---

## Verification done (no chip)
sfpi compiles `heartbeat.s`→5 words & `counter.c`→20 words with `_start` (crt0) at load base `0x30001000`;
package imports + CLI (`examples`/`help`) work; `have_rust` correctly False; all stub/trampoline blobs
decode-verified against the assembler. Device path = same logic the standalone scripts ran successfully.
