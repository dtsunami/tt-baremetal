# bhtop-l2cpu — live code loader for the Blackhole L2CPU (SiFive x280)

Bring the L2CPU harts out of reset and **load/compile/run/redirect your own bare-metal
code on them — live, repeatably, no reset between iterations.** Write a kernel in
assembly, C, or Rust; the loader compiles it, drops it into the tile's DRAM, and seizes
a running hart over to it via an RNMI. A tiny telemetry block lets the hart surface
values back to the host.

> **New to this silicon?** Read [HARDWARE.md](HARDWARE.md) first — a newcomer's tour of
> Blackhole, the L2CPU tile, the memory/register map, and how your kernel actually runs.
> Then `bhtop-l2cpu map` (the whole register map, no chip needed) and `bhtop-l2cpu regs 0`
> (read a tile's control registers live). The hardware harness (`bh.h` / `bh.rs` /
> `bh.inc`) gives you named registers and `bh_hartid()`/`bh_cycles()`/`TELE[]` in every
> language — see *Writing a kernel* below.

> **Reset-once silicon bug:** each L2CPU tile's 4 harts can be brought out of reset
> only ONCE. To re-do bringup you must reset the whole ASIC: `tt-smi -r 0`. *Redirecting*
> already-running harts (what `load` does after `bringup`) is unlimited and needs no reset.

## Hardware/transport notes
- **L2CPU NoC** access (DRAM, peripherals) goes through **tt-exalens**; **ARC** (PLL +
  reset) through **pyluwen `axi_*`** — a separate transport that avoids the NoC0-hang
  hazard of poking ARC over the NoC. tt-exalens is initialized first.
- Every device op is address-guarded (L2CPU writes restricted to the safe x280
  passthrough window; ARC to the 3 PLL/reset regs), with a canary read before writes.
  Any device error aborts with a `tt-smi -r 0` hint.

---

## Install / configure

1. **RISC-V C/asm toolchain — already present.** It ships with tt-metal:
   `~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-{gcc,as,ld,objcopy}` (GCC 15.1.0).
   Nothing to install. (If `TT_METAL_HOME` is elsewhere, the loader looks under
   `~/tt-metal`; adjust `toolchain.SFPI` if needed.)
2. **pyluwen** — already a bhtop dependency (used for ARC). Confirm: `python -c "import pyluwen"`.
3. **Register the CLI** (optional): `pip install -e .` in the repo gives you `bhtop-l2cpu`.
   Otherwise run `python -m bhtop.l2cpu …` — identical.
4. **Rust toolchain (only if you want Rust kernels)** — not installed by default:
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   . "$HOME/.cargo/env"
   rustup target add riscv64gc-unknown-none-elf   # = RV64GC, lp64d, no_std (exactly x280)
   ```
   The loader detects this automatically; until then C/asm work and Rust prints this hint.

---

## Quickstart

```sh
# (optional) clean slate
tt-smi -r 0

# bring tile 0's harts out of reset (ONE-SHOT) — installs the redirect trampoline,
# parks all 4 harts. (tile 0 = noc0 (8,3); tiles 1/2/3 = (8,9)/(8,5)/(8,7))
bhtop-l2cpu bringup 0

# compile + load + run a C kernel on tile 0, hart 0, then watch its telemetry
bhtop-l2cpu load 0 0 src/bhtop/l2cpu/examples/counter.c
bhtop-l2cpu tele 0

# iterate: edit the source and load again — the hart is redirected live, no reset
bhtop-l2cpu load 0 0 my_kernel.c
```

Or interactively:
```
$ bhtop-l2cpu
l2cpu> tiles
l2cpu> bringup 0
l2cpu> load 0 0 counter.c
l2cpu> tele 0
l2cpu> quit
```

### Commands
| command | does |
|---|---|
| `tiles` | list the 4 L2CPU tiles + reset state |
| `status <t>` | reset state, hart status, per-hart reset vectors |
| `bringup <t> [--yes]` | release tile `t`'s harts (one-shot), install trampoline, park harts |
| `load <t> <hart> <file> [--addr 0x..] [--lang asm\|c\|rust]` | compile + load + redirect a hart, live |
| `tele <t> [n]` | read `n` telemetry slots (slot 0 = heartbeat convention) |
| `peek <t> <addr> [n]` / `poke <t> <addr> <val>` | raw L2CPU reads/writes |
| `map` | print the annotated register/memory map (no device needed) |
| `regs <t> [hart]` | read + decode a tile's live hart-control registers |
| `disasm <file> [--addr 0x..]` | compile and show the disassembly (no device needed) |
| `examples` | list the bundled example sources |
| `reset` | reminder of the `tt-smi -r 0` recovery command |

---

## Telemetry — the easy way to see what your hart is doing

Hart code writes 32-bit slots to a fixed DRAM block (`0x30002000`, 64 slots); the host
reads them with `tele`. No caches involved, so writes are visible immediately.

**C** (`#include <tele.h>` — on the include path automatically):
```c
#include <tele.h>
int main(void){ unsigned i=0; for(;;){ TELE[0]=++i; TELE[1]=i*i; } }
```
**asm:** `lui t0,0x30002; … sw <val>, 0(t0)` (slot 0), `sw <val>, 4(t0)` (slot 1), …
**Rust:** `write_volatile(0x3000_2000 as *mut u32, v)`.

Convention: **slot 0 = a monotonically increasing heartbeat** (liveness); slots 1–63 are
yours. `bhtop-l2cpu tele <t>` prints non-zero slots and labels slot 0.

---

## Writing a kernel

The **hardware harness** gives you named registers and helpers in every language, so you
don't sprinkle magic addresses through your code. It mirrors [regmap.py](regmap.py) (the
one canonical map) — see [HARDWARE.md](HARDWARE.md) for what each name means.

- **C** — just write `int main(void)`; `rt/crt0.s` provides `_start` (sets the stack,
  zeroes `.bss`, calls `main`, parks). `#include <bh.h>` for the full harness —
  `bh_hartid()`, `bh_cycles()`, `bh_rd32()/bh_wr32()`, `BH_RESET_VEC(n)`, and `TELE[]`.
  (`#include <tele.h>` alone if you only want telemetry.) See `examples/hwinfo.c`.
- **Rust** — `#![no_std] #![no_main]` then `include!(concat!(env!("BH_RT"), "/bh.rs"));`
  and write only `extern "C" fn kmain() -> !` — the harness owns `_start` + the
  `#[panic_handler]` and gives you `bh_hartid()`/`bh_cycles()`/`bh_tele()`. See
  `examples/hwinfo.rs`. (Or do it all from scratch like `examples/blink.rs`.)
- **assembly** — provide your own `_start` in section `.text._start` (so it lands at the
  load address); `.include "bh.inc"` for named addresses (`BH_TELE_BASE`, …). See
  `examples/heartbeat.s`.

All three are linked at the **load address** (default `0x30001000`, override with `--addr`)
by `rt/link.ld`, which puts `.text._start` first, packs rodata/data, and reserves bss+stack.
`.bss` is not in the flat image, so it's zeroed at startup (crt0 for C; do it yourself in
asm/Rust if you use bss).

Examples: `examples/{heartbeat.s, counter.c, blink.rs}` (from-scratch) and
`examples/{hwinfo.c, hwinfo.rs}` (using the harness — read real CPU registers).

---

## Memory map (per tile)

| region | address | notes |
|---|---|---|
| RNMI redirect trampoline | DRAM `0x30000000` | installed by `bringup`; self-re-arming |
| **user code (load addr)** | DRAM `0x30001000` | default; `--addr` to change |
| telemetry block | DRAM `0x30002000` | 64 × u32 |
| per-hart reset vector | `0x20010000 + hart*8` | initial pc (set by load → trampoline jumps here) |
| RNMI trap / exc handler | `0x20010418 / 0x20010420 + hart*16` | trap→trampoline, exc→safe spin |
| RNMI trigger | `0x20010414` bit `hart` | pull to seize that hart |
| ARC PLL4 / L2CPU_RESET | `0x80020500 / 0x80030014` | via pyluwen `axi_*` |

Tiles: `0`→noc0 `(8,3)` bit4, `1`→`(8,9)` bit5, `2`→`(8,5)` bit6, `3`→`(8,7)` bit7.

How `load` works: writes your image to DRAM, sets the hart's reset-vector to it, and pulls
the RNMI trigger. The hart traps into the trampoline (`clear trigger; NMIE=1; fence.i; ld
reset-vec; jr`) and jumps into your code. Because the trampoline re-enables NMIE and clears
the trigger, the next `load` works the same way — **iterate freely.**

---

## Linux + SSH on the x280

The harts are full RV64GC application cores and **can boot Linux** — that's what
[tt-bh-linux](https://github.com/tenstorrent-riscv-software/tt-bh-linux) does (loads
OpenSBI→kernel→DTB into the tile's DRAM and releases the harts). See `LINUX.md` for the
feasibility writeup on **SSH-ing into that Linux over the PCIe link** and the concrete steps.
(This loader handles bare-metal kernels; full Linux bringup is tt-bh-linux's job — the two
share the same reset/clock mechanism.)

---

## Toolchain & ISA documentation

- **RISC-V ISA specs** (unprivileged + privileged): https://riscv.org/technical/specifications/
- **RISC-V psABI** (lp64d calling convention): https://github.com/riscv-non-isa/riscv-elf-psabi-doc
- **RISC-V assembly manual**: https://github.com/riscv-non-isa/riscv-asm-manual
- **Instruction quick reference**: https://msyksphinz-self.github.io/riscv-isadoc/html/index.html
- **SiFive X280**: https://www.sifive.com/cores/intelligence-x280 · docs: https://www.sifive.com/documentation
- **GNU as / ld / objcopy**: https://sourceware.org/binutils/docs/as/ · https://sourceware.org/binutils/docs/ld/ · linker scripts: https://sourceware.org/binutils/docs/ld/Scripts.html
- **Freestanding C** (`-ffreestanding`): https://gcc.gnu.org/onlinedocs/gcc/Standards.html · RISC-V GCC options: https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html
- **Bare-metal Rust**: Embedded Rust Book https://docs.rust-embedded.org/book/ · Embedonomicon https://docs.rust-embedded.org/embedonomicon/ · `riscv`/`riscv-rt` https://github.com/rust-embedded/riscv · target https://doc.rust-lang.org/rustc/platform-support/riscv64gc-unknown-none-elf.html
- **Blackhole L2CPU ISA** (reset, RNMIs, memory map, caches): https://github.com/tenstorrent/tt-isa-documentation/tree/main/BlackholeA0/L2CPUTile — also browsable in the bhtop **Kernel Lab** docs pane.
