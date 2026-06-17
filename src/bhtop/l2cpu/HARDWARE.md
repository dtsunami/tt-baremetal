# The hardware, for newcomers — Blackhole, the L2CPU, and how your kernel runs

This is the "where am I and what am I poking" guide. If you've never touched this
silicon, read this once and the rest of `bhtop-l2cpu` will make sense. The bhtop way:
**the chip is just memory-mapped registers — here's the map, go poke it and watch.**

Two commands give you the same map live:

```sh
bhtop-l2cpu map          # the whole register/memory map, annotated (no chip needed)
bhtop-l2cpu regs 0       # read tile 0's control registers RIGHT NOW, decoded
```

Everything below is also in [regmap.py](regmap.py) (the one canonical copy the loader,
the C/Rust/asm harness, and `map`/`regs` all read).

---

## 1. What is Blackhole?

Blackhole is a Tenstorrent AI accelerator — a big grid of small **tiles** wired together
by an on-chip **NoC** (Network-on-Chip). You talk to the whole thing from the host over
PCIe. Think of it as a city laid out on a grid:

```
        NoC grid (every tile has an (x,y) coordinate)
   x→ 0   1   2   ...        8   ...
 y ┌───┬───┬───┬───┐     ┌───────────┐
 0 │ T │ T │ D │   │ ... │           │   T = Tensix (the AI compute tiles)
 1 │ T │ T │   │   │     │           │   D = DRAM (off-chip GDDR controllers)
 2 │ T │ T │ D │   │     │           │   A = ARC (the management microcontroller)
 . │   │   │   │   │     │           │   P = PCIe (your link from the host)
 . │           x=8 →  L2CPU column   │   L = L2CPU  ← THIS is what we use
   └───┴───┴───┴───┘     └───────────┘
```

A **NoC address** is "(x, y) tile + an offset inside it". The host reads/writes any tile
by sending NoC packets. bhtop's whole job is making that traffic visible; this loader
uses it to talk to one special kind of tile.

There are actually two NoCs (NoC0 and NoC1) overlaid on the same grid. We pin everything
to **NoC0** on purpose (mixing them, or poking the wrong window, can hang the link — see
§6). The main `bhtop` cockpit visualizes this grid and its live traffic.

---

## 2. The L2CPU tile = four real Linux-capable CPUs

Most tiles do AI math. The **L2CPU** tile is different: it's a cluster of **four SiFive
x280 RISC-V cores** ("harts" = hardware threads). These are full **RV64GC** application
processors — 64-bit, with the general-purpose + compressed + float extensions. They can
run bare-metal code (what we do here) or even **boot Linux** (see [LINUX.md](LINUX.md)).

A Blackhole has **four L2CPU tiles**, so 16 harts total:

| tile | NoC0 coord | `L2CPU_RESET` bit |
|------|-----------|-------------------|
| 0    | (8, 3)    | bit 4 |
| 1    | (8, 9)    | bit 5 |
| 2    | (8, 5)    | bit 6 |
| 3    | (8, 7)    | bit 7 |

`bhtop-l2cpu tiles` lists them and whether each is in reset.

**The one silicon gotcha:** a tile's harts can be brought out of reset **only once**.
Redoing bring-up needs a full ASIC reset (`tt-smi -r 0`). *Redirecting* an
already-running hart to new code (what `load` does) is unlimited — that's the whole point
of the RNMI trick in §5.

---

## 3. The memory map — where things live inside a tile

Every address your kernel uses is an x280 **physical** address, which (in the low window)
equals the NoC offset 1:1. Three regions matter:

```
  0x20010000  ┌─────────────────────────────┐  PERIPHERAL registers (the hart knobs)
              │ reset vectors, RNMI, status │  — boot/park/seize (§4); cmd mailbox @0x20010100
  0x20020000  └─────────────────────────────┘
                          ...
  0x30000000  ┌─────────────────────────────┐  DRAM (uncached off-chip GDDR)
              │ 0x30000000  trampoline      │  ← installed by `bringup` (+ park spin @0x30000020)
              │ 0x30002000  telemetry       │  ← you write, host reads (`tele`); per-hart
              │ 0x30003000  arch-state      │  ← bh_dump_state() snapshots GPRs + CSRs here
              │ 0x30005000  vector-state    │  ← bh_dump_vec() snapshots v0..v31 + vec CSRs
              │ 0x30008000  YOUR CODE       │  ← `load` drops your kernel here, hart runs it
              └─────────────────────────────┘
```

**Per-hart windows.** Both telemetry and arch-state are *per-hart*: hart N's telemetry is
at `0x30002000 + N*0x100`, its register dump at `0x30003000 + N*0x200`. The harness targets
the running hart automatically, so kernels on different harts never clobber each other's
slots — and the cockpit shows each hart separately.

**Seeing the registers.** The host can't read a hart's GPRs/CSRs over the NoC (no debug
halt on these tiles). So `bh_dump_state()` makes the hart write its own register file to
its arch-state block; the cockpit's **Arch** tab decodes it (all 32 registers + CSRs, with
hover tooltips). See `examples/dumpstate.c`.

Because this DRAM window is **uncached**, values your kernel writes (like telemetry) are
visible to the host immediately — no cache flush. Your *instructions* do go through the
I-cache, but the loader's trampoline runs a `fence.i` for you on every redirect, so you
never have to think about cache coherency. (`bhtop-l2cpu map` prints the full region list,
including the **danger** window you must never poke — §6.)

---

## 4. The control registers — how a hart is steered

These live in the peripheral block at `0x2001_0000`. The ones with `+N` repeat for each
hart N. You rarely write these by hand (the loader does), but reading them — via
`bhtop-l2cpu regs <tile>` or `peek` — is how you see what state a hart is in.

| register | address | what it does |
|----------|---------|--------------|
| `RESET_VEC` | `0x20010000 + N*8` | hart N's **initial PC** — where it starts/restarts. `load` sets this to your code. |
| `HART_STATUS` | `0x20010400` | run/halt/wfi status, 4 bits per hart. `0x0000` = all parked. |
| `TRIGGER` | `0x20010414` | write `1<<N` to fire an **RNMI** on hart N (the "seize"). |
| `RNMI_TRAP` | `0x20010418 + N*16` | where hart N jumps when its RNMI fires (→ the trampoline). |
| `RNMI_EXC` | `0x20010420 + N*16` | where hart N jumps on a fault (→ a safe spin). |

There's also a second class of registers, **CSRs** (Control & Status Registers), that are
part of the RISC-V core itself. The host *cannot* read these over the NoC — only code
**running on the hart** can, with `csrr`/`csrw`. The harness wraps the useful ones:

| CSR | harness call | what it is |
|-----|--------------|------------|
| `mhartid` | `bh_hartid()` | which hart am I (0..3) |
| `mcycle` | `bh_cycles()` | free-running cycle counter — a cheap timer |
| `minstret` | `bh_instret()` | instructions retired |
| `mnstatus` | (used by trampoline) | bit 3 `NMIE` gates RNMI delivery (see §5) |

---

## 5. How `load` actually runs your code (the RNMI trick)

The clever part. After `bringup`, every hart sits parked in a tiny spin loop, and an
**RNMI redirect trampoline** is installed at `0x30000000`. To run new code with no reset:

```
  host: bhtop-l2cpu load 0 0 mykernel.c
    1. compile mykernel.c → flat image, drop it at 0x30008000 (DRAM: user code)
    2. set RESET_VEC[hart] = 0x30008000
    3. write TRIGGER bit hart  ──fires a Resumable NMI──▶  hart jumps to the trampoline:
                                                            clear TRIGGER
                                                            set mnstatus.NMIE = 1   (re-arm)
                                                            fence.i                 (sync I-cache)
                                                            load RESET_VEC, jump  ──▶ YOUR CODE
```

Because the trampoline **re-arms** (re-enables `NMIE` and clears the trigger), the *next*
`load` works exactly the same way — you iterate freely, no reset between tries. The loader
confirms the seize landed by checking that `TRIGGER` cleared; if it didn't, it says so
instead of pretending success. (`heartbeat climbing` alone does **not** prove your new
code is running — a stale kernel can keep a counter going; the trigger-cleared check is
the real proof.)

---

## 6. Safety — the two rules that keep the chip alive

1. **Bring-up is one-shot.** Releasing a tile's harts can't be undone without `tt-smi -r 0`.
   `bringup` refuses to re-release and tells you. Redirecting running harts (`load`) is free.
2. **Stay in the safe window.** The low passthrough window (`0x0..0x7FFF_FFFF_FFFF`) is the
   intended access path. The **NIU config window at `0xFFFF_FFFF_FF00_0000`** will **hang
   NoC0** if you read or write it. The loader address-guards every op and reads a canary
   first; if a read ever comes back all-ones, it aborts and tells you to `tt-smi -r 0`.
   ARC (clock/reset) is reached over a *separate* transport (pyluwen `axi_*`), never the NoC.

If the link ever wedges: `tt-smi -r 0`, then `bringup` again. Nothing you can type into a
kernel can brick the board — worst case is a reset.

---

## 7. Your turn — a 60-second loop

```sh
bhtop-l2cpu bringup 0                                  # once per power cycle
bhtop-l2cpu load 0 0 src/bhtop/l2cpu/examples/hwinfo.c # read real CPU registers
bhtop-l2cpu tele 0                                     # slot1=hartid, slot2=cycles, ...
bhtop-l2cpu regs 0                                     # see the hart's control regs live
# edit hwinfo.c, load again — redirected live, no reset
```

Write a kernel as just `int main(void)` in C (`#include <bh.h>`), `fn kmain()` in Rust
(`include!(concat!(env!("BH_RT"), "/bh.rs"))`), or a `_start` in assembly
(`.include "bh.inc"`). The harness gives you named registers and `bh_hartid()` /
`bh_cycles()` / `TELE[]` in every language. See [README.md](README.md) for the kernel-writing
details and [the examples](examples/).
