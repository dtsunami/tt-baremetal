# Resident bootloader — deploy & adjust kernels on the fly

A resident "bootloader" kernel, multicast by tt-metal to **every** Tensix worker, that never
returns: it polls a fixed L1 control mailbox and, on command, applies live params or **jumps into
host-staged machine code** in an L1 code slot. tt-metal does the bring-up (device open, NoC + Tensix
backend init, multicast load, initial `go`); after that the host **parks** and `bhtop` drives the
grid live over tt-exalens — no re-JIT-launch, no soft-reset, no teardown.

This is metal's own firmware mechanism generalized: firmware casts an L1 address to a function
pointer and calls it ([brisc.cc:517](../../../hw/firmware/src/tt-1xx/brisc.cc#L517)), after a
hardware i-cache invalidate ([brisc.cc:239](../../../hw/firmware/src/tt-1xx/brisc.cc#L239)). We put
that one call in a loop you control. It's the x280 RNMI code-swap, minus RNMI — cooperative polling.

## The division of labor

| tt-metal (one-time, then parks)        | bhtop / exalens (live)                          |
|----------------------------------------|-------------------------------------------------|
| open device, NoC + backend init        | poke params (multicast = whole grid in 1 write) |
| multicast bootloader to all cores      | stage `.text` into a code slot                  |
| initial `go`; hold device open         | ring doorbell → invalidate i$ → call the code   |
|                                        | read heartbeat / telemetry                      |

**JIT is demoted from lifecycle-owner to code generator.** New machine code still needs the
riscv32 compiler (a core runs instructions) — but compiling no longer resets/relaunches anything.
Compile a snippet → stage bytes → ring doorbell.

## Build the host launcher

```bash
cd ~/tt-metal
# adds the `bootloader` target alongside vecadd (CMakeLists already wired)
cmake --build build_Release --target programming_examples   # or your usual examples build
```

## Run (slow dispatch recommended for resident kernels)

```bash
export TT_METAL_HOME=~/tt-metal        # REQUIRED: binary path AND the JIT kernel-source lookup
unset TT_METAL_WATCHER                  # MUST unset, not set-empty: presence enables it (interval 0 = spam),
                                        # and it would flag a never-finishing kernel as "hung"
export TT_METAL_SLOW_DISPATCH_MODE=1    # slow dispatch: LaunchProgram fire-and-forget (recommended)
$TT_METAL_HOME/build_Release/programming_examples/contributed/bootloader
# -> "Deploying resident bootloader to NxN ...", "Launched via slow dispatch ...",
#    "Bootloader resident on the grid. Control mailbox @ L1 0x140000 per core."
# leave it running (it parks); Ctrl-C halts + closes.
#
# NOTE: if TT_METAL_HOME is unset, the binary exits 127 and NOTHING deploys — but bl-status still
# "works" because it reads zeroed L1 (decodes as status=BOOT, hb=0). A real run shows status=IDLE
# with the heartbeat advancing. BOOT/hb=0 on every core == the launcher didn't actually run.
```

## L1 contract (fixed addresses — see `bootloader_abi.h`)

| Region        | Addr        | Purpose                                  |
|---------------|-------------|------------------------------------------|
| `CTRL_BASE`   | `0x140000`  | mailbox: doorbell, args, params, heartbeat, status |
| `TELEM_BASE`  | `0x141000`  | 4 KiB scratch the kernel/overlay publishes |
| `CODE_SLOT_A` | `0x150000`  | 64 KiB overlay slot A                    |
| `CODE_SLOT_B` | `0x160000`  | 64 KiB overlay slot B (double-buffer)    |

High-L1 window, above metal's firmware/mailbox (low L1) and our tiny kernel `.text`. The launcher
allocates no CBs/buffers, so nothing competes.

## Drive it from bhtop (IMPLEMENTED — `bhtop.tensix`)

`tensix/bootloader.py` (`Bootloader`, Python twin of `bootloader_abi.h`) + `bl-*` CLI commands. `x y`,
or `--all` to broadcast across the whole grid:

```
bhtop-tensix bl-status <x> <y>            # decode CTRL_BASE: status, heartbeat, last_cmd, ret, params
bhtop-tensix bl-status --all              # one line per resident core
bhtop-tensix bl-param  <x> <y> <i> <val>  # poke PARAM[i]  (no compile, no re-go)
bhtop-tensix bl-stage  <x> <y> overlay.bin -s A   # NoC block-write .text into code slot A
bhtop-tensix bl-exec   <x> <y> -s A --wait        # set ARG0=slot, ring EXEC, wait for ack
bhtop-tensix bl-halt   --all              # DOORBELL=CMD_HALT (grid-wide)
bhtop-tensix bl-watch  <x> <y>            # live heartbeat + status + telemetry (TELEM_BASE)
```

`bl-stage` is a chunked NoC block write of the overlay `.bin`; `bl-exec` is two word pokes (ARG0 then
DOORBELL). Re-`go` is gone — the resident loop reads the mailbox every iteration. Driver verified
offline against the real compiled overlay; `--all` currently loops cores (true single-transaction
multicast is a future optimization).

## Build a code overlay (the "load new kernel" path)

```bash
cd kernels
SFPI=~/tt-metal/runtime/sfpi          # toolchain ships with metal
$SFPI/compiler/bin/riscv-tt-elf-g++ -Os -march=rv32im -mabi=ilp32 \
    -nostdlib -ffreestanding -fno-exceptions -fno-rtti \
    -T overlay.ld overlay_blink.cpp -o overlay_blink.elf
$SFPI/compiler/bin/riscv-tt-elf-objcopy -O binary -j .text -j .rodata \
    overlay_blink.elf overlay_blink.bin
# stage + exec:
bhtop-tensix bl-stage 1 2 overlay_blink.bin --slot A
bhtop-tensix bl-param 1 2 0 100000          # PARAM0 = iters
bhtop-tensix bl-exec  1 2 --slot A
bhtop-tensix bl-watch 1 2                    # telem[0] counts up; telem[2]=0xC0FFEE at done
```

## On-device validation points (the untested assumptions)

1. **i-cache flush.** The kernel writes CFG word **185** (`RISCV_IC_INVALIDATE.InvalidateAll`, mask
   `0x1f`) at `TENSIX_CFG_BASE=0xFFEF0000` + a short NOP drain — the same MMIO store firmware does
   (brisc.cc:239). Confirm this works from a BRISC data-movement kernel; if a stale-cache symptom
   appears, swap `flush_icache`'s body back to the 3072-NOP flood (noted in the source).
2. **Park semantics.** Two paths implemented: slow dispatch (`TT_METAL_SLOW_DISPATCH_MODE=1` →
   `detail::LaunchProgram(dev, program, wait=false, force_slow=true)`, fire-and-forget) and fast
   dispatch (non-blocking `EnqueueMeshWorkload`, never `Finish`). Slow dispatch is the safer default
   for a never-returning kernel; confirm exalens can attach while parked.
3. **Fixed-region collision.** Confirm metal places nothing in `0x140000–0x170000` (it shouldn't —
   no CBs/buffers allocated). `bhtop-tensix peek` the region right after launch to be sure.
4. **Overlay self-containment.** `overlay.ld` assumes `.text/.rodata` only. If the linker reports a
   non-empty `.data`/`.bss`, move that state into the param/telemetry region or add an init copy.
5. **Don't overwrite running code.** Only `bl-stage` a slot that isn't executing. Use slots A/B as a
   double-buffer, or `bl-halt`-the-overlay → confirm `STATUS=IDLE` → restage.

## Next milestones

- BRISC overlay with **NoC** (replicate `noc_init` from firmware) → real data movement you hot-swap.
- A **command handshake** so overlays can be swapped while the grid runs (status-gated restage).
- **TRISC/FPU overlays** — honor the compute-kernel ABI; backend is already inited by metal.
