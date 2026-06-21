# Tensix launch ABI (Blackhole) — reference for the exalens loader

Goal: read and **poke** a Tensix kernel's runtime args directly in L1 over the NoC, and re-trigger
the firmware — on-the-fly editing with no recompile/rebuild. This is the x280-loader move applied
to tt-metal's resident firmware launch protocol.

Captured from the **installed** tt-metal (`~/tt-metal`); re-verify after any tt-metal upgrade with
the probe at the bottom. All values below are for **Blackhole**.

## What tt-metal does on the device (the 5 stages)

1. **Build (JIT)** — generates `compile_time_args.h` (CTAs become `constexpr`, baked in) + CB/NoC
   config, compiles kernel + per-RISC firmware → one binary per RISC, cached by hash.
   `TT_METAL_INSPECTOR=1` dumps the hash↔source↔program map (bhtop already sets it).
2. **Load** — writes firmware + kernel binaries into each core's L1 (`llrt::write_hex_vec_to_core`).
3. **Configure** — writes **runtime args**, CB configs, semaphore inits to L1.
4. **Launch** — writes a `launch_msg_t` to the L1 mailbox + a go signal
   (`llrt::write_launch_msg_to_core` + `send_reset_go_signal`). Firmware polls the mailbox, runs
   the kernel, writes DONE.
5. **Readback** — poll DONE, read L1/DRAM.

`llrt` (low-level runtime) does stages 2–5 with exactly the primitives tt-exalens gives us
(`read/write words`, reset). **exalens is the transport; we re-implement the bits of `llrt` we
need.** The build (stage 1) we leave to tt-metal.

## The one fact that makes poking work

`get_arg_val<T>(i)` is a plain L1 read (`tt_metal/hw/inc/api/compute/common.h`):

```
get_arg_val<T>(i)  ==  *(T*)&rta_l1_base[i]      // rta_l1_base = kernel_config_base + rta_offset[proc]
```

So runtime args are just bytes in L1 → poke + re-go, no rebuild. Compile-time args are `constexpr`
in the binary (`get_compile_time_arg_val(i) → get_ct_arg<i>()`) → can't poke, need a rebuild.

## L1 layout (probe-verified, this build)

`mailboxes_t` lives at L1 byte **`MEM_MAILBOX_BASE = 96`** (`MEM_MAILBOX_SIZE = 12912`), in every
Tensix worker. Field offsets within `mailboxes_t` (it is **not** packed — alignment matters):

| field | offset | L1 addr | notes |
|---|---|---|---|
| `launch_msg_rd_ptr` | 12 | 108 | u32 ring read pointer (0..7) |
| `launch[8]` | 16 | 112 | stride **112** (`sizeof(launch_msg_t)`) |
| `go_messages[9]` | 912 | 1008 | stride 4 (`go_msg_t`) |
| `go_message_index` | 960 | 1056 | u32, which `go_messages[]` is live |

`launch[idx].kernel_config` (`kernel_config_msg_t`, **packed**, 112 B):

| field | offset | type | use |
|---|---|---|---|
| `kernel_config_base[4]` | 0 | u32×4 | L1 base of the config region, per `ProgrammableCoreType` (TENSIX=0) |
| `rta_offset[5]` | 28 | `{u16 rta, u16 crta}`×5 | per-processor RTA/CRTA offset from `kernel_config_base` |
| `mode` | 48 | u8 | 0=DEV dispatch, 1=HOST dispatch |
| `kernel_text_offset[5]` | 52 | u32×5 | per-processor kernel binary offset |
| `host_assigned_id` | 84 | u32 | program id (match a known run) |
| `enables` | 88 | u32 | bit *i* ⇒ processor *i* enabled |

Processors (`TensixProcessorTypes`): `DM0=0, DM1=1, MATH0=2, MATH1=3, MATH2=4` (`MaxProcessorsPerCoreType=5`).

**RTA address** for processor *p*: `kernel_config_base[TENSIX] + rta_offset[p].rta`; arg *i* at `+ i*4`.

`go_msg_t` (u32): byte0 `dispatch_message_offset`, b1 `master_x`, b2 `master_y`, **b3 `signal`**.
Signals: `INIT=0x40`, `GO=0x80`, `RESET_RD_PTR=0xC0`, `DONE=0x00`.

## Hybrid flow (v1 — what loader.py implements)

1. tt-metal opens device + JIT-builds + runs the program once (firmware resident, kernel binary +
   launch_msg + RTAs in L1).
2. `TensixLauncher(coord, ctx)`:
   - `read_ring()` / `snapshot()` — decode the live launch ring, kernel_config, RTA addresses, go.
   - `write_rta(proc, [vals], arg_offset=…)` — poke new runtime-arg words into L1.
   - `go()` — set the active go signal to GO → re-run with the new args.

Cleanest in **slow dispatch** (`TT_METAL_SLOW_DISPATCH_MODE=1`): the host owns the launch ring, so
poke+go is the whole story. Under fast dispatch a dispatch core also drives the ring, so `go()` is
experimental there (the dispatcher may overwrite the entry). Done-detection (poll signal→DONE) and
owning the ring outright are the next things to reverse-engineer.

## Safety

Tensix worker **L1 reads/writes over NoC are in the safe zone** — they do not touch the
ARC/Security/PCIe/L2CPU register surface that hangs NoC0 (see the BH NoC hang-hazard note;
recovery is `tt-smi -r 0`). `loader.py` bounds every access to `[0, MEM_L1_SIZE)`. Functional risk
only: a wrong RTA/go can make a kernel misbehave, not hang the NoC.

## Re-deriving the offsets after a tt-metal upgrade

`abi.py`'s offsets come from an `offsetof` probe that faithfully reproduces the structs in
`tt_metal/hw/inc/hostdev/dev_msgs.h` with the Blackhole counts from
`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h`
(`ProgrammableCoreType::COUNT=4`, `MaxProcessorsPerCoreType=5`,
`MaxProcessorsForThreadingVariables=0`, `subordinate_map_size=4`) and `MEM_MAILBOX_BASE=96` from
that arch's `dev_mem_map.h`. If those structs/counts change, re-run the probe (kept in the repo
history / regenerable from the headers) and update `abi.py`.
```
g++ -std=c++17 probe.cpp && ./a.out   # prints sizeof + offsetof for every field above
```
