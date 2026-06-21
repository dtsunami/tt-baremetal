"""
tensix.abi — the Blackhole Tensix launch ABI, captured from the installed tt-metal headers so
bhtop can read/poke a kernel's runtime args (RTAs) directly in L1 over the NoC — the same move the
x280 loader makes, but against tt-metal's firmware launch protocol instead of our own crt0.

GROUND TRUTH (verified by an offsetof probe, see tensix/TENSIX_ABI.md):
  - tt_metal/hw/inc/hostdev/dev_msgs.h           — mailboxes_t / launch_msg_t / kernel_config_msg_t / go_msg_t
  - tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h   — MEM_MAILBOX_BASE = 96
  - tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h   — the enum counts below

Each Tensix worker core has a `mailboxes_t` at L1 byte MEM_MAILBOX_BASE. The firmware (resident
after tt-metal opens the device) polls `launch[rd_ptr].kernel_config` + a `go_messages[]` signal.
Runtime args live in L1 at `kernel_config_base[TENSIX] + rta_offset[proc].rta_offset`, and
`get_arg_val<T>(i)` is literally `*(uint32_t*)(rta_base + i*4)` — so they're pokeable.

PURE (no device): all addresses + (un)packers; tensix.loader does the NoC I/O. Offsets are for
THIS installed Blackhole build — re-run the probe in TENSIX_ABI.md after a tt-metal upgrade.
"""
import struct

# ---- L1 memory map (Blackhole) ------------------------------------------------------
MEM_L1_BASE = 0x0
MEM_L1_SIZE = 1536 * 1024
MEM_MAILBOX_BASE = 96            # dev_mem_map.h: where mailboxes_t starts in each worker's L1
MEM_MAILBOX_SIZE = 12912

# ---- core_config.h (Blackhole) counts -----------------------------------------------
PROGRAMMABLE_CORE_COUNT = 4      # TENSIX=0, ACTIVE_ETH=1, IDLE_ETH=2, DRAM=3
CORE_TYPE_TENSIX = 0
MAX_PROCS = 5                    # MaxProcessorsPerCoreType
# TensixProcessorTypes — the 5 RISCs that read args on a worker
PROC = {"DM0": 0, "DM1": 1, "MATH0": 2, "MATH1": 3, "MATH2": 4}
PROC_NAME = {v: k for k, v in PROC.items()}

# ---- mailboxes_t field offsets (probe-verified; unpacked struct, so alignment matters) ----
LAUNCH_RD_PTR_OFF = 12
LAUNCH_OFF = 16
LAUNCH_STRIDE = 112              # sizeof(launch_msg_t) == sizeof(kernel_config_msg_t)
LAUNCH_ENTRIES = 8
GO_MSG_OFF = 912
GO_STRIDE = 4                    # sizeof(go_msg_t)
GO_ENTRIES = 9
GO_MSG_INDEX_OFF = 960

# ---- kernel_config_msg_t field offsets (packed, 112 bytes) --------------------------
KCFG_BASE_OFF = 0               # uint32 kernel_config_base[PROGRAMMABLE_CORE_COUNT]
KCFG_RTA_OFF = 28               # rta_offset_t rta_offset[MAX_PROCS]  ({u16 rta, u16 crta})
KCFG_MODE_OFF = 48              # uint8 mode (0 = DISPATCH_MODE_DEV, 1 = DISPATCH_MODE_HOST)
KCFG_KTEXT_OFF = 52            # uint32 kernel_text_offset[MAX_PROCS]
KCFG_HOST_ID_OFF = 84          # uint32 host_assigned_id  (program id, set by host)
KCFG_ENABLES_OFF = 88          # uint32 enables (bit i => processor i enabled)
KCFG_WATCHER_IDS_OFF = 92      # uint16 watcher_kernel_ids[MAX_PROCS] — join w/ Inspector kernels.yaml

# ---- go_msg_t signals (dev_msgs.h) --------------------------------------------------
RUN_MSG_INIT = 0x40
RUN_MSG_GO = 0x80
RUN_MSG_RESET_READ_PTR = 0xC0
RUN_MSG_DONE = 0x00
SIGNAL_NAME = {0x40: "INIT", 0x80: "GO", 0xC0: "RESET_RD_PTR", 0x00: "DONE"}


# ---- address helpers ----------------------------------------------------------------
def launch_rd_ptr_addr():
    return MEM_MAILBOX_BASE + LAUNCH_RD_PTR_OFF


def launch_addr(idx):
    """L1 byte address of launch[idx].kernel_config."""
    return MEM_MAILBOX_BASE + LAUNCH_OFF + idx * LAUNCH_STRIDE


def go_addr(idx):
    """L1 byte address of go_messages[idx] (a single u32)."""
    return MEM_MAILBOX_BASE + GO_MSG_OFF + idx * GO_STRIDE


def go_index_addr():
    return MEM_MAILBOX_BASE + GO_MSG_INDEX_OFF


# ---- (un)packing --------------------------------------------------------------------
def words_to_bytes(words):
    return struct.pack(f"<{len(words)}I", *[w & 0xFFFFFFFF for w in words])


def bytes_to_words(b):
    if len(b) % 4:
        b = b + b"\x00" * (4 - len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def decode_kernel_config(buf):
    """Decode a launch[idx].kernel_config (112 raw bytes) into a dict of the fields bhtop needs."""
    base = list(struct.unpack_from(f"<{PROGRAMMABLE_CORE_COUNT}I", buf, KCFG_BASE_OFF))
    rta_raw = struct.unpack_from(f"<{MAX_PROCS * 2}H", buf, KCFG_RTA_OFF)
    rta = [{"rta": rta_raw[2 * i], "crta": rta_raw[2 * i + 1]} for i in range(MAX_PROCS)]
    ktext = list(struct.unpack_from(f"<{MAX_PROCS}I", buf, KCFG_KTEXT_OFF))
    mode = buf[KCFG_MODE_OFF]
    host_id = struct.unpack_from("<I", buf, KCFG_HOST_ID_OFF)[0]
    enables = struct.unpack_from("<I", buf, KCFG_ENABLES_OFF)[0]
    wids = list(struct.unpack_from(f"<{MAX_PROCS}H", buf, KCFG_WATCHER_IDS_OFF))
    return {"kernel_config_base": base, "rta_offset": rta, "kernel_text_offset": ktext,
            "mode": mode, "host_assigned_id": host_id, "enables": enables,
            "watcher_kernel_ids": wids,
            "enabled_procs": [p for p in range(MAX_PROCS) if enables & (1 << p)]}


def rta_l1_addr(kcfg, proc, core_type=CORE_TYPE_TENSIX):
    """L1 byte address where processor `proc` reads its (unique) runtime args:
    kernel_config_base[core_type] + rta_offset[proc].rta_offset. get_arg_val(i) = *(addr + i*4)."""
    return kcfg["kernel_config_base"][core_type] + kcfg["rta_offset"][proc]["rta"]


def crta_l1_addr(kcfg, proc, core_type=CORE_TYPE_TENSIX):
    """L1 byte address of the COMMON runtime args (SetCommonRuntimeArgs) for `proc`."""
    return kcfg["kernel_config_base"][core_type] + kcfg["rta_offset"][proc]["crta"]


def decode_go(word):
    """Decode a go_msg_t u32: byte0=dispatch_message_offset, b1=master_x, b2=master_y, b3=signal."""
    return {"dispatch_message_offset": word & 0xFF, "master_x": (word >> 8) & 0xFF,
            "master_y": (word >> 16) & 0xFF, "signal": (word >> 24) & 0xFF,
            "signal_name": SIGNAL_NAME.get((word >> 24) & 0xFF, hex((word >> 24) & 0xFF))}


def with_signal(word, signal):
    """Return the go_msg word with its signal byte (high byte) replaced — keeps master_x/y."""
    return (word & 0x00FFFFFF) | ((signal & 0xFF) << 24)
