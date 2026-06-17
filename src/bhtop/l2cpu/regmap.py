"""
bhtop.l2cpu.regmap — ONE canonical model of the L2CPU hardware environment.

Everything that needs an address — the loader (__init__.py), the C/asm/Rust harness
(include/bh.h, rt/bh.rs), the CLI `map`/`regs` inspector, and the HARDWARE.md docs —
draws from the tables here, so there is a single place the chip is described and no
two copies to drift apart. This is the bhtop philosophy applied to bring-up: the chip
is just a pile of memory-mapped registers; keep one honest map of them and surface it.

Three kinds of state live in an L2CPU tile:

  * DRAM windows        — big regions you write code/data/telemetry into (REGIONS)
  * peripheral registers — the knobs that boot, park and seize harts          (REGISTERS)
  * CSRs                 — per-hart control regs, reachable ONLY from hart code (CSRS)

Addresses are x280 *physical*, which equal the NoC address 1:1 in the low passthrough
window (see REGIONS). Per-hart registers repeat every `stride` bytes for hart N.

NOTHING in here touches the device; it is pure data + tiny formatting helpers.
"""

# ---- tiles: index -> (noc0 coord, L2CPU_RESET bit) (ISA L2CPUTile/README.md) ----
TILES = {0: ((8, 3), 4), 1: ((8, 9), 5), 2: ((8, 5), 6), 3: ((8, 7), 7)}
HARTS = 4                          # x280 harts per L2CPU tile

# ---- address windows as seen from the NoC ----
PASS_HI = 0x7FFF_FFFF_FFFF         # 0..this = 1:1 passthrough to x280 physical (SAFE)
NIU_DANGER = 0xFFFF_FFFF_FF00_0000  # NIU cfg window — poking it HANGS NoC0 (see bh-noc-hang-hazard)

# ---- L2CPU peripheral registers (x280 phys; per-hart = base + N*stride) ----
RESET_VEC = 0x20010000             # +N*8   initial PC for hart N (low32, high32)
HART_STATUS = 0x20010400           # all-harts run/halt/wfi/debug status (read-only)
SPIN_ADDR = 0x30000020             # `j .` park/exc target (moved to DRAM; scratch now holds the cmd mailbox)
TRIGGER = 0x20010414               # write bit N -> fire an RNMI on hart N (the "seize")
RNMI_TRAP = 0x20010418             # +N*16  RNMI trap-handler address for hart N
RNMI_EXC = 0x20010420              # +N*16  RNMI exception-handler address for hart N

# ---- DRAM layout we impose (uncached GDDR window, per tile) ----
TRAMP_ADDR = 0x30000000            # self-re-arming RNMI redirect trampoline (bringup installs)
# User code loads ABOVE all the fixed data blocks (tele/arch/cmd/varch end at ~0x30007400) so a
# kernel of any size can never collide with the telemetry it writes (a big kernel at 0x30001000
# used to spill into 0x30002000). Code grows up from here into open GDDR; nothing sits above it.
CODE_ADDR = 0x30008000             # default user-code load address (was 0x30001000, too small)
TELE_ADDR = 0x30002000             # telemetry: hart 0's window (host reads back)
TELE_SLOTS = 64                    # u32 slots per hart
TELE_STRIDE = 0x100                # per-hart stride: hart N window = TELE_ADDR + N*0x100

# ---- arch-state dump: a hart writes its whole register file here; host decodes it ----
# (the host can't read a hart's GPRs/CSRs directly, so `bh_dump_state()` snapshots them
#  to DRAM. Per-hart block: 32 GPRs as u64 at +0x00, then key CSRs.)
ARCH_ADDR = 0x30003000             # hart 0's arch-state block
ARCH_STRIDE = 0x200                # per-hart stride (512 B): 32 GPRs (256 B) + CSRs

# ---- host->hart command mailbox: a DRAM doorbell the hart polls to update a register ----
# The host CANNOT write an x280 CSR/GPR/vreg over the NoC (only the hart can, from its own
# code). So a live "register update" is cooperative: the host writes a value here and rings
# the doorbell; the hart polls it and applies it to its own register.
#
# CRITICAL — must live in UNCACHED memory: the x280 D-cache does NOT snoop the host's NoC
# writes, so a mailbox in cached GDDR (0x3000_xxxx) reads STALE on the hart (it worked only
# while the line was cold, then went deaf). The peripheral scratch 0x2001_0100 IS uncached
# (it is the same region the RNMI trampoline reads its host-written reset-vector from, which
# is why the seize works), so host writes here are immediately visible to hart reads. cbo.inval
# would also fix a cached mailbox but TRAPS as illegal on this x280, so we use region choice.
# 64 B scratch holds 4 harts x 16 B (seq/op/arg0/arg1); SPIN was moved out to DRAM (0x30000020).
CMD_ADDR = 0x20010100              # hart 0's command window (uncached peripheral scratch)
CMD_STRIDE = 0x10                  # per-hart stride: hart N window = CMD_ADDR + N*0x10
CMD_SLOTS = 4                      # u32 slots: [0]=seq doorbell [1]=op [2]=arg0 [3]=arg1
CMD_OPS = {0: "nop", 1: "set_csr(mscratch)", 2: "set_vreg(v16)",
           3: "set_vtype(0=e32m1,1=e32m4)", 4: "park", 5: "run",
           10: "vvirus:select_class(arg0, 0xFFFFFFFF=all)", 11: "vvirus:set_seed(arg0)",
           12: "vvirus:mutate(0=fixed,1=auto-randomize)"}

# ---- vector-register snapshot: a hart writes its WHOLE vector file + vector CSRs here ----
# bh_dump_state covers GPRs + scalar CSRs but NO vector state; bh_dump_vec() fills this gap:
# 32 vector regs (VLEN/8 = 64 B each = 0x800) then the 7 vector CSRs as u64. Host can't read
# vector regs over the NoC any more than GPRs, so the hart stores them to DRAM and we decode.
VARCH_ADDR = 0x30005000            # hart 0's vector-state block
VARCH_STRIDE = 0x900               # per-hart (2304 B): 32 vregs (0x800) + CSRs/magic (0x800+)
VARCH_VREGS = 32                   # v0..v31
VARCH_MAGIC = 0x0D0DEC00           # marker bh_dump_vec writes when the block is valid
VARCH_CSR_OFF = {                  # vector CSR -> byte offset (after the 0x800 of vregs)
    "vstart": 0x800, "vxsat": 0x808, "vxrm": 0x810, "vcsr": 0x818,
    "vl": 0x820, "vtype": 0x828, "vlenb": 0x830, "magic": 0x838,
}
VEC_CSRS = [                       # (number, name, desc) — the RVV control/status CSRs
    (0x008, "vstart", "element index where the next vector op resumes (usually 0)"),
    (0x009, "vxsat", "fixed-point saturation flag"),
    (0x00A, "vxrm", "fixed-point rounding mode"),
    (0x00F, "vcsr", "vector control/status (vxrm<<1 | vxsat)"),
    (0xC20, "vl", "active vector length (elements) set by the last vsetvl{i}"),
    (0xC21, "vtype", "SEW/LMUL/tail/mask config set by the last vsetvl{i}"),
    (0xC22, "vlenb", "bytes per vector register (VLEN/8 = 64 on x280)"),
]
ARCH_MAGIC = 0x0D0DEAD0            # marker bh_dump_state writes so the host knows it's valid
ARCH_CSR_OFF = {                   # CSR name -> byte offset within a hart's block
    "mhartid": 0x100, "mcycle": 0x108, "minstret": 0x110, "mstatus": 0x118,
    "mepc": 0x120, "mcause": 0x128, "mtval": 0x130, "mnepc": 0x138,
    "mncause": 0x140, "pc": 0x148, "magic": 0x150,
}

# ---- single source of truth for the kernel-facing memory map ----------------------------
# THIS module is the one place the map lives. The lab toolchain injects every name below into
# each compile (C: -D, asm: -Wa,--defsym, Rust: a generated include) so the harness headers
# (bh.h / tele.h / bh.inc / bh.rs) never hardcode the numbers — they carry #ifndef/.ifndef
# fallbacks that the injected values override, and Rust includes the generated file outright.
# Move an address HERE and every kernel picks it up on the next build, with nothing to keep in
# sync. (BH_CODE_BASE is injected per-compile = the actual load address; see harness_defines.)
# NOTE: bh.h's file-scope asm (bh_dump_state / bh_dump_vec) can't take a C macro, so the addresses
# + magics it embeds as `li`/`lui` literals (ARCH_ADDR/VARCH_ADDR, ARCH_MAGIC/VARCH_MAGIC) are NOT
# injected — they mirror this module by hand. Everything reached from C / asm-via-bh.inc / Rust is.
HARNESS_MAP = {
    "BH_TRAMP_BASE": TRAMP_ADDR,
    "TELE_BASE": TELE_ADDR, "TELE_STRIDE": TELE_STRIDE, "TELE_SLOTS": TELE_SLOTS,
    "BH_TELE_BASE": TELE_ADDR, "BH_TELE_STRIDE": TELE_STRIDE, "BH_TELE_SLOTS": TELE_SLOTS,
    "BH_ARCH_BASE": ARCH_ADDR, "BH_ARCH_STRIDE": ARCH_STRIDE,
    "BH_CMD_BASE": CMD_ADDR, "BH_CMD_STRIDE": CMD_STRIDE,
    "BH_VARCH_BASE": VARCH_ADDR, "BH_VARCH_STRIDE": VARCH_STRIDE,
}


def harness_defines(base):
    """name -> int for every compile-time constant the lab injects into a kernel build.
    `base` is the per-compile load address, surfaced as BH_CODE_BASE so it always matches
    where the image is actually loaded (= the linker's LOAD_ADDR), even on a custom deploy."""
    return {**HARNESS_MAP, "BH_CODE_BASE": base}

# ---- ARC registers (reached via pyluwen axi_*, NOT the NoC — avoids the hang hazard) ----
PLL4_BASE = 0x80020500             # L2CPU PLL #4 control block
PLL_CNTL_1 = 0x4                   #   +0x4  refdiv/postdiv/fbdiv
PLL_CNTL_5 = 0x14                  #   +0x14 four per-lane postdivs
L2CPU_RESET = 0x80030014           # one bit per tile; 0->1 releases harts (ONE-SHOT)
ARC_ALLOW = {PLL4_BASE + PLL_CNTL_1, PLL4_BASE + PLL_CNTL_5, L2CPU_RESET}


# ---- annotated tables (the machine-readable map; drives `map`/`regs` + docs) ----------
# Each REGION: (name, base, size_bytes, access, note)
REGIONS = [
    ("DRAM: trampoline", TRAMP_ADDR, 0x1000, "RW/X",
     "self-re-arming RNMI trampoline (installed by bringup)"),
    ("DRAM: user code", CODE_ADDR, 0x1000, "RW/X",
     "where `load` drops your compiled kernel; the hart runs from here"),
    ("DRAM: telemetry", TELE_ADDR, TELE_SLOTS * 4 * HARTS, "RW",
     "per-hart 64-slot windows (hart N at +N*0x100); your kernel writes, host reads (`tele`)"),
    ("DRAM: arch-state", ARCH_ADDR, ARCH_STRIDE * HARTS, "RW",
     "per-hart register-file dump (hart N at +N*0x200); bh_dump_state() writes 32 GPRs + CSRs"),
    ("Peripheral: cmd mailbox", CMD_ADDR, CMD_STRIDE * HARTS, "RW",
     "host->hart doorbell in uncached scratch (hart N at +N*0x10); host writes op/arg + bumps seq, hart polls"),
    ("DRAM: uncached GDDR", 0x30000000, 0x10000000, "RW/X",
     "the whole uncached off-chip GDDR window; code/data live here, no cache flush needed"),
    ("Peripheral (passthrough)", 0x20000000, 0x00020000, "RW",
     "x280 hart-control registers (reset vectors, RNMI, status) — see REGISTERS"),
    ("NoC passthrough", 0x0, PASS_HI + 1, "RW",
     "low window: NoC address == x280 physical 1:1 (the SAFE access path)"),
    ("NIU config (DANGER)", NIU_DANGER, 0x01000000, "—",
     "DO NOT TOUCH over the NoC — reading/writing here hangs NoC0 (tt-smi -r 0 to recover)"),
]

# Each REGISTER: dict(name, addr, stride, count, width, access, desc, fields)
#   stride/count: per-hart registers repeat (stride bytes, count harts); None => single.
#   access: RO read-only · RW read/write · W1 write-bit-to-act
#   fields: optional list of (bit_or_range, label) for decoding a read value.
REGISTERS = [
    dict(name="RESET_VEC", addr=RESET_VEC, stride=8, count=HARTS, width=8, access="RW",
         desc="Hart N's initial PC. `load` sets this, then fires TRIGGER so the "
              "trampoline jumps the hart here. 64-bit (low word @+0, high @+4).",
         fields=None),
    dict(name="HART_STATUS", addr=HART_STATUS, stride=0, count=1, width=2, access="RO",
         desc="Per-hart run state (4 bits/hart). 0x0000 = all parked/in-reset; nonzero "
              "bits mean a hart ceased/halted/wfi/debug.",
         fields=[(0, "hart0"), (1, "hart1"), (2, "hart2"), (3, "hart3")]),
    dict(name="TRIGGER", addr=TRIGGER, stride=0, count=1, width=4, access="W1",
         desc="Write 1<<N to fire a Resumable NMI on hart N — the 'seize'. Our trampoline "
              "clears this on entry, so a bit that STAYS set means the seize did not land.",
         fields=[(0, "seize h0"), (1, "seize h1"), (2, "seize h2"), (3, "seize h3")]),
    dict(name="RNMI_TRAP", addr=RNMI_TRAP, stride=16, count=HARTS, width=8, access="RW",
         desc="Where hart N jumps when its RNMI fires. We point this at the trampoline.",
         fields=None),
    dict(name="RNMI_EXC", addr=RNMI_EXC, stride=16, count=HARTS, width=8, access="RW",
         desc="Where hart N jumps on a fault inside RNMI context. We point it at a safe "
              "spin so a buggy kernel parks instead of running off into garbage.",
         fields=None),
]

# CSRs: reachable ONLY from hart code (asm/intrinsics), never over the NoC from the host.
#   (number, name, where, desc)
CSRS = [
    (0xF14, "mhartid", "RO", "this hart's index (0..3) — read it to self-identify"),
    (0xB00, "mcycle", "RW", "free-running cycle counter (64-bit; great cheap timer)"),
    (0xB02, "minstret", "RW", "instructions retired (64-bit)"),
    (0x353, "mnstatus", "RW", "RNMI status; bit3 NMIE gates RNMI delivery (our trampoline "
                              "re-sets it so redirects stay repeatable)"),
    (0x351, "mnepc", "RW", "RNMI saved PC (where an RNMI interrupted)"),
    (0x352, "mncause", "RW", "RNMI cause"),
    (0x350, "mnscratch", "RW", "scratch register for RNMI handlers"),
    (0x300, "mstatus", "RW", "machine status (global interrupt enables, prev mode, ...)"),
    (0x305, "mtvec", "RW", "machine trap-vector base (normal traps, not RNMI)"),
]

# PLL solutions used by bringup (target MHz -> [fbdiv, [4 postdivs]]); clock.py verbatim.
PLL_SOL = {200: [128, [15, 15, 15, 15]], 1750: [140, [1, 1, 1, 1]]}

# GPR ABI names (x0..x31) for arch-state decode/display (matches the psABI)
GPR_ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
           "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4",
           "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
GPR = [{"x": n, "abi": GPR_ABI[n]} for n in range(32)]


# ---- helpers -------------------------------------------------------------------------
def hart_addr(base, hart, stride):
    """Address of a per-hart register for hart N."""
    return base + hart * stride


def tile_coord(tile):
    return TILES[tile][0]


def decode_bits(value, fields):
    """Return the labels whose bit is set in `value` (for fields=[(bit,label),...])."""
    return [label for bit, label in (fields or []) if value & (1 << bit)]


def render_map():
    """Human-readable rendering of the whole map for `bhtop-l2cpu map` (no device)."""
    out = ["L2CPU hardware map — addresses are x280 physical (== NoC addr, low window)\n"]
    out.append("MEMORY REGIONS")
    for name, base, size, acc, note in REGIONS:
        out.append(f"  0x{base:016X}  {acc:5}  {name:26}  {note}")
    out.append("\nPERIPHERAL REGISTERS  (per-hart ones repeat every `stride` for hart N)")
    for r in REGISTERS:
        span = f" +N*{r['stride']}" if r["stride"] else ""
        out.append(f"  0x{r['addr']:08X}{span:7} {r['access']:3} {r['width']}B  {r['name']}")
        out.append(f"             {r['desc']}")
    out.append("\nCSRs  (read/write from HART code only — not reachable from the host)")
    for num, name, acc, desc in CSRS:
        out.append(f"  csr 0x{num:03X}  {acc:2}  {name:10} {desc}")
    out.append("\nARC registers  (host side, via pyluwen axi_* — a separate transport)")
    out.append(f"  0x{PLL4_BASE:08X}  PLL4 control (clock glide during bringup)")
    out.append(f"  0x{L2CPU_RESET:08X}  L2CPU_RESET — bit per tile, 0->1 releases (ONE-SHOT)")
    out.append("\nTILES  (index -> noc0 coord, reset bit)")
    for i, (xy, bit) in TILES.items():
        out.append(f"  tile {i}: noc0 {xy}  L2CPU_RESET bit {bit}")
    return "\n".join(out)
