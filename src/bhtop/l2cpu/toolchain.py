"""
bhtop.l2cpu.toolchain — compile asm / C / Rust to a flat binary for the x280.

Produces a position-fixed flat image (list of u32 words) linked at `base`, ready to
drop into DRAM and jump to. Uses the RISC-V toolchain that ships with tt-metal
(riscv-tt-elf-* under runtime/sfpi); Rust uses a rustup `riscv64gc-unknown-none-elf`
toolchain if installed.

Layout: the bundled rt/link.ld places `.text._start` first at `base` (overridable via
--defsym=LOAD_ADDR), packs rodata/data, and reserves stack/bss. For C we also link
rt/crt0.s so the user only writes `int main(void)` (crt0 sets sp, zeroes bss, calls
main, then parks). asm/Rust provide their own `_start`.
"""
import os
import struct
import subprocess
import tempfile

from .regmap import harness_defines, CODE_ADDR

SFPI = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin")
HERE = os.path.dirname(__file__)
RT = os.path.join(HERE, "rt")
INCLUDE = os.path.join(HERE, "include")
LINK_LD = os.path.join(RT, "link.ld")
CRT0 = os.path.join(RT, "crt0.s")
DEFAULT_BASE = CODE_ADDR           # single source of truth: regmap.CODE_ADDR (the load window)

RUST_TARGET = "riscv64gc-unknown-none-elf"
EXTS = {".s": "asm", ".S": "asm", ".c": "c", ".rs": "rust"}


class ToolError(RuntimeError):
    pass


def tool(name):
    return os.path.join(SFPI, f"riscv-tt-elf-{name}")


def have_gcc():
    return os.path.exists(tool("gcc"))


def have_rust():
    # apt `rustc` lists every target but lacks bare-metal core; need rustup + the
    # installed target. (rustup also shadows apt rustc so `rustc` picks up core.)
    if not _which("rustup"):
        return False
    return RUST_TARGET in _run(["rustup", "target", "list", "--installed"], check=False)[1]


def _which(p):
    from shutil import which
    return which(p)


def _run(cmd, check=True, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise ToolError(f"$ {' '.join(cmd)}\n{(r.stderr or r.stdout).strip()}")
    return r.returncode, (r.stdout + r.stderr)


def _rust_env(mapf):
    """rustc env: BH_RT points at rt/ so kernels can
    `include!(concat!(env!(\"BH_RT\"), \"/bh.rs\"))`; BH_MAP points at the generated
    memory-map include so bh.rs pulls the canonical regmap.py values (same single
    source as the C -D / asm --defsym injection)."""
    return {**os.environ, "BH_RT": RT, "BH_MAP": mapf}


def _write_rust_map(d, base, defines=None):
    """Emit the canonical map (+ per-kernel `defines`) as Rust consts for bh.rs to include!.
    Keeps Rust kernels in lockstep with the map with nothing to hand-sync."""
    body = "// generated from regmap.py per build — do not edit; the lab injects this\n"
    body += "".join(f"pub const {k}: usize = {v:#x};\n" for k, v in _defmap(base, defines).items())
    p = os.path.join(d, "bh_map.rs")
    with open(p, "w") as fh:
        fh.write(body)
    return p


def detect_lang(path):
    return EXTS.get(os.path.splitext(path)[1], None)


def _to_words(binpath):
    b = open(binpath, "rb").read()
    if len(b) % 4:
        b += b"\x00" * (4 - len(b) % 4)
    return list(struct.unpack("<%dI" % (len(b) // 4), b)) if b else []


def _objcopy(elf, out):
    _run([tool("objcopy"), "-O", "binary", elf, out])


def _defmap(base, defines=None):
    """The canonical map (regmap.harness_defines) merged with optional per-kernel `defines`,
    kernel values winning. Values are coerced to int so a kernel.json may pass 256 or "0xAA"
    alike. These are injected as -D / --defsym / rust consts — see _gcc_cmd / _write_rust_map.
    NOTE: a per-kernel define only takes effect if the source leaves it #ifndef-guarded (or
    undefined); an unconditional `#define ITERS 256` in the source would override the -D."""
    m = dict(harness_defines(base))
    for k, v in (defines or {}).items():
        m[k] = v if isinstance(v, int) else int(str(v), 0)
    return m


def _gcc_cmd(elf, base, lang, path, defines=None, march="rv64gc"):
    defs = _defmap(base, defines)                            # canonical map + per-kernel overrides
    cmd = [tool("gcc"), f"-march={march}", "-mabi=lp64d", "-nostdlib", "-nostartfiles",
           "-fno-pic", f"-T{LINK_LD}", f"-Wl,--defsym=LOAD_ADDR={base:#x}",
           "-Wl,--no-relax", "-o", elf]
    if lang == "c":
        cmd += [f"-D{k}={v:#x}u" for k, v in defs.items()]   # inject the map; headers #ifndef-fallback
        cmd += ["-ffreestanding", "-O2", "-fno-builtin", "-fno-stack-protector",
                "-fno-tree-loop-distribute-patterns", "-ffunction-sections",
                f"-I{INCLUDE}", path, CRT0]                  # crt0 provides _start->main
    else:
        cmd += [f"-Wa,--defsym={k}={v:#x}" for k, v in defs.items()]  # asm: same map as .equ-overriding symbols
        cmd += [f"-Wa,-I{INCLUDE}", path]                    # asm: -I lets `.include "bh.inc"` resolve
    return cmd


def _rustc_cmd(elf, base, path):
    return ["rustc", "--edition", "2021", "--target", RUST_TARGET, "-C", "panic=abort",
            "-C", "opt-level=2", "-C", "relocation-model=static",
            "-C", f"link-arg=-T{LINK_LD}", "-C", f"link-arg=--defsym=LOAD_ADDR={base:#x}",
            "-C", "link-arg=--no-relax", "-o", elf, path]


def _build_elf(path, elf, base, lang, defines=None, march="rv64gc"):
    """Compile any supported language to a position-fixed ELF at `elf`."""
    if lang not in ("asm", "c", "rust"):
        raise ToolError(f"unknown language for {path} (use .s/.c/.rs or pass lang=)")
    if lang in ("asm", "c"):
        if not have_gcc():
            raise ToolError(f"sfpi gcc not found at {SFPI} — is tt-metal present?")
        _run(_gcc_cmd(elf, base, lang, path, defines, march=march))
    else:
        if not have_rust():
            raise ToolError(
                "Rust bare-metal toolchain not ready. Install:\n"
                "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n"
                "  . \"$HOME/.cargo/env\" && rustup target add " + RUST_TARGET)
        mapf = _write_rust_map(os.path.dirname(elf), base, defines)   # canonical map + overrides for bh.rs
        _run(_rustc_cmd(elf, base, path), env=_rust_env(mapf))


def compile_source(path, base=DEFAULT_BASE, lang=None, defines=None, march="rv64gc"):
    """Compile `path` to a flat image linked at `base`. Returns list[u32 words].
    `defines` (name->int|hex-str) are injected on top of the canonical map (kernel wins).
    `march` overrides the ISA string (use 'rv64gcv' for kernels with RVV intrinsics)."""
    lang = lang or detect_lang(path)
    with tempfile.TemporaryDirectory() as d:
        elf = os.path.join(d, "a.elf")
        binf = os.path.join(d, "a.bin")
        _build_elf(path, elf, base, lang, defines, march=march)
        _objcopy(elf, binf)
        return _to_words(binf)


def disasm(path, base=DEFAULT_BASE, lang=None, defines=None, march="rv64gc"):
    """Compile and return objdump disassembly (works for asm/C/Rust alike)."""
    lang = lang or detect_lang(path)
    with tempfile.TemporaryDirectory() as d:
        elf = os.path.join(d, "a.elf")
        _build_elf(path, elf, base, lang, defines, march=march)
        return _run([tool("objdump"), "-d", elf], check=False)[1]
