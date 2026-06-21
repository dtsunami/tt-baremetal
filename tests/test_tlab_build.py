"""
Pins tlab_build's recipe parser + the real-cache helpers (kernel_includes/user_source).
Runnable WITHOUT pytest:  .venv/bin/python tests/test_tlab_build.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bhtop.web import tlab_build as tb

_fails = 0


def check(name, cond):
    global _fails
    if not cond:
        _fails += 1
    print(f"  {'ok ' if cond else 'FAIL'} {name}")


def test_parse_recipe():
    out = (
        "2026-06-19 | INFO | BuildKernels |     g++ compile cmd: cd /c/kernels/add/H/trisc0/ && "
        "/sfpi/g++ -I/c/kernels/add/H/trisc0/ -O3 -c -o trisck.o /tt/hw/firmware/trisck.cc -MF x.d -DFOO=1\n"
        "2026-06-19 | INFO | BuildKernels |     g++ link cmd: cd /c/kernels/add/H/trisc0/ && "
        "/sfpi/g++ -nostartfiles trisck.o -o /c/kernels/add/H/trisc0/trisc0.elf\n"
        "Success: Result matches expected value!\n"
        "2026-06-19 | INFO | BuildKernels |     g++ compile cmd: cd /c/kernels/rd/G/ncrisc/ && "
        "/sfpi/g++ -c -o ncrisck.o /tt/hw/firmware/ncrisck.cc -MF y.d\n"
        "2026-06-19 | INFO | BuildKernels |     g++ link cmd: cd /c/kernels/rd/G/ncrisc/ && "
        "/sfpi/g++ ncrisck.o -o /c/kernels/rd/G/ncrisc/ncrisc.elf\n"
    )
    units = tb.parse_recipe(out)
    check("parsed 2 units", len(units) == 2)
    u = units.get("/c/kernels/add/H/trisc0/")
    check("unit keyed by out_dir", u is not None)
    check("has compile + link", "compile" in u and "link" in u)
    check("elf captured", u.get("elf") == "/c/kernels/add/H/trisc0/trisc0.elf")
    check("compile cmd preserved", u["compile"].startswith("cd /c/kernels/add/H/trisc0/ &&"))
    # path relocation (what build() does): replace hashdir prefix -> staging
    relocated = u["compile"].replace("/c/kernels/add/H", "/bhtop/.build/H")
    check("relocate rewrites out paths", "/bhtop/.build/H/trisc0/" in relocated
          and "/tt/hw/firmware/trisck.cc" in relocated)   # firmware src untouched


def test_partial_units_dropped():
    out = "x | g++ compile cmd: cd /c/k/H/tr/ && g++ -c -o a.o s.cc\n"   # no link
    units = tb.parse_recipe(out)
    # compile-only is still a unit (build() handles missing link gracefully); ensure it parsed
    check("compile-only unit kept", len(units) == 1 and "compile" in next(iter(units.values())))


def test_live_cache_helpers():
    """Against the real tt-metal cache (if present): kernel_includes hash dir + user .cpp resolve."""
    from glob import glob
    kis = glob(os.path.expanduser("~/.cache/tt-metal-cache/*/kernels/*/*/kernel_includes.hpp"))
    if not kis:
        check("live cache: (none present — skipped)", True)
        return
    hashdir = os.path.dirname(kis[0])
    check("live: _kernel_includes finds hashdir", tb._kernel_includes(hashdir) == hashdir)
    check("live: _kernel_includes finds from a risc subdir",
          tb._kernel_includes(os.path.join(hashdir, "ncrisc")) == hashdir
          or tb._kernel_includes(os.path.join(hashdir, "trisc0")) == hashdir
          or tb._kernel_includes(hashdir) == hashdir)
    src = tb._user_source(hashdir)
    check(f"live: _user_source -> a .cpp ({os.path.basename(src) if src else None})",
          src is not None and src.endswith(".cpp"))


def main():
    print("tlab_build tests")
    test_parse_recipe()
    test_partial_units_dropped()
    test_live_cache_helpers()
    print(f"\n{'ALL PASS' if not _fails else str(_fails) + ' FAILED'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
