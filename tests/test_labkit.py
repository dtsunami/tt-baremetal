"""
Regression net for the shared lab helpers (Phase 0 of the labs unification).

Runnable WITHOUT pytest:  .venv/bin/python tests/test_labkit.py
Pins the behavior the nlab/xlab backends rely on, so the Phase B/C migrations onto
labkit can't silently drift: parse_compiler_errors matches the OLD lab/l2lab regexes,
safe_path matches the OLD _safe traversal/extension rules, and Broadcaster delivers the
EXACT frame object (byte-identical — the critique's non-negotiable) with drop-on-full.
"""
import asyncio
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bhtop.web import labkit

# the regex the xlab/nlab backends used BEFORE the labkit migration — pin equivalence
_OLD_L2_ERR_RE = re.compile(r"([\w./+-]+):(\d+):(\d+):\s*(?:fatal\s+)?error:\s*(.*)", re.I)


def check(name, cond):
    print(f"  {'ok ' if cond else 'FAIL'} {name}")
    if not cond:
        raise AssertionError(name)


def test_parse_compiler_errors():
    print("parse_compiler_errors")
    gcc = ("kernel.c: In function 'main':\n"
           "kernel.c:7:5: error: unknown type name 'broken'\n"
           "kernel.c:7:18: error: expected ';' before '}' token\n")
    errs = labkit.parse_compiler_errors(gcc)
    check("finds both gcc errors", len(errs) == 2)
    check("line/col parsed", errs[0] == {"file": "kernel.c", "line": 7, "col": 5,
                                         "msg": "unknown type name 'broken'"})
    # equivalence with the OLD l2lab regex (xlab compiles a TempDir file then basenames)
    old = [{"file": os.path.basename(m.group(1)), "line": int(m.group(2)), "col": int(m.group(3)),
            "msg": m.group(4).strip()} for m in _OLD_L2_ERR_RE.finditer(gcc)]
    new = [{**e, "file": os.path.basename(e["file"])} for e in labkit.parse_compiler_errors(gcc)]
    check("matches OLD l2lab._ERR_RE output", old == new)
    # nlab ninja cpp error: same error, line/col/msg, basename — the only intended change is
    # that the unified parser keeps the full path (the old nlab regex's (?:^|/) stripped a
    # leading segment, e.g. ../tests/x.cpp -> tests/x.cpp). Cosmetic; not frame-affecting.
    ninja = ("[3/9] Building CXX object .../test_x.cpp.o\n"
             "../tests/foo/test_x.cpp:42:10: error: no member named 'bar'\n"
             "ninja: build stopped.\n")
    new_n = labkit.parse_compiler_errors(ninja)
    check("finds the one cpp error", len(new_n) == 1)
    check("basename + line/col/msg correct",
          os.path.basename(new_n[0]["file"]) == "test_x.cpp" and new_n[0]["line"] == 42
          and new_n[0]["col"] == 10 and new_n[0]["msg"] == "no member named 'bar'")
    check("no false positive on 'ninja: build stopped.'",
          all("ninja" not in e["file"] for e in new_n))


def test_safe_path():
    print("safe_path")
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "kernels"))
        open(os.path.join(root, "a.c"), "w").close()
        exts = {".c", ".s", ".rs"}
        # valid flat file
        check("valid flat file resolves", labkit.safe_path(root, "a.c", exts) == os.path.realpath(os.path.join(root, "a.c")))
        # valid nested file (nlab needs kernels/ subdirs)
        check("valid nested file resolves", labkit.safe_path(root, "kernels/x.c", exts).endswith("/kernels/x.c"))
        # traversal rejected
        for bad in ("../escape.c", "../../etc/passwd.c", "kernels/../../out.c"):
            try:
                labkit.safe_path(root, bad, exts); check(f"traversal rejected: {bad}", False)
            except ValueError:
                check(f"traversal rejected: {bad}", True)
        # bad extension rejected
        try:
            labkit.safe_path(root, "a.txt", exts); check("bad ext rejected", False)
        except ValueError:
            check("bad ext rejected", True)


def test_broadcaster():
    print("Broadcaster")

    async def run():
        # seeded channel hands the snapshot to a new subscriber
        seeded = labkit.Broadcaster(seed=lambda: {"snap": 1})
        q = await seeded.subscribe()
        first = q.get_nowait()
        check("seed delivered to new subscriber", first == {"snap": 1})

        b = labkit.Broadcaster()
        q1 = await b.subscribe()
        q2 = await b.subscribe()
        check("two subscribers tracked", len(b) == 2)
        frame = {"ts": 1.0, "tiles": {"8,3": {"noc0": 5}}}
        b.broadcast(frame)
        got1 = q1.get_nowait()
        got2 = q2.get_nowait()
        # the EXACT same object — no copy/mutation (byte-identical guarantee)
        check("frame delivered by identity (no mutation)", got1 is frame and got2 is frame)
        # drop-on-full: queue maxsize is 4; broadcasting 6 without draining must not raise
        b.unsubscribe(q2)
        for i in range(6):
            b.broadcast({"i": i})
        check("drop-on-full never raises", True)
        check("queue capped at maxsize 4", q1.qsize() == 4)
        # unsubscribe removes
        b.unsubscribe(q1)
        check("unsubscribe removes client", len(b) == 0)

    asyncio.run(run())


if __name__ == "__main__":
    test_parse_compiler_errors()
    test_safe_path()
    test_broadcaster()
    print("\nALL LABKIT CHECKS PASSED")
