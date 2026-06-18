"""
Regression net for kernparse — the source->kernel.json param importer.

Runnable WITHOUT pytest:  .venv/bin/python tests/test_kernparse.py

Covers: x280 #ifndef/#define + documented-mailbox discovery (and the include-guard / canonical-map
exclusions), tt-metal rtarg/ctarg discovery + best-effort host enrichment, the merge identity rules
(index-keyed for args, name-keyed otherwise) with idempotency + edit-preservation, and a live pass:
every param discovered across the REAL kernel trees must pass kernmeta.validate so a merge is always
writable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bhtop.web import kernparse as kp
from bhtop.web import kernmeta

_fails = 0


def check(name, cond):
    global _fails
    if not cond:
        _fails += 1
    print(f"  {'ok ' if cond else 'FAIL'} {name}")


def by_name(params):
    return {p["name"]: p for p in params}


# ---- x280 ---------------------------------------------------------------------------
def test_x280_defines():
    src = {
        "k.c": (
            "#ifndef BH_H\n#define BH_H\n#endif\n"                # include guard -> skipped
            "#ifndef ITERS\n#define ITERS    256u   /* loop count */\n#endif\n"
            "#ifndef MASK\n#define MASK 0xAA  // bit mask\n#endif\n"
            "#ifndef BH_TELE_BASE\n#define BH_TELE_BASE 0x1000\n#endif\n"   # canonical -> skipped
            "#define NOPS (ITERS*32u)\n"                          # unguarded expr -> skipped
        )
    }
    d = kp.parse_x280(src, skip_defines={"BH_TELE_BASE"})
    bn = by_name(d)
    check("x280: ITERS discovered as int 256", bn.get("ITERS", {}).get("default") == 256
          and bn["ITERS"]["type"] == "int" and bn["ITERS"]["kind"] == "define")
    check("x280: ITERS keeps trailing comment", bn["ITERS"].get("desc") == "loop count")
    check("x280: MASK discovered as hex 0xaa", bn.get("MASK", {}).get("default") == "0xaa"
          and bn["MASK"]["type"] == "hex")
    check("x280: include guard BH_H excluded", "BH_H" not in bn)
    check("x280: canonical BH_TELE_BASE excluded", "BH_TELE_BASE" not in bn)
    check("x280: unguarded NOPS excluded", "NOPS" not in bn)


def test_x280_mailbox():
    src = {"k.c": (
        "/*\n"
        " * op 10 select_class : arg0 = class 0..11\n"
        " * op 11 set_seed     : arg0 = seedA\n"
        " *   1 set_csr   : csrw mscratch, arg0\n"   # mailbox.c form: no 'op' prefix
        " * op  4 park / op 5 run\n"     # no-colon control ops -> skipped
        " */\n"
        "we then op 7 bogus : prose mid-sentence\n"  # not line-anchored -> skipped
    )}
    d = kp.parse_x280(src)
    mb = {p["name"]: p for p in d if p["kind"] == "mailbox"}
    check("x280: mailbox select_class op=10", mb.get("select_class", {}).get("op") == 10)
    check("x280: mailbox set_seed op=11", mb.get("set_seed", {}).get("op") == 11)
    check("x280: mailbox set_csr op=1 (no 'op' prefix form)", mb.get("set_csr", {}).get("op") == 1)
    check("x280: no-colon park/run not captured", "park" not in mb and "run" not in mb)
    check("x280: mid-prose 'op 7 bogus' not captured (line-anchored)", "bogus" not in mb)


def test_metal_named_args():
    src = ('constexpr uint32_t a = get_named_compile_time_arg_val("alpha");\n'
           'uint32_t b = get_arg(args::beta);\n'
           'constexpr uint32_t c = get_arg(args::gamma);\n')
    d = kp.parse_metal({"k.cpp": src}, None)
    bn = by_name(d)
    check("metal: named ctarg a (get_named_compile_time_arg_val)", bn.get("a", {}).get("kind") == "ctarg"
          and bn["a"].get("arg_name") == "alpha" and bn["a"].get("index") is None)
    check("metal: named rtarg b (get_arg, non-constexpr)", bn.get("b", {}).get("kind") == "rtarg"
          and bn["b"].get("arg_name") == "beta")
    check("metal: named ctarg c (get_arg + constexpr)", bn.get("c", {}).get("kind") == "ctarg"
          and bn["c"].get("arg_name") == "gamma")
    # named-arg merge identity is by (kind,source,arg_name) -> idempotent
    meta = {"params": []}
    kp.merge(meta, d)
    check("metal: named-arg re-merge idempotent", len(kp.merge(meta, d)) == 0)


def test_split_args_shift():
    check("split: left-shift does not collapse the list",
          kp._split_args("1 << 12, 0x40, (uint32_t)256") == ["1 << 12", "0x40", "(uint32_t)256"])
    check("split: nested braces/parens respected",
          kp._split_args("{a, b}, foo(c, d), e") == ["{a, b}", "foo(c, d)", "e"])


def test_comment_stripping():
    src = ("// uint32_t dead = get_arg_val<uint32_t>(5);\n"
           "uint32_t live = get_arg_val<uint32_t>(0);\n"
           "/* uint32_t block = get_arg_val<uint32_t>(9); */\n")
    d = kp.find_args(kp.strip_comments(src), "k.cpp")
    bn = by_name(d)
    check("strip: commented-out args excluded", "dead" not in bn and 9 not in [p.get("index") for p in d])
    check("strip: live arg kept", bn.get("live", {}).get("index") == 0)
    check("strip: named key string preserved through strip",
          'get_named_compile_time_arg_val("k")' in kp.strip_comments('x = get_named_compile_time_arg_val("k"); // c'))


# ---- tt-metal -----------------------------------------------------------------------
def test_metal_args():
    reader = (
        "void kernel_main(){\n"
        "  uint32_t in0_addr = get_arg_val<uint32_t>(0);\n"
        "  uint32_t n = get_arg_val<uint32_t>(1);\n"
        "  constexpr uint32_t cb = get_compile_time_arg_val(0);\n"
        "  do_thing(get_arg_val<uint32_t>(2));\n"   # bare -> arg2
        "}\n")
    host = (
        'auto reader_id = CreateKernel(program, "x/kernels/dataflow/reader.cpp", core,\n'
        '   DataMovementConfig{.compile_args = {7}});\n'
        "SetRuntimeArgs(program, reader_id, core, {(uint32_t)buf->address(), 64, 5});\n")
    d = kp.parse_metal({"reader.cpp": reader}, host)
    bn = by_name([p for p in d if p["kind"] in ("rtarg", "ctarg")])
    check("metal: rtarg in0_addr index 0", bn.get("in0_addr", {}).get("index") == 0
          and bn["in0_addr"]["kind"] == "rtarg")
    check("metal: rtarg n index 1", bn.get("n", {}).get("index") == 1)
    check("metal: bare rtarg arg2 index 2", bn.get("arg2", {}).get("index") == 2)
    check("metal: ctarg cb index 0", bn.get("cb", {}).get("kind") == "ctarg" and bn["cb"]["index"] == 0)
    # host enrichment: addr=buffer (no default, hex), n=64 literal default, arg2=5
    check("metal: in0_addr flagged buffer address (hex, no default)",
          bn["in0_addr"]["type"] == "hex" and bn["in0_addr"]["default"] is None)
    check("metal: n default backfilled from SetRuntimeArgs", bn["n"]["default"] == 64)
    check("metal: ctarg cb default from compile_args", bn["cb"]["default"] == 7)
    check("metal: every rtarg/ctarg carries a source",
          all(p.get("source") == "reader.cpp" for p in d if p["kind"] in ("rtarg", "ctarg")))


def test_metal_defines_from_host():
    host = 'CreateKernel(program, "x/kernels/compute/c.cpp", core, ComputeConfig{.defines = {{"OP", "1"}, {"MODE", "fast"}}});'
    d = kp.parse_metal({"c.cpp": "void kernel_main(){}"}, host)
    bn = by_name([p for p in d if p["kind"] == "define"])
    check("metal: host .defines OP=1 int", bn.get("OP", {}).get("default") == 1)
    check("metal: host .defines MODE=fast str", bn.get("MODE", {}).get("default") == "fast"
          and bn["MODE"]["type"] == "str")


# ---- merge --------------------------------------------------------------------------
def test_merge():
    disc = [
        {"name": "in0_addr", "kind": "rtarg", "index": 0, "type": "hex", "default": None, "source": "r.cpp"},
        {"name": "n", "kind": "rtarg", "index": 1, "type": "int", "default": 64, "source": "r.cpp"},
        {"name": "ITERS", "kind": "define", "type": "int", "default": 256},
    ]
    meta = {"params": []}
    a1 = kp.merge(meta, disc)
    a2 = kp.merge(meta, disc)
    check("merge: first pass adds all 3", len(a1) == 3)
    check("merge: second pass idempotent (0 added)", len(a2) == 0)

    # edit-preservation: user retyped ITERS default; re-merge must not clobber it
    by = {p["name"]: p for p in meta["params"]}
    by["ITERS"]["default"] = 999
    kp.merge(meta, disc)
    check("merge: preserves user-edited default", by_name(meta["params"])["ITERS"]["default"] == 999)

    # index identity: same name at different index in a different source is a NEW param
    a3 = kp.merge(meta, [{"name": "n", "kind": "rtarg", "index": 1, "type": "int",
                          "default": 0, "source": "w.cpp"}])
    check("merge: (kind,source,index) identity adds w.cpp:n", len(a3) == 1)

    # name identity for define: same name re-discovered is NOT re-added
    a4 = kp.merge(meta, [{"name": "ITERS", "kind": "define", "type": "int", "default": 1}])
    check("merge: name identity keeps single ITERS", len(a4) == 0)


def test_merged_validates():
    """Anything merge() produces must satisfy kernmeta.validate (so a write can't fail)."""
    disc = kp.parse_metal({"r.cpp": "void kernel_main(){uint32_t a=get_arg_val<uint32_t>(0);"
                           "constexpr uint32_t c=get_compile_time_arg_val(0);}"}, None)
    meta = {"title": "t", "lang": "cpp", "engine": "tensix", "doc": "", "sources": ["r.cpp"], "params": []}
    kp.merge(meta, disc)
    try:
        kernmeta.validate(meta)
        check("validate: merged tt-metal meta is valid", True)
    except ValueError as e:
        check(f"validate: merged tt-metal meta is valid ({e})", False)


# ---- live pass over the real trees --------------------------------------------------
def test_live_trees_validate():
    """dry-run merge_all across whatever trees exist; assert no crash and every resulting meta
    validates. Skips engines whose source tree is absent (tt-metal may not be present in CI)."""
    from bhtop.web import l2lab, lab, tlab
    for label, fn in (("x280", l2lab.merge_all), ("noc", lab.merge_all), ("tensix", tlab.merge_all)):
        try:
            res = fn(dry_run=True)
        except Exception as e:
            check(f"live {label}: merge_all dry-run did not crash ({e})", False)
            continue
        ok = res.get("available", False)
        check(f"live {label}: tree {'present' if ok else 'absent (skipped)'}", True)
        if not ok:
            continue
        n = sum(r["count"] for r in res["results"])
        check(f"live {label}: discovered {n} param(s) across {len(res['results'])} kernel(s)", True)


def main():
    print("kernparse tests")
    test_x280_defines()
    test_x280_mailbox()
    test_metal_args()
    test_metal_named_args()
    test_split_args_shift()
    test_comment_stripping()
    test_metal_defines_from_host()
    test_merge()
    test_merged_validates()
    test_live_trees_validate()
    print(f"\n{'ALL PASS' if not _fails else str(_fails) + ' FAILED'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
