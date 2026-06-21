"""
tensix.isa — machine-readable Tensix (Blackhole) ISA, parsed from tt-llk's assembly.yaml.

Turns the 137-instruction `assembly.yaml` into a JSON-able map the cockpit uses to make
Tensix instructions self-describe: opcode, execution unit, per-operand bit-fields and their
human descriptions. This is what powers the hover tooltips over TT_OP_* tokens in the overlay
editor and the searchable ISA reference panel — the Tensix analogue of the x280 RVV per-slot
telemetry labels ([[project-crt-matmul]]).

Pure host (no device). Parsed once and cached, keyed by the yaml's mtime so a `git pull` in
tt-metal is picked up without a server restart. Bit-field widths are not in the yaml, so each
field's width is derived from the next field's start_bit (the documented approach); the last
operand is capped at the 24-bit operand region, since op_binary owns the top byte (bits 24..31)
of the 32-bit instruction word.
"""
import os
import threading

import yaml

# tt-llk ships the canonical Blackhole ISA here. TT_METAL_HOME wins; fall back to ~/tt-metal.
_REL = os.path.join("tt_metal", "tt-llk", "tt_llk_blackhole", "instructions", "assembly.yaml")

_lock = threading.Lock()
_cache = None          # parsed {mnemonics, ...}
_cache_key = None      # (path, mtime) the cache was built from


def yaml_path():
    """Absolute path to assembly.yaml, or None if tt-llk isn't present."""
    home = os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")
    p = os.path.join(home, _REL)
    return p if os.path.isfile(p) else None


# Coarse category for grouping/filtering in the reference panel. Name-prefix first (most
# reliable), then execution resource. Mirrors the buckets called out in the cockpit plan.
def _category(name, ex_resource):
    n = name.upper()
    if n.startswith(("UNPACR", "PACR", "PACK")):
        return "unpack/pack"
    if n.startswith(("SEM", "STALL", "ATGETM", "ATRELM", "NOP", "DMANOP", "SYNC")):
        return "sync"
    if n.startswith(("MOV", "SHIFT", "ZERO", "TRNSP", "TRANSP", "SETDVALID", "CLEAR")):
        return "data-move"
    if n.startswith(("WRCFG", "RDCFG", "CFG", "SFPCONFIG", "SETC", "RSTDMA", "SET")):
        return "config"
    if n.startswith("SFP"):
        return "vector (SFPU)"
    res = (ex_resource or "").upper()
    if res == "MATH":
        return "compute"
    if res == "SYNC":
        return "sync"
    if res in ("UNPACK", "PACK", "XSEARCH", "XMOV", "THCON"):
        return "unpack/pack"
    return "other"


def _to_int(v):
    """op_binary is '0x26' (str) or 38 (int). Returns int or None."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v), 0)
    except (TypeError, ValueError):
        return None


def _parse_args(raw_args):
    """Normalize one instruction's `arguments` into [{name,start_bit,width,field_type,desc}],
    sorted by start_bit, with width derived from the next field's start_bit (last field capped
    at the 24-bit operand region)."""
    args = []
    for a in raw_args or []:
        if not isinstance(a, dict):
            continue
        sb = a.get("start_bit")
        try:
            sb = int(sb)
        except (TypeError, ValueError):
            sb = None
        desc = a.get("description")
        args.append({
            "name": a.get("name", "?"),
            "start_bit": sb,
            "field_type": a.get("field_type"),
            "desc": " ".join(str(desc).split()) if desc else "",
        })
    placed = sorted([a for a in args if a["start_bit"] is not None], key=lambda a: a["start_bit"])
    for i, a in enumerate(placed):
        nxt = placed[i + 1]["start_bit"] if i + 1 < len(placed) else 24
        a["width"] = max(1, min(nxt, 24) - a["start_bit"]) if nxt > a["start_bit"] else 1
    # append any fields with no start_bit at the end (rare; width unknown)
    for a in args:
        if a["start_bit"] is None:
            a["width"] = None
            placed.append(a)
    return placed


def _build(path):
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    instrs = {}
    for name, body in (doc or {}).items():
        if not isinstance(body, dict):
            continue
        if "op_binary" not in body and "arguments" not in body:
            continue                                   # not an instruction entry
        ex = body.get("ex_resource")
        desc = body.get("description")
        instrs[name] = {
            "name": name,
            "opcode": _to_int(body.get("op_binary")),
            "unit": ex,
            "instr_type": body.get("instrn_type"),
            "src_mask": _to_int(body.get("src_mask")),
            "category": _category(name, ex),
            "desc": " ".join(str(desc).split()) if desc else "",
            "args": _parse_args(body.get("arguments")),
        }
    return {"mnemonics": instrs, "count": len(instrs), "source": path}


def load(force=False):
    """Parsed ISA, cached by (path, mtime). Returns {available, mnemonics, count, source}.
    available=False (with an empty map) when tt-llk's assembly.yaml isn't on disk."""
    global _cache, _cache_key
    path = yaml_path()
    if not path:
        return {"available": False, "mnemonics": {}, "count": 0, "source": None,
                "error": "assembly.yaml not found (set TT_METAL_HOME or install tt-metal)"}
    key = (path, os.path.getmtime(path))
    with _lock:
        if force or _cache is None or _cache_key != key:
            _cache = _build(path)
            _cache_key = key
        out = dict(_cache)
    out["available"] = True
    return out


def one(mnemonic):
    """A single instruction's decoded record, or None. Case-insensitive."""
    m = load().get("mnemonics", {})
    if mnemonic in m:
        return m[mnemonic]
    up = mnemonic.upper()
    for k, v in m.items():
        if k.upper() == up:
            return v
    return None


def encode(mnemonic, **fields):
    """Assemble a 32-bit instruction word: (opcode << 24) | OR of (field_value << start_bit).
    Unknown field names raise; missing fields default to 0. The cockpit's name->word encoder so
    overlays can be written by mnemonic instead of magic numbers."""
    info = one(mnemonic)
    if not info:
        raise ValueError(f"unknown Tensix instruction {mnemonic!r}")
    if info["opcode"] is None:
        raise ValueError(f"{mnemonic} has no opcode in the ISA yaml")
    word = (info["opcode"] & 0xFF) << 24
    known = {a["name"]: a for a in info["args"]}
    for fname, val in fields.items():
        a = known.get(fname)
        if not a or a["start_bit"] is None:
            raise ValueError(f"{mnemonic} has no placed field {fname!r}; have {list(known)}")
        word |= (int(val) & ((1 << (a["width"] or 1)) - 1)) << a["start_bit"]
    return word & 0xFFFFFFFF


if __name__ == "__main__":   # quick self-test: python -m bhtop.tensix.isa
    d = load()
    print(f"available={d['available']} count={d['count']} source={d['source']}")
    mv = one("MVMUL")
    if mv:
        print(f"MVMUL opcode=0x{mv['opcode']:x} unit={mv['unit']} cat={mv['category']}")
        for a in mv["args"]:
            print(f"  [{a['start_bit']:>2}:+{a['width']}] {a['name']:<14} {a['field_type']:<4} {a['desc'][:60]}")
        print(f"  encode(dst=5, clear_dvalid=3) = 0x{encode('MVMUL', dst=5, clear_dvalid=3):08x}")
    cats = {}
    for v in d["mnemonics"].values():
        cats[v["category"]] = cats.get(v["category"], 0) + 1
    print("categories:", cats)
