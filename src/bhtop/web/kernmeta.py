"""
kernmeta — per-kernel metadata (`kernel.json`): the knobs a kernel exposes, with descriptions,
applied by KIND at deploy/run time. This is the user-facing half of the "preprocessor across
toolchains": a kernel declares its params once, and each is routed to the right mechanism.

  * define  — a compile-time constant, injected as -D / --defsym / rust const on top of the
              canonical regmap map (see l2cpu.toolchain._defmap). Only takes effect if the
              source leaves the macro #ifndef-guarded.
  * deploy  — a field of the deploy request (tile, hart, all_harts, addr).
  * mailbox — a live /api/l2/cmd op (op, arg0) sent AFTER load (cooperative steering; no recompile).
  * rtarg   — a tt-metal RUNTIME arg (get_arg_val<T>(index)); set by the host at run. Carries an
              `index` (+ `source` = device kernel file). Documentation/UI only — the tt-metal host
              owns the apply path, so route() does not bucket it. Discovered by kernparse.
  * ctarg   — a tt-metal COMPILE-TIME arg (get_compile_time_arg_val(index)); same shape as rtarg.

A kernel folder holds its source(s) + kernel.json. Kernels without a sidecar fall back to a
default meta (just the deploy knobs) so the param dialog always has tile/hart. PURE (no device,
no FastAPI) so it's unit-testable.

kernel.json shape:
    {
      "title": "vec_virus", "lang": "c", "engine": "x280",
      "doc": "one-line description shown in the dialog header",
      "sources": ["vec_virus.c"],          # entry = sources[0]
      "params": [ {param}, ... ],
      "src_map": {"file.cpp": "/abs/tt-metal/path"}   # NOC/TENSIX only; null otherwise
    }
  {param} = {"name","kind"(define|deploy|mailbox|rtarg|ctarg),"type"(int|hex|enum|bool|str),
             "default","desc","op"?(mailbox),"index"?(rtarg/ctarg),"source"?(tt-metal device file),
             "choices"?(enum labels),"vals"?(enum arg0 ints),"min"?,"max"?}
"""
import json
import os

from ..l2cpu.regmap import CODE_ADDR

META_NAME = "kernel.json"
KINDS = ("define", "deploy", "mailbox", "rtarg", "ctarg")
TYPES = ("int", "hex", "enum", "bool", "str")

# The deploy knobs every X280 kernel has, even with no sidecar — so tile/hart always show.
# Both are fixed sets (their `choices` come from the hardware): one tile, but ANY grouping of
# harts (pick one, a subset, or all four = the whole tile). `multi` makes hart a set selector.
DEFAULT_DEPLOY = [
    {"name": "tile", "kind": "deploy", "type": "int", "choices": [0, 1, 2, 3], "default": 0,
     "desc": "which L2CPU tile to run on"},
    {"name": "hart", "kind": "deploy", "type": "int", "choices": [0, 1, 2, 3], "multi": True,
     "default": [0], "desc": "which hart(s) to load onto — pick any grouping (all four = whole tile)"},
    {"name": "addr", "kind": "deploy", "type": "hex", "default": hex(CODE_ADDR),
     "desc": "load address (default user-code window, above the data blocks)"},
]


def _coerce_one(t, value, param):
    if value is None:
        value = param.get("default")
    if t == "int":
        return int(value)
    if t == "hex":
        return value if isinstance(value, int) else int(str(value), 0)
    if t == "bool":
        return bool(value) if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
    if t == "enum":
        choices = [str(c) for c in (param.get("choices") or [])]
        v = str(value)
        if choices and v not in choices:
            raise ValueError(f"{param['name']}={v!r} not in {choices}")
        return v
    return str(value)


def coerce(param, value):
    """Coerce a raw value to the param's type. A `multi` param returns a list (any grouping of
    its choices). Raises ValueError on a bad value."""
    t = param.get("type", "int")
    try:
        if param.get("multi"):
            seq = value if isinstance(value, (list, tuple)) else ([] if value in (None, "") else [value])
            return [_coerce_one(t, x, param) for x in seq]
        return _coerce_one(t, value, param)
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad value for {param.get('name')}: {e}")


def validate(meta):
    """Normalize + sanity-check a meta dict. Raises ValueError on a malformed param."""
    if not isinstance(meta, dict):
        raise ValueError("kernel.json must be an object")
    params = meta.get("params") or []
    for p in params:
        if "name" not in p:
            raise ValueError("param missing 'name'")
        if p.get("kind") not in KINDS:
            raise ValueError(f"param {p['name']}: kind must be one of {KINDS}")
        if p.get("type", "int") not in TYPES:
            raise ValueError(f"param {p['name']}: type must be one of {TYPES}")
        if p["kind"] == "mailbox" and "op" not in p:
            raise ValueError(f"param {p['name']}: mailbox param needs an 'op'")
        if p["kind"] in ("rtarg", "ctarg"):
            idx = p.get("index")
            if idx is None:                 # named tt-metal arg (no positional index) — needs a key
                if not (p.get("arg_name") or p.get("name")):
                    raise ValueError(f"param {p['name']}: {p['kind']} needs an 'index' or 'arg_name'")
            elif isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                raise ValueError(f"param {p['name']}: {p['kind']} 'index' must be an integer >= 0")
        if p.get("type") == "enum" and not p.get("choices"):
            raise ValueError(f"param {p['name']}: enum param needs 'choices'")
    return meta


def default_meta(kdir, sources, lang, engine="x280"):
    """Synthesize a minimal meta for a folder lacking a kernel.json: title from the folder, the
    discovered sources (entry = first), and — for x280 only — the tile/hart deploy knobs. Other
    engines start with no params (the JSON editor lets you add your own)."""
    params = [dict(p) for p in DEFAULT_DEPLOY] if engine == "x280" else []
    return {"title": os.path.basename(kdir.rstrip("/")), "lang": lang, "engine": engine,
            "doc": "", "sources": list(sources), "params": params, "src_map": None}


def load(kdir, sources=None, lang="c", engine="x280"):
    """Load + validate kdir/kernel.json, or synthesize a default. `sources`/`lang`/`engine` seed
    the default when no sidecar exists. Always returns a dict with a 'params' list."""
    p = os.path.join(kdir, META_NAME)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            meta = json.load(fh)
        validate(meta)
        if engine == "x280":            # guarantee the deploy knobs are present (x280 only)
            have = {q["name"] for q in meta.get("params", [])}
            meta.setdefault("params", [])
            for d in DEFAULT_DEPLOY:
                if d["name"] not in have:
                    meta["params"].append(dict(d))
        return meta
    return default_meta(kdir, sources or [], lang, engine)


def save(kdir, meta):
    """Write kernel.json (validated) into the kernel folder."""
    validate(meta)
    with open(os.path.join(kdir, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def defaults(meta):
    """{name: default} for every param (already type-correct for the UI)."""
    return {p["name"]: p.get("default") for p in meta.get("params", [])}


def _arg0(param, value):
    """Compute the mailbox arg0 for a param value. enum -> vals[idx] (or idx); hex/int -> int;
    bool -> 1/0."""
    v = coerce(param, value)
    t = param.get("type", "int")
    if t == "enum":
        idx = (param["choices"]).index(v)
        vals = param.get("vals")
        return int(vals[idx]) if vals else idx
    if t == "bool":
        return 1 if v else 0
    return int(v)


def route(meta, values):
    """Split user-supplied {name: value} into the three application buckets:
        {"defines": {NAME: int}, "deploy": {name: value}, "mailbox": [{name, op, arg0}]}
    Missing values fall back to the param default. Unknown names are ignored."""
    byname = {p["name"]: p for p in meta.get("params", [])}
    out = {"defines": {}, "deploy": {}, "mailbox": []}
    for name, p in byname.items():
        raw = values.get(name, p.get("default")) if values else p.get("default")
        if p["kind"] == "define":
            out["defines"][name] = coerce(p, raw)
        elif p["kind"] == "deploy":
            out["deploy"][name] = coerce(p, raw)
        elif p["kind"] == "mailbox":
            out["mailbox"].append({"name": name, "op": int(p["op"]), "arg0": _arg0(p, raw)})
        # rtarg / ctarg are tt-metal host-owned (documentation only) — not routed here
    return out
