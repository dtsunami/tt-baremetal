"""
kernparse — discover a kernel's params straight from its source(s), then MERGE them into the
kernel.json so the param schema gets populated instead of hand-typed. The "preprocessor across
toolchains" (see kernmeta) gets an importer: read the source, emit the param the source already
implies, and fold it into the sidecar without clobbering anything you've edited.

Two source flavors, mapped onto the kernmeta param kinds:

  x280 bare-metal (.c/.s/.rs)
    * `#ifndef NAME` / `#define NAME <literal>`  -> a `define` param (skips the canonical regmap
      names — those are the harness map, not a knob; and skips value-less guards, so include
      guards never leak in). Default + type come from the literal; desc from a trailing comment.
    * a `op <N> <name> : <desc>` line in the header comment -> a best-effort `mailbox` param.

  tt-metal (.cpp device kernels + the host .cpp)
    * `get_arg_val<T>(N)`            -> an `rtarg` param (runtime arg, index N), name from the LHS.
    * `get_compile_time_arg_val(N)`  -> a `ctarg` param (compile-time arg, index N).
    * `#ifndef` defines              -> a `define` param (same as x280).
    * best-effort host scan: CreateKernel(...) maps a KernelHandle var -> kernel file, then
      SetRuntimeArgs(...) literals backfill rtarg defaults, and a Config's `.compile_args` /
      `.defines` backfill ctarg / define defaults. Buffer-address args are left default-less.

merge() is idempotent and edit-preserving: a param already in the sidecar (matched by identity —
(kind,source,index) for rtarg/ctarg, (kind,name) otherwise) is left exactly as-is; only genuinely
new discoveries are appended. So re-running a merge after you've tuned a default is safe.

PURE (no device, no FastAPI, no regmap import — the caller passes the names to skip) so it's
unit-testable in isolation.
"""
import os
import re

# kind ordering for stable output (the order params are appended in a fresh sidecar)
_KIND_RANK = {"define": 0, "ctarg": 1, "rtarg": 2, "mailbox": 3, "deploy": 4}


# ---- literal parsing ----------------------------------------------------------------
def parse_c_int(s):
    """Parse a C integer literal -> (value:int|None, is_hex:bool). Strips u/U/l/L suffixes and a
    surrounding paren. Returns (None, False) for anything that isn't a plain integer literal (an
    expression, a name, empty) so the caller can fall back to a string/skip."""
    if s is None:
        return None, False
    t = s.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    if not t:
        return None, False
    t = re.sub(r"[uUlL]+$", "", t)
    is_hex = t.lower().startswith(("0x", "-0x"))
    try:
        return int(t, 0), is_hex
    except ValueError:
        return None, False


def _trailing_comment(c1, c2):
    """Pick the trailing comment text from a /* */ (c1) or // (c2) capture, trimmed."""
    txt = (c1 or c2 or "").strip()
    return re.sub(r"\s+", " ", txt)


def strip_comments(s):
    """Blank out // and /* */ comments (preserving newlines + string/char literals) so the arg
    regexes don't capture commented-out calls. String literals are KEPT — the named-arg APIs carry
    the param key inside a "string", and CreateKernel carries the kernel path as a string."""
    out, i, n, st = [], 0, len(s), None      # st: None|'str'|'char'|'line'|'block'
    while i < n:
        c = s[i]
        d = s[i + 1] if i + 1 < n else ""
        if st is None:
            if c == "/" and d == "/":
                st = "line"; out.append("  "); i += 2; continue
            if c == "/" and d == "*":
                st = "block"; out.append("  "); i += 2; continue
            if c == '"':
                st = "str"
            elif c == "'":
                st = "char"
            out.append(c); i += 1; continue
        if st == "line":
            if c == "\n":
                st = None; out.append("\n")
            else:
                out.append(" ")
            i += 1; continue
        if st == "block":
            if c == "*" and d == "/":
                st = None; out.append("  "); i += 2; continue
            out.append("\n" if c == "\n" else " "); i += 1; continue
        # inside a string/char literal — copy verbatim, honoring backslash escapes
        out.append(c)
        if c == "\\":
            out.append(d); i += 2; continue
        if (st == "str" and c == '"') or (st == "char" and c == "'"):
            st = None
        i += 1
    return "".join(out)


# ---- shared: #ifndef / #define knobs ------------------------------------------------
_IFNDEF_DEF = re.compile(
    r"#ifndef[ \t]+(?P<name>\w+)[ \t]*\r?\n"
    r"[ \t]*#define[ \t]+(?P=name)\b[ \t]*(?P<val>[^\n/\r]*?)[ \t]*"
    r"(?:/\*(?P<c1>.*?)\*/|//(?P<c2>[^\n]*))?[ \t]*\r?\n",
    re.M,
)


def find_defines(text, skip=()):
    """Discovered `define` params from `#ifndef NAME` / `#define NAME <literal>` pairs. Skips names
    in `skip` (the canonical regmap) and value-less guards (include guards / bare toggles)."""
    skip = set(skip)
    out = []
    seen = set()
    for m in _IFNDEF_DEF.finditer(text):
        name = m.group("name")
        if name in skip or name in seen:
            continue
        raw = (m.group("val") or "").strip()
        if not raw:                       # value-less #ifndef/#define = include guard / toggle
            continue
        seen.add(name)
        val, is_hex = parse_c_int(raw)
        if val is None:                   # an expression literal -> keep the raw text as a string
            p = {"name": name, "kind": "define", "type": "str", "default": raw}
        else:
            p = {"name": name, "kind": "define", "type": "hex" if is_hex else "int",
                 "default": (hex(val) if is_hex else val)}
        desc = _trailing_comment(m.group("c1"), m.group("c2"))
        if desc:
            p["desc"] = desc
        out.append(p)
    return out


# ---- x280: mailbox ops from the header comment --------------------------------------
# A documented-op line, anchored to the start of a (comment) line so prose can't match mid-sentence.
# The leading `op` is optional — both conventions occur: `op 10 select_class : …` (vec_virus) and
# `1 set_csr : …` (mailbox.c). The colon is required (clear intent).
_MBOX_LINE = re.compile(
    r"^[ \t]*\*?[ \t]*(?:op[ \t]+)?(?P<op>\d+)[ \t]+(?P<name>[A-Za-z]\w*)[ \t]*:[ \t]*(?P<desc>[^\n\r]*)",
    re.M)


def find_mailbox(text):
    """Best-effort `mailbox` params from documented `[op] <N> <name> : <desc>` lines in a header
    comment. Conservative: only the colon form (clear intent); type defaults to int, default 0 —
    refine choices/vals by hand in the JSON editor."""
    out = []
    seen = set()
    for m in _MBOX_LINE.finditer(text):
        name = m.group("name")
        if name in seen:
            continue
        seen.add(name)
        desc = re.sub(r"\s+", " ", m.group("desc").strip().rstrip("*/ ")).strip()
        p = {"name": name, "kind": "mailbox", "op": int(m.group("op")),
             "type": "int", "default": 0}
        if desc:
            p["desc"] = desc
        out.append(p)
    return out


def parse_x280(sources, skip_defines=()):
    """Discover params for an x280 bare-metal kernel from its source files.
    `sources` = {filename: text}. Defines and documented mailbox ops are scanned across every file
    (de-duped by name), so a kernel split over multiple files still surfaces them all."""
    discovered = []
    for name, text in sources.items():
        discovered += find_defines(text, skip_defines)
    for name, text in sources.items():
        discovered += find_mailbox(text)
    return _dedup(discovered)


# ---- tt-metal: runtime / compile-time args ------------------------------------------
_RTARG = re.compile(
    r"(?:(?:const|constexpr|volatile)\s+)*"            # optional qualifiers
    r"(?:[\w:]+\s+)?"                                  # optional LHS type
    r"(?P<name>\w+)\s*=\s*"                            # the variable receiving the value
    r"get_arg_val\s*<\s*(?P<t>[\w:]+)\s*>\s*\(\s*(?P<idx>\d+)\s*\)")
_RTARG_BARE = re.compile(r"get_arg_val\s*<\s*(?P<t>[\w:]+)\s*>\s*\(\s*(?P<idx>\d+)\s*\)")
_CTARG = re.compile(
    r"(?:(?:const|constexpr|volatile)\s+)*"
    r"(?:[\w:]+\s+)?"
    r"(?P<name>\w+)\s*=\s*"
    r"get_compile_time_arg_val\s*\(\s*(?P<idx>\d+)\s*\)")
_CTARG_BARE = re.compile(r"get_compile_time_arg_val\s*\(\s*(?P<idx>\d+)\s*\)")
# Named-arg APIs (metal-2.0 / experimental kernel_args): no positional index — the key is the
# binding name. `constexpr ... = get_named_compile_time_arg_val("k")` is always compile-time;
# `[constexpr] ... = get_arg(args::k)` is compile-time when constexpr, else runtime.
_NAMED_CT = re.compile(
    r"(?P<qual>(?:[\w:]+\s+)*?)(?P<name>\w+)\s*=\s*"
    r'get_named_compile_time_arg_val\s*\(\s*"(?P<key>[^"]+)"\s*\)')
_GET_ARG = re.compile(
    r"(?P<qual>(?:[\w:]+\s+)*?)(?P<name>\w+)\s*=\s*"
    r"get_arg\s*\(\s*args::(?P<key>\w+)\s*\)")

_ADDR_HINT = re.compile(r"\b\w*(?:addr|base|ptr)\w*", re.I)


def _arg_type(template_type, name):
    """int by default; hex when the C++ type is wide (64-bit) or the name reads like an address."""
    if "64" in (template_type or "") or _ADDR_HINT.search(name or ""):
        return "hex"
    return "int"


def _named(name, key, kind, source):
    """A named tt-metal arg (no positional index): displayed by its LHS var name, identified +
    host-matched by its binding `key`."""
    return {"name": name, "kind": kind, "arg_name": key, "type": _arg_type("", name),
            "default": None, "source": source}


def find_args(text, source):
    """Discovered rtarg + ctarg params from one device kernel's text. Handles positional args
    (get_arg_val<T>(N) / get_compile_time_arg_val(N)) keyed by index, and named args
    (get_named_compile_time_arg_val("k") / get_arg(args::k)) keyed by binding name. The LHS var
    names the param; bare positional calls fall back to arg<N>/ctarg<N> so the slot still surfaces."""
    rt, ct, named = {}, {}, []
    for m in _RTARG.finditer(text):
        i = int(m.group("idx"))
        rt[i] = {"name": m.group("name"), "kind": "rtarg", "index": i,
                 "type": _arg_type(m.group("t"), m.group("name")), "default": None, "source": source}
    for m in _RTARG_BARE.finditer(text):
        i = int(m.group("idx"))
        rt.setdefault(i, {"name": f"arg{i}", "kind": "rtarg", "index": i,
                          "type": _arg_type(m.group("t"), ""), "default": None, "source": source})
    for m in _CTARG.finditer(text):
        i = int(m.group("idx"))
        ct[i] = {"name": m.group("name"), "kind": "ctarg", "index": i,
                 "type": "int", "default": None, "source": source}
    for m in _CTARG_BARE.finditer(text):
        i = int(m.group("idx"))
        ct.setdefault(i, {"name": f"ctarg{i}", "kind": "ctarg", "index": i,
                          "type": "int", "default": None, "source": source})
    seen_keys = set()
    for m in _NAMED_CT.finditer(text):
        if ("ctarg", m.group("key")) not in seen_keys:
            seen_keys.add(("ctarg", m.group("key")))
            named.append(_named(m.group("name"), m.group("key"), "ctarg", source))
    for m in _GET_ARG.finditer(text):
        kind = "ctarg" if "constexpr" in m.group("qual") else "rtarg"
        if (kind, m.group("key")) not in seen_keys:
            seen_keys.add((kind, m.group("key")))
            named.append(_named(m.group("name"), m.group("key"), kind, source))
    return [ct[i] for i in sorted(ct)] + [rt[i] for i in sorted(rt)] + named


# ---- tt-metal: best-effort host enrichment ------------------------------------------
_CREATE_KERNEL = re.compile(r"(?:(?P<handle>\w+)\s*=\s*)?CreateKernel\s*\((?P<body>.*?)\)\s*;", re.S)
_KPATH = re.compile(r'"([^"]*\.cpp)"')
_SET_RT = re.compile(r"SetRuntimeArgs\s*\(\s*\w+\s*,\s*(?P<handle>\w+)\s*,[^,]*,\s*\{(?P<vals>.*?)\}\s*\)", re.S)
_DEFINE_PAIR = re.compile(r'\{\s*"(?P<k>[^"]+)"\s*,\s*"?(?P<v>[^",}]*)"?\s*\}')


def _brace_group(s, i):
    """s[i] must be '{'; return the substring through its matching '}' (inclusive), brace-balanced."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return s[i:]


def _field_group(body, field):
    """The brace-balanced `{ ... }` value of a `.field = {...}` struct member, or None."""
    fi = body.find(field)
    if fi < 0:
        return None
    bi = body.find("{", fi)
    return _brace_group(body, bi) if bi >= 0 else None


def _resolve_src(path, by_source):
    """Map a CreateKernel path to its key in by_source. Device sources are keyed by basename, but
    nested examples that collide on basename are keyed by a relative path — so match by basename and
    fall back to the longest path-suffix match; return None if still ambiguous (skip enrichment)."""
    base = os.path.basename(path)
    cands = [k for k in by_source if os.path.basename(k) == base]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    np = path.replace("\\", "/")
    best = max(cands, key=lambda k: len(os.path.commonprefix([k.replace("\\", "/")[::-1], np[::-1]])))
    others = [k for k in cands if k != best]
    return best if all(  # require the suffix match to be unambiguous
        len(os.path.commonprefix([k.replace("\\", "/")[::-1], np[::-1]]))
        < len(os.path.commonprefix([best.replace("\\", "/")[::-1], np[::-1]])) for k in others) else None


def _split_args(s):
    """Split a brace-list body into top-level comma items, ignoring commas inside nested ()/[]/{}.
    Angle brackets are NOT counted — `1 << N` (left-shift) is far more common in arg lists than a
    bare template, and counting `<`/`>` would collapse the whole list on the first shift."""
    items, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return items


def _value_default(expr):
    """A SetRuntimeArgs / compile_args expression -> (default, is_address). A plain integer literal
    becomes the default; anything referencing a buffer/address is a runtime pointer (no static
    default)."""
    e = expr.strip()
    # strip a leading C-style cast e.g. (uint32_t)expr
    e2 = re.sub(r"^\(\s*[\w:]+\s*\)\s*", "", e)
    val, is_hex = parse_c_int(e2)
    if val is not None:
        return (hex(val) if is_hex else val), False
    # only a real buffer-address call counts as an address — NOT a bare `->` member access
    # (e.g. cfg->num_tiles) or an 'addr' substring in an unrelated name.
    if re.search(r"\.address\s*\(\)|->\s*address\s*\(\)|\bbuffer\b", e, re.I):
        return None, True
    return None, False


def enrich_from_host(host_text, by_source):
    """Backfill rtarg/ctarg/define defaults from the host .cpp. `by_source` = {device_filename:
    [params]} (mutated in place). Never raises — host parsing is heuristic, so any failure just
    leaves the device-derived params as they were."""
    if not host_text:
        return []
    extra_defines = []
    try:
        # 1) CreateKernel -> device kernel basename (+ body). handle (if the call is assigned)
        #    maps to SetRuntimeArgs; the body carries inline compile_args / defines either way.
        handle_src = {}
        kernel_bodies = []                       # [(src_basename, body)]
        for m in _CREATE_KERNEL.finditer(host_text):
            paths = _KPATH.findall(m.group("body"))
            if not paths:
                continue
            src = _resolve_src(paths[0], by_source)
            kernel_bodies.append((src, m.group("body")))
            if m.group("handle") and src:
                handle_src[m.group("handle")] = src

        # 2) SetRuntimeArgs(...) literals -> rtarg defaults for that handle's kernel
        for m in _SET_RT.finditer(host_text):
            src = handle_src.get(m.group("handle"))
            if not src or src not in by_source:
                continue
            rt = sorted((p for p in by_source[src] if p["kind"] == "rtarg" and p.get("index") is not None),
                        key=lambda p: p["index"])
            vals = _split_args(m.group("vals"))
            for p, expr in zip(rt, vals):
                default, is_addr = _value_default(expr)
                if default is not None and p.get("default") is None:
                    p["default"] = default
                    if isinstance(default, str) and default.startswith("0x"):
                        p["type"] = "hex"
                elif is_addr:
                    p["type"] = "hex"
                    p.setdefault("desc", "buffer address — set by host at runtime")

        # 3) inline compile_args -> ctarg defaults; inline .defines -> define defaults
        for src, body in kernel_bodies:
            ca = _field_group(body, ".compile_args")
            if ca and src in by_source:
                ct = sorted((p for p in by_source[src] if p["kind"] == "ctarg" and p.get("index") is not None),
                            key=lambda p: p["index"])
                for p, expr in zip(ct, _split_args(ca[1:-1])):
                    default, _ = _value_default(expr)
                    if default is not None and p.get("default") is None:
                        p["default"] = default
                        if isinstance(default, str) and default.startswith("0x"):
                            p["type"] = "hex"
            dg = _field_group(body, ".defines")
            if dg:
                for pm in _DEFINE_PAIR.finditer(dg):
                    val, is_hex = parse_c_int(pm.group("v"))
                    extra_defines.append({
                        "name": pm.group("k"), "kind": "define",
                        "type": "hex" if is_hex else ("int" if val is not None else "str"),
                        "default": (hex(val) if is_hex else val) if val is not None else pm.group("v").strip(),
                        "source": src})
    except Exception:
        return extra_defines
    return extra_defines


def parse_metal(device_sources, host_text=None):
    """Discover params for a tt-metal example/project.
    `device_sources` = {filename: text} for the compute/dataflow kernels; `host_text` = the host
    .cpp (optional). Returns a flat param list (positional rtarg/ctarg carry a `source` + `index`,
    named args a `source` + `arg_name`, so the same name in two kernels never collides). Args are
    scanned over comment-stripped text (so commented-out calls don't leak); defines keep raw text
    so their trailing-comment descriptions survive."""
    by_source = {}
    discovered = []
    for name, text in device_sources.items():
        args = find_args(strip_comments(text), name)
        defs = [dict(d, source=name) for d in find_defines(text)]
        by_source[name] = args
        discovered += args + defs
    discovered += enrich_from_host(strip_comments(host_text) if host_text else host_text, by_source)
    return _dedup(discovered)


# ---- merge --------------------------------------------------------------------------
def _ident(p):
    """Identity for de-dup + merge. tt-metal args are keyed within their source kernel by positional
    index (names repeat across reader/writer/compute) or by binding name for named args; everything
    else is name-keyed."""
    if p.get("kind") in ("rtarg", "ctarg"):
        slot = p["index"] if p.get("index") is not None else ("@" + str(p.get("arg_name") or p.get("name")))
        return (p["kind"], p.get("source"), slot)
    return (p.get("kind"), p.get("name"))


def _sort_key(p):
    # positional args sort by index; named args (no index) sort after, by binding name
    idx = p.get("index")
    return (_KIND_RANK.get(p.get("kind"), 9), p.get("source") or "",
            (0, idx) if isinstance(idx, int) else (1, p.get("arg_name") or p.get("name") or ""))


def _dedup(params):
    out, seen = [], set()
    for p in sorted(params, key=_sort_key):
        k = _ident(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def merge(meta, discovered):
    """Fold `discovered` params into `meta['params']` in place, idempotently. Existing params (by
    identity) are left untouched — your hand-tuned defaults/descs win — and only new discoveries
    are appended (stably ordered). Returns the list of params that were added."""
    params = meta.setdefault("params", [])
    have = {_ident(p) for p in params}
    added = []
    for d in _dedup(discovered):
        if _ident(d) not in have:
            params.append(d)
            have.add(_ident(d))
            added.append(d)
    return added
