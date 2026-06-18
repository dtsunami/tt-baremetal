"""
kernconf — per-kernel kernel.json get/put (the raw "config" the JSON editor edits), plus the
overlay location for engines whose source isn't bhtop-owned.

  * x280  : the kernel.json lives IN the working kernel folder (~/bhtop/kernels/x280/<name>/).
  * noc   : source stays in the tt-metal tree (edited in place); the kernel.json is a bhtop-side
    tensix  OVERLAY at ~/bhtop/kernels/<engine>/<project-or-example>/kernel.json — metadata only,
            no source copy, so it never pollutes tt-metal.

raw_get auto-creates a sensible default kernel.json if one doesn't exist yet (so every kernel is
editable). PURE filesystem (no device).
"""
import json
import os

from . import kernmeta

OVERLAY = os.path.expanduser("~/bhtop/kernels")   # working/overlay root (gitignored)


def _safe_overlay(engine, rel):
    """Resolve OVERLAY/<engine>/<rel> (rel may be multi-segment), refusing empty keys and any
    traversal so an overlay write can never escape the per-engine overlay namespace."""
    rel = (rel or "").strip().strip("/")
    if not rel or any(seg in ("", ".", "..") for seg in rel.split("/")):
        raise ValueError(f"invalid kernel key: {rel!r}")
    base = os.path.realpath(os.path.join(OVERLAY, engine))
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and os.path.commonpath([full, base]) != base:
        raise ValueError(f"kernel key escapes the overlay: {rel}")
    return full


def overlay_kdir(engine, key):
    """The overlay kernel-config dir for a non-bhtop-owned engine: keyed by the top path
    segment of the selected file (the project/example). Returns (kdir, top)."""
    top = (key or "").split("/")[0]
    return _safe_overlay(engine, top), top


def overlay_kdir_rel(engine, rel):
    """Overlay dir keyed by a full (sanitized) example-relative path rather than just the top
    segment — needed where a project nests sub-examples that each own a kernels/ dir (tt-metal
    matmul/, profiler/). Returns (kdir, rel). For a flat example rel == top, so the path is
    identical to overlay_kdir."""
    return _safe_overlay(engine, rel), rel


def raw_get(kdir, sources, lang, engine):
    """Read kdir/kernel.json as text, creating a default first if it doesn't exist."""
    os.makedirs(kdir, exist_ok=True)
    p = os.path.join(kdir, kernmeta.META_NAME)
    if not os.path.isfile(p):
        kernmeta.save(kdir, kernmeta.default_meta(kdir, sources, lang, engine))
    with open(p, encoding="utf-8") as fh:
        return {"json": fh.read()}


def raw_put(kdir, text):
    """Validate (JSON + schema) and write kdir/kernel.json from the editor text."""
    data = json.loads(text)            # raises on bad JSON -> 400 at the route edge
    kernmeta.validate(data)
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, kernmeta.META_NAME), "w", encoding="utf-8") as fh:
        fh.write(text)
    return {"ok": True}
