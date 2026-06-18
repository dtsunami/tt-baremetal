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


def overlay_kdir(engine, key):
    """The overlay kernel-config dir for a non-bhtop-owned engine: keyed by the top path
    segment of the selected file (the project/example). Returns (kdir, top)."""
    top = (key or "").split("/")[0]
    return os.path.join(OVERLAY, engine, top), top


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
