"""
inspector — identify the RUNNING tt-metal kernels by their JIT build hash, instead of
guessing the newest cache dir by mtime (the old tlab_disasm heuristic).

tt-metal's Inspector (enabled via TT_METAL_INSPECTOR — see metal._env) writes a dump under
$TT_METAL_HOME/generated/inspector/:
  * kernels.yaml            — every compiled kernel of the live programs:
                              {watcher_kernel_id, name, path(.../<hash>/), source, program_id}
  * mesh_workloads_log.yaml — mesh_workload_add_program events: {program_id, coordinates:[[x,y]]}

The trailing dir of a kernel's `path` IS its build hash (a hash of source + compile args +
arch) — the stable identity tt-metal itself uses. We key running kernels by that hash, group
them into the *program* they belong to, attach the Tensix coords the program was placed on,
and expose a source -> build map so the device tree can badge a file **running** / **stale**.

A user PROGRAM spans several kernel hash dirs — one per engine role:
  reader (DM)  -> <hash>/ncrisc/ncrisc.elf
  writer (DM)  -> <hash>/brisc/brisc.elf
  compute      -> <hash>/{trisc0,trisc1,trisc2}/*.elf
so engine_elfs() assembles all five ELFs *across* a program's dirs by role — which is why
reader/writer disassembly now appears (the old single-dir scan missed them).

Pure + host-side (no device, no FastAPI). Reads the freshest inspector dir among the live
$TT_METAL_HOME location (where runs write, cwd=$TT_METAL_HOME) and the repo snapshot.
"""
import os

import yaml

from .. import metal

# command-queue / dispatch infra kernels — present in every run, not user workloads
_INFRA_PREFIX = "tt_metal/impl/dispatch/"

# engine role -> (kernel-dir subdir, elf basename). Matches metal._TRISC_ROLE + tlab_disasm.
_ROLE_SUBDIR = {
    "reader": "ncrisc",
    "unpack": "trisc0",
    "math":   "trisc1",
    "pack":   "trisc2",
    "writer": "brisc",
}


def _candidate_dirs():
    """Inspector dirs to consider, freshest wins: the live one under $TT_METAL_HOME (where a
    run with cwd=$TT_METAL_HOME writes it) and the committed repo snapshot (offline/dev)."""
    out = []
    h = metal.metal_home()
    if h:
        out.append(os.path.join(h, "generated", "inspector"))
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out.append(os.path.join(pkg, "generated", "inspector"))   # .../bhtop/generated/inspector
    return out


def _inspector_dir():
    """The inspector dir with the most recently written kernels.yaml, or None."""
    best, best_mt = None, -1.0
    for d in _candidate_dirs():
        kf = os.path.join(d, "kernels.yaml")
        if os.path.exists(kf):
            mt = os.path.getmtime(kf)
            if mt > best_mt:
                best, best_mt = d, mt
    return best


def _load(d, name):
    try:
        with open(os.path.join(d, name)) as fh:
            return yaml.safe_load(fh) or []
    except (OSError, yaml.YAMLError):
        return []


def _program_coords(idir):
    """program_id -> [[x,y], ...] from the mesh-workload add_program events."""
    coords = {}
    for ev in _load(idir, "mesh_workloads_log.yaml"):
        body = ev.get("mesh_workload_add_program") if isinstance(ev, dict) else None
        if body and "program_id" in body:
            coords[body["program_id"]] = body.get("coordinates", [])
    return coords


def read():
    """Snapshot of the live Inspector dump. Returns {ok, dir, kernels[], programs{}, by_source{}}.

    kernels: every kernel (infra + user) with {watcher_kernel_id, name, source, basename,
             hash, path, program_id, role, infra}.
    programs: program_id -> {coords, user, kernels:[watcher_kernel_id...]} for USER programs.
    by_source: basename -> the user kernel build (latest program wins) for fast tree badging.
    """
    idir = _inspector_dir()
    if not idir:
        return {"ok": False, "error": "no Inspector dump found — run a tt-metal kernel first "
                "(needs TT_METAL_INSPECTOR=1, set by metal._env).", "kernels": [],
                "programs": {}, "by_source": {}}

    coords = _program_coords(idir)
    kernels = []
    for ent in _load(idir, "kernels.yaml"):
        k = ent.get("kernel") if isinstance(ent, dict) else None
        if not k:
            continue
        src = k.get("source", "")
        path = (k.get("path") or "").rstrip("/")
        kernels.append({
            "watcher_kernel_id": k.get("watcher_kernel_id"),
            "name": k.get("name", ""),
            "source": src,
            "basename": os.path.basename(src),
            "hash": os.path.basename(path),
            "path": path,
            "program_id": k.get("program_id"),
            "role": _role_of(path),
            "infra": src.startswith(_INFRA_PREFIX),
        })

    programs, by_source = {}, {}
    for k in kernels:
        if k["infra"]:
            continue
        pid = k["program_id"]
        p = programs.setdefault(pid, {"coords": coords.get(pid, []), "kernels": []})
        p["kernels"].append(k["watcher_kernel_id"])
        # latest program wins (runs are sequential -> program_id increases)
        prev = by_source.get(k["basename"])
        if prev is None or (pid is not None and prev["program_id"] is not None and pid >= prev["program_id"]):
            by_source[k["basename"]] = k

    return {"ok": True, "dir": idir, "kernels": kernels, "programs": programs,
            "by_source": by_source}


def _role_of(path):
    """Infer an engine role from which subdir the kernel dir actually contains."""
    if not os.path.isdir(path):
        return None
    subs = set(os.listdir(path))
    if "ncrisc" in subs:
        return "reader"
    if "brisc" in subs:
        return "writer"
    if "trisc1" in subs:
        return "compute"
    return None


def latest_user_program(snap=None):
    """(program_id, kernels[]) of the most recent USER program, or (None, []).

    'Most recent' = highest program_id (runs are sequential). This is the program the last
    Run executed — the one the disasm / running views care about."""
    snap = snap or read()
    user = [k for k in snap["kernels"] if not k["infra"] and k["program_id"] is not None]
    if not user:
        return None, []
    pid = max(k["program_id"] for k in user)
    return pid, [k for k in user if k["program_id"] == pid]


def by_watcher_id(snap=None):
    """watcher_kernel_id -> kernel dict {name, source, basename, hash, path, program_id, role, infra}.

    This is the join key the device uses: a Tensix launch_msg carries watcher_kernel_ids[proc],
    so reading those off a core and looking them up here tells you WHICH kernel (name/source/build
    hash) is loaded on each engine of that core. Returns {} when no Inspector dump exists yet."""
    snap = snap or read()
    return {k["watcher_kernel_id"]: k for k in snap.get("kernels", [])
            if k.get("watcher_kernel_id") is not None}


def engine_elfs(kernels):
    """role -> elf path, assembled ACROSS a program's kernel dirs (compute dir holds the trisc
    ELFs; the reader/writer DM kernels live in their own ncrisc/brisc dirs). Missing engines
    are simply absent from the map."""
    out = {}
    for role, sub in _ROLE_SUBDIR.items():
        for k in kernels:
            elf = os.path.join(k["path"], sub, f"{sub}.elf")
            if os.path.exists(elf):
                out[role] = elf
                break
    return out
