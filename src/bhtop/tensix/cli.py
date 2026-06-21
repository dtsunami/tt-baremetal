"""
bhtop-tensix — read/poke a Tensix worker kernel's runtime args in L1 over the NoC, re-trigger go,
and watch L1 live. The terminal twin of the web launch cockpit (see TENSIX_ABI.md); scriptable for
automation. Works in noc0 (x, y) coords like the rest of bhtop.

  bhtop-tensix cores                          # list Tensix worker cores (noc0)
  bhtop-tensix snapshot 1 2                    # decode the live launch mailbox on core (1,2)
  bhtop-tensix rta 1 2 DM0                     # read DM0's runtime args
  bhtop-tensix poke 1 2 DM0 0x40 64 --offset 0 # write runtime-arg words (no recompile)
  bhtop-tensix go 1 2                          # re-issue go (re-run with the current RTAs)
  bhtop-tensix peek 1 2 0x10000 8             # raw L1 read (8 words)
  bhtop-tensix pokeword 1 2 0x10000 0xdead    # raw L1 write
  bhtop-tensix watch 1 2 0x10000 8 --hz 5     # live L1 window — watch your kernel run

Resident BOOTLOADER console ([[tensix-bootloader]]) — drive the resident kernel deployed by the
metal launcher (poke params / hot-swap code overlays, no relaunch/reset). The bootloader is resident
on ALL 5 baby RISCs of each tile (BRISC,NCRISC,TRISC0,TRISC1,TRISC2) — each owns its own L1 mailbox.
Select with --risc N|NAME (default BRISC) or --all-riscs; x y, or --all = whole grid:
  bhtop-tensix bl-status 1 2 --all-riscs       # decode all 5 RISC mailboxes on core 1,2
  bhtop-tensix bl-status --all                 # one-line per resident core, grid-wide (BRISC)
  bhtop-tensix bl-param 1 2 0 100000 --risc NCRISC   # poke PARAM[0] on NCRISC (live, no re-go)
  bhtop-tensix bl-stage 1 2 overlay.bin -s A   # NoC-write a compiled overlay into BRISC code slot A
  bhtop-tensix bl-exec 1 2 -s A --wait         # invalidate i$ + call slot A; wait for ack (BRISC-only:
                                               #   NCRISC/TRISC can't fetch from L1 -> exec wedges them)
  bhtop-tensix bl-watch 1 2 --risc TRISC1      # live heartbeat + telemetry for TRISC1
  bhtop-tensix bl-halt --all --all-riscs        # break every resident loop (all cores, all 5 RISCs)

Rides a tt-metal-loaded program: runtime args are pokeable, compile-time args are baked in. Opens
its own tt-exalens context; Tensix L1 over NoC is the safe surface (not the ARC/PCIe hang hazard).
Add --json to snapshot/rta/peek for machine-readable output.
"""
import argparse
import json
import sys
import time

from . import abi
from .loader import TensixLauncher, worker_coord, worker_coords


def _i(s):
    return int(s, 0)


def _proc(s):
    """Accept a processor name (DM0/DM1/MATH0/MATH1/MATH2) or an int id."""
    if s.upper() in abi.PROC:
        return abi.PROC[s.upper()]
    return int(s, 0)


def _ctx():
    from ttexalens import init_ttexalens
    return init_ttexalens()


def _kmap():
    """watcher_kernel_id -> kernel info, from the tt-metal Inspector dump (or {} if none)."""
    try:
        from ..web import inspector
        return inspector.by_watcher_id()
    except Exception:
        return {}


def _launcher(args, ctx):
    return TensixLauncher.at(args.x, args.y, ctx=ctx)


# ---- bootloader helpers (resident-program console) ----------------------------------
def _bl_targets(args, ctx):
    """The coords this bl-command applies to: one (x,y), or every worker with --all (grid broadcast)."""
    if getattr(args, "all", False):
        return worker_coords(ctx)
    if args.x is None or args.y is None:
        raise ValueError("give x y, or --all for the whole grid")
    return [worker_coord(ctx, args.x, args.y)]


def _bl(coord, ctx, risc=0):
    from .bootloader import Bootloader
    return Bootloader(TensixLauncher(coord, ctx=ctx), risc=risc)


def _riscs(args):
    """The RISC indices a bl-command targets: --risc R (one), or all 5 with --all-riscs."""
    from .bootloader import NUM_RISCS, risc_idx
    if getattr(args, "all_riscs", False):
        return list(range(NUM_RISCS))
    return [risc_idx(getattr(args, "risc", 0))]


# ---- commands -----------------------------------------------------------------------
def c_cores(args, ctx):
    cs = worker_coords(ctx)
    print(f"{len(cs)} Tensix worker cores (noc0):")
    line = []
    for c in cs:
        try:
            x, y = c.to("noc0")
            line.append(f"{x},{y}")
        except Exception:
            continue
        if len(line) == 12:
            print("  " + "  ".join(line)); line = []
    if line:
        print("  " + "  ".join(line))


def c_scan(args, ctx):
    """Which cores have a resident program + which kernel runs there ('which core ran what')."""
    kmap = _kmap()
    rows = []
    for c in worker_coords(ctx):
        try:
            x, y = c.to("noc0")
        except Exception:
            continue
        try:
            b = TensixLauncher(c, ctx=ctx).brief(kernels=kmap)
        except Exception as e:
            b = {"resident": False, "error": str(e)}
        if args.all or b.get("resident"):
            rows.append({"x": x, "y": y, **b})
    if args.json:
        print(json.dumps(rows)); return
    res = [r for r in rows if r.get("resident")]
    note = "" if kmap else "  (no Inspector dump — run a tt-metal kernel to see kernel names)"
    print(f"{len(res)} resident core(s)" + ("" if args.all else " — use --all to list every core") + ":" + note)
    for r in sorted(rows, key=lambda r: (r["y"], r["x"])):
        flag = "●" if r.get("resident") else ("x" if r.get("error") else "·")
        if r.get("resident"):
            names = ", ".join(r.get("kernel_names") or []) or f"prog {r.get('host_id')}"
            tag = "" if r.get("user_kernel") else "  [infra]"
            info = f"{names}  (go {r.get('signal')}){tag}"
        else:
            info = "read error" if r.get("error") else "idle"
        print(f"  {flag} {r['x']:>2},{r['y']:<2}  {info}")


def c_snapshot(args, ctx):
    snap = _launcher(args, ctx).snapshot(args.index, kernels=_kmap())
    if args.json:
        print(json.dumps(snap, indent=2)); return
    print(f"core noc0 {args.x},{args.y}  rd_ptr={snap['rd_ptr']}  active=#{snap['active_index']}  "
          f"mode={snap['mode']}  enables={snap['enables']}  prog_id={snap['host_assigned_id']:#x}")
    if snap.get("kernel_names"):
        print(f"  kernels: {', '.join(snap['kernel_names'])}")
    print(f"  go: {snap['go']['signal_name']} (master {snap['go']['master_x']},{snap['go']['master_y']})")
    if not snap["procs"]:
        print("  (no resident program — run a tt-metal kernel on this core first)")
    for p in snap["procs"]:
        k = p.get("kernel")
        kinfo = f"  {k['name']} <{k['source'].split('/')[-1]}> hash {k['hash'][:8]}" if k else ""
        print(f"  {p['proc']:<6} rta @ {p['rta_addr']}   crta @ {p['crta_addr']}{kinfo}")


def c_rta(args, ctx):
    vals = _launcher(args, ctx).read_rta(_proc(args.proc), args.n, index=args.index, common=args.common)
    if args.json:
        print(json.dumps(vals)); return
    kind = "crta" if args.common else "rta"
    print(f"{kind}[{abi.PROC_NAME.get(_proc(args.proc), args.proc)}]: " +
          "  ".join(f"[{i}]={v:#x}({v})" for i, v in enumerate(vals)))


def c_poke(args, ctx):
    addr = _launcher(args, ctx).write_rta(_proc(args.proc), [_i(v) for v in args.values],
                                          index=args.index, arg_offset=args.offset, common=args.common)
    print(f"poked {len(args.values)} word(s) @ {addr:#x} (arg {args.offset}+)")


def c_go(args, ctx):
    r = _launcher(args, ctx).go(_i(args.signal) if args.signal is not None else abi.RUN_MSG_GO)
    print(f"go[{r['go_index']}] @ {r['addr']:#x} -> {r['signal_name']}")


def c_loop(args, ctx):
    """Re-issue go in a loop so the one-shot kernel runs forever. Ctrl-C (or --count) to stop."""
    L = _launcher(args, ctx)
    if not args.force:
        b = L.brief(kernels=_kmap())
        if b.get("resident") and not b.get("user_kernel"):
            print(f"refusing: ({args.x},{args.y}) runs dispatch infra "
                  f"({', '.join(b.get('kernel_names') or [])}). Use --force to loop it anyway.",
                  file=sys.stderr)
            return 2
    addr = _i(args.watch) if args.watch else None
    period, tty, prev, i = 1.0 / args.hz, sys.stdout.isatty(), None, 0
    tgt = f" — watching {args.n}w @ {addr:#x}" if addr is not None else ""
    stop = f"{args.count} runs" if args.count else "infinite (Ctrl-C)"
    print(f"looping go on core {args.x},{args.y} @ {args.hz} Hz, {stop}{tgt}")
    try:
        while True:
            L.go()
            i += 1
            if addr is not None:
                words = L.rd(addr, args.n)
                cells = []
                for j, w in enumerate(words):
                    ch = prev is not None and prev[j] != w
                    s = f"{w:08x}"
                    cells.append(f"\033[33m{s}\033[0m" if (ch and tty) else (f"*{s}" if ch else s))
                print(f"{i:>6} {time.strftime('%H:%M:%S')} " + " ".join(cells)); prev = words
            elif i % max(1, int(args.hz)) == 0:
                print(f"  {i} runs")
            if args.count and i >= args.count:
                print(f"done — {i} runs."); break
            time.sleep(period)
    except KeyboardInterrupt:
        print(f"\nstopped after {i} runs.")
    return 0


def c_peek(args, ctx):
    words = _launcher(args, ctx).rd(_i(args.addr), args.n)
    if args.json:
        print(json.dumps(words)); return
    base = _i(args.addr)
    for i in range(0, len(words), 8):
        row = words[i:i + 8]
        print(f"  {base + i*4:#010x}: " + " ".join(f"{w:08x}" for w in row))


def c_pokeword(args, ctx):
    _launcher(args, ctx).wr(_i(args.addr), [_i(v) for v in args.values])
    print(f"wrote {len(args.values)} word(s) @ {_i(args.addr):#x}")


def c_watch(args, ctx):
    """Live L1 window — poll N words at ADDR and print, marking changed words. Ctrl-C to stop."""
    L = _launcher(args, ctx)
    addr, n, period = _i(args.addr), args.n, 1.0 / args.hz
    tty = sys.stdout.isatty()
    prev = None
    print(f"watching {n} word(s) @ {addr:#x} on core {args.x},{args.y} at {args.hz} Hz (Ctrl-C to stop)")
    try:
        while True:
            words = L.rd(addr, n)
            cells = []
            for i, w in enumerate(words):
                changed = prev is not None and prev[i] != w
                s = f"{w:08x}"
                cells.append(f"\033[33m{s}\033[0m" if (changed and tty) else (f"*{s}" if changed else s))
            print(f"{time.strftime('%H:%M:%S')} " + " ".join(cells))
            prev = words
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped.")


# ---- bootloader commands ------------------------------------------------------------
def c_bl_status(args, ctx):
    rows = []
    for c in _bl_targets(args, ctx):
        try:
            x, y = c.to("noc0")
        except Exception:
            continue
        for r in _riscs(args):
            try:
                s = _bl(c, ctx, risc=r).status()
            except Exception as e:
                s = {"risc": r, "error": str(e)}
            rows.append((x, y, s))
    if args.json:
        print(json.dumps([{"x": x, "y": y, **s} for x, y, s in rows])); return
    for x, y, s in sorted(rows, key=lambda r: (r[1], r[0], r[2].get("risc", 0))):
        name = s.get("risc_name", f"r{s.get('risc', '?')}")
        if "error" in s:
            print(f"  {x:>2},{y:<2} {name:<6}  err {s['error']}"); continue
        print(f"  {x:>2},{y:<2} {name:<6} {s['status_name']:<7} hb={s['heartbeat']:<10} "
              f"door={s['doorbell_name']:<8} last={s['last_cmd_name']:<8} "
              f"ret={s['ovl_ret']:#x}  params={[hex(p) for p in s['params']]}")


def c_bl_param(args, ctx):
    for c in _bl_targets(args, ctx):
        x, y = c.to("noc0")
        for r in _riscs(args):
            b = _bl(c, ctx, risc=r)
            addr = b.set_param(args.index, _i(args.value))
            print(f"  {x},{y} {b.risc_name}: PARAM[{args.index}] = {_i(args.value):#x} @ {addr:#x}")


def c_bl_stage(args, ctx):
    with open(args.bin, "rb") as f:
        data = f.read()
    for c in _bl_targets(args, ctx):
        x, y = c.to("noc0")
        for r in _riscs(args):
            b = _bl(c, ctx, risc=r)
            res = b.stage(data, slot=args.slot)
            print(f"  {x},{y} {b.risc_name}: staged {res['bytes']}B ({res['words']}w) "
                  f"-> slot {res['slot']} @ {res['addr']:#x}")


def c_bl_exec(args, ctx):
    for c in _bl_targets(args, ctx):
        x, y = c.to("noc0")
        for r in _riscs(args):
            b = _bl(c, ctx, risc=r)
            try:
                res = b.exec(slot=args.slot, force=getattr(args, "force", False))
            except ValueError as e:
                print(f"  {x},{y} {b.risc_name}: SKIP — {e}"); continue
            if args.wait:
                ok = b.wait_ack(timeout=args.timeout)
                st = b.status()
                print(f"  {x},{y} {b.risc_name}: exec slot {res['slot']} -> {'ack' if ok else 'NO-ACK'}  "
                      f"status={st['status_name']} ret={st['ovl_ret']:#x}")
            else:
                print(f"  {x},{y} {b.risc_name}: exec slot {res['slot']} @ {res['addr']:#x} (rang doorbell)")


def c_bl_halt(args, ctx):
    for c in _bl_targets(args, ctx):
        x, y = c.to("noc0")
        for r in _riscs(args):
            b = _bl(c, ctx, risc=r)
            b.halt()
            print(f"  {x},{y} {b.risc_name}: HALT")


def c_bl_reset(args, ctx):
    """Soft-reset (recover) a wedged baby RISC over the NoC — no tt-smi -r 0, no metal relaunch."""
    for c in _bl_targets(args, ctx):
        x, y = c.to("noc0")
        for r in _riscs(args):
            b = _bl(c, ctx, risc=r)
            try:
                res = b.soft_reset(revive=not args.halt,
                                   entry=(_i(args.entry) if args.entry else None))
            except Exception as e:
                print(f"  {x},{y} {b.risc_name}: reset FAILED — {e}"); continue
            mode = "held in reset" if args.halt else "revived"
            print(f"  {x},{y} {b.risc_name}: {mode} (was_in_reset={res['was_in_reset']} "
                  f"now_in_reset={res['in_reset']})")
            if not args.halt:
                time.sleep(0.1)
                a = b.alive()
                print(f"     -> heartbeat {'ADVANCING' if a['advancing'] else 'static'} "
                      f"(hb={a['heartbeat']:#x})")


def c_bl_watch(args, ctx):
    """Live heartbeat + telemetry window for one core's resident bootloader. Ctrl-C to stop."""
    from . import bootloader as bl
    b = _bl(worker_coord(ctx, args.x, args.y), ctx, risc=getattr(args, "risc", 0))
    period, tty, prev = 1.0 / args.hz, sys.stdout.isatty(), None
    print(f"bl-watch core {args.x},{args.y} {b.risc_name} @ {args.hz} Hz (Ctrl-C to stop): "
          f"heartbeat | status | telem[0..{args.n - 1}] @ {b.telem:#x}")
    try:
        while True:
            hb = b.L.rd(b.ctrl_addr(bl.HEARTBEAT), 1)[0]
            st = bl.STATUS_NAME.get(b.L.rd(b.ctrl_addr(bl.STATUS), 1)[0], "?")
            tel = b.L.rd(b.telem, args.n)
            cells = []
            for i, w in enumerate(tel):
                ch = prev is not None and prev[i] != w
                s = f"{w:08x}"
                cells.append(f"\033[33m{s}\033[0m" if (ch and tty) else (f"*{s}" if ch else s))
            print(f"{time.strftime('%H:%M:%S')} hb={hb:<10} {st:<7} " + " ".join(cells))
            prev = tel
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bhtop-tensix", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def xy(sp):
        sp.add_argument("x", type=int); sp.add_argument("y", type=int)

    sub.add_parser("cores", help="list Tensix worker cores (noc0)").set_defaults(func=c_cores)

    sp = sub.add_parser("scan", help="which cores have a resident program")
    sp.add_argument("--all", action="store_true", help="list every core, not just resident ones")
    sp.add_argument("--json", action="store_true"); sp.set_defaults(func=c_scan)

    sp = sub.add_parser("snapshot", help="decode the live launch mailbox"); xy(sp)
    sp.add_argument("--index", type=int, default=None); sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=c_snapshot)

    sp = sub.add_parser("rta", help="read a processor's runtime args"); xy(sp)
    sp.add_argument("proc"); sp.add_argument("--n", type=int, default=8)
    sp.add_argument("--index", type=int, default=None); sp.add_argument("--common", action="store_true")
    sp.add_argument("--json", action="store_true"); sp.set_defaults(func=c_rta)

    sp = sub.add_parser("poke", help="write runtime-arg words (no recompile)"); xy(sp)
    sp.add_argument("proc"); sp.add_argument("values", nargs="+")
    sp.add_argument("--offset", type=int, default=0); sp.add_argument("--index", type=int, default=None)
    sp.add_argument("--common", action="store_true"); sp.set_defaults(func=c_poke)

    sp = sub.add_parser("go", help="re-issue the go signal"); xy(sp)
    sp.add_argument("--signal", default=None, help="go_msg signal byte (default GO=0x80)")
    sp.set_defaults(func=c_go)

    sp = sub.add_parser("loop", help="re-issue go forever so the kernel runs continuously"); xy(sp)
    sp.add_argument("--hz", type=float, default=10.0); sp.add_argument("--count", type=int, default=0,
                    help="stop after N runs (0 = infinite)")
    sp.add_argument("--watch", default=None, help="L1 addr to read+print each run")
    sp.add_argument("--n", type=int, default=8, help="words for --watch")
    sp.add_argument("--force", action="store_true", help="allow looping a dispatch-infra core")
    sp.set_defaults(func=c_loop)

    sp = sub.add_parser("peek", help="raw L1 read"); xy(sp)
    sp.add_argument("addr"); sp.add_argument("n", type=int, nargs="?", default=8)
    sp.add_argument("--json", action="store_true"); sp.set_defaults(func=c_peek)

    sp = sub.add_parser("pokeword", help="raw L1 write"); xy(sp)
    sp.add_argument("addr"); sp.add_argument("values", nargs="+"); sp.set_defaults(func=c_pokeword)

    sp = sub.add_parser("watch", help="live L1 window (watch your kernel run)"); xy(sp)
    sp.add_argument("addr"); sp.add_argument("n", type=int, nargs="?", default=8)
    sp.add_argument("--hz", type=float, default=4.0); sp.set_defaults(func=c_watch)

    # ---- resident bootloader console ----
    def risc_opt(sp):
        """Which of the 5 baby RISCs to drive: --risc 0..4 or a name (BRISC/NCRISC/TRISC0..2)."""
        sp.add_argument("-r", "--risc", default=0,
                        help="RISC index 0..4 or name (BRISC,NCRISC,TRISC0,TRISC1,TRISC2); default BRISC")
        sp.add_argument("--all-riscs", action="store_true", help="apply to all 5 RISCs of the core")

    def xy_opt(sp):
        """x y optional so `--all` can target the whole grid; plus per-RISC selection."""
        sp.add_argument("x", type=int, nargs="?"); sp.add_argument("y", type=int, nargs="?")
        sp.add_argument("--all", action="store_true", help="apply to every worker (grid broadcast)")
        risc_opt(sp)

    sp = sub.add_parser("bl-status", help="decode the resident bootloader's control mailbox")
    xy_opt(sp); sp.add_argument("--json", action="store_true"); sp.set_defaults(func=c_bl_status)

    sp = sub.add_parser("bl-param", help="poke a live PARAM word (no compile, no re-go)"); xy_opt(sp)
    sp.add_argument("index", type=int); sp.add_argument("value"); sp.set_defaults(func=c_bl_param)

    sp = sub.add_parser("bl-stage", help="NoC-write a compiled overlay .bin into a code slot"); xy_opt(sp)
    sp.add_argument("bin", help="raw overlay binary (objcopy -O binary)")
    sp.add_argument("-s", "--slot", default="A", choices=["A", "B", "a", "b"]); sp.set_defaults(func=c_bl_stage)

    sp = sub.add_parser("bl-exec", help="invalidate i$ + call a staged code slot"); xy_opt(sp)
    sp.add_argument("-s", "--slot", default="A", choices=["A", "B", "a", "b"])
    sp.add_argument("--wait", action="store_true", help="wait for the kernel to ack + report return")
    sp.add_argument("--timeout", type=float, default=2.0)
    sp.add_argument("--force", action="store_true",
                    help="allow EXEC on NCRISC/TRISC (they can't fetch from L1 -> will wedge the core)")
    sp.set_defaults(func=c_bl_exec)

    sp = sub.add_parser("bl-halt", help="break the resident loop (kernel returns)"); xy_opt(sp)
    sp.set_defaults(func=c_bl_halt)

    sp = sub.add_parser("bl-reset",
                        help="soft-reset/recover a wedged baby RISC over NoC (no tt-smi, no relaunch)")
    xy_opt(sp)
    sp.add_argument("--halt", action="store_true",
                    help="assert reset and LEAVE held (clean halt) instead of reviving")
    sp.add_argument("--entry", default=None,
                    help="reset-PC for revive (NCRISC/TRISC only); default = restart from existing vector")
    sp.set_defaults(func=c_bl_reset)

    sp = sub.add_parser("bl-watch", help="live heartbeat + telemetry for one core"); xy(sp)
    sp.add_argument("n", type=int, nargs="?", default=4, help="telemetry words to show")
    sp.add_argument("--hz", type=float, default=4.0); risc_opt(sp); sp.set_defaults(func=c_bl_watch)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help(); return 1
    try:
        return args.func(args, _ctx()) or 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
