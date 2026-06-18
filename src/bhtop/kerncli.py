"""
bhtop-kern — batch param importer.

Parse every kernel's source and MERGE the discovered params into its kernel.json — the same
per-kernel "Merge from source" the web labs expose (ParamPanel), run across the whole tree.
The merge is idempotent and edit-preserving: a param already in a sidecar is left exactly as you
tuned it; only genuinely new discoveries are appended (see web.kernparse).

  bhtop-kern merge                  # x280 + noc + tensix (the working / overlay trees)
  bhtop-kern merge --engine x280    # one engine only (x280 | noc | tensix | all)
  bhtop-kern merge --canonical      # x280: ALSO populate the tracked, shipped sidecars
  bhtop-kern merge --dry-run        # report what would change; write nothing

Runs filesystem-only (no device). Also invocable as `python -m bhtop.kerncli`.
"""
import argparse
import sys


def _report(title, res):
    """Print a per-kernel breakdown for one engine's merge_all() result; return params added."""
    if not res.get("available", True):
        print(f"  {title}: unavailable (root: {res.get('root')})")
        return 0
    total = 0
    for r in res.get("results", []):
        total += r["count"]
        flag = "+" if r["count"] else "·"
        names = ("  " + ", ".join(r["added"])) if r["added"] else ""
        print(f"  {flag} {r['kernel']:<30} {r['count']:>2}{names}")
    print(f"  → {title}: {total} param(s) across {len(res.get('results', []))} kernel(s)\n")
    return total


def cmd_merge(args):
    engines = ["x280", "noc", "tensix"] if args.engine == "all" else [args.engine]
    grand = 0
    for e in engines:
        if e == "x280":
            from .web import l2lab
            print("== X280 — bare-metal (working tree) ==")
            grand += _report("x280", l2lab.merge_all(dry_run=args.dry_run))
            if args.canonical:
                print("== X280 — canonical (tracked, shipped) ==")
                grand += _report("x280-canonical",
                                 l2lab.merge_all(root=l2lab.KERN_CANON, dry_run=args.dry_run))
        elif e == "noc":
            from .web import lab
            print("== NOC — tt-metal data_movement ==")
            grand += _report("noc", lab.merge_all(dry_run=args.dry_run))
        elif e == "tensix":
            from .web import tlab
            print("== TENSIX — tt-metal programming_examples ==")
            grand += _report("tensix", tlab.merge_all(dry_run=args.dry_run))
    print(f"{'(dry-run) ' if args.dry_run else ''}TOTAL params merged: {grand}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="bhtop-kern", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    m = sub.add_parser("merge", help="parse kernel sources + merge discovered params into kernel.json")
    m.add_argument("--engine", choices=["x280", "noc", "tensix", "all"], default="all")
    m.add_argument("--canonical", action="store_true",
                   help="x280: also update the tracked shipped sidecars (src/bhtop/kernels/x280)")
    m.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    m.set_defaults(func=cmd_merge)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
