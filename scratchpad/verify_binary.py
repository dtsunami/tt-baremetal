"""Verify the new FPU eltwise-binary helper (add/sub/mul) on bare-metal Tensix vs host reference."""
import sys, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import sfpu as SF

ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
rnd = random.Random(3)
a = [rnd.uniform(-4, 4) for _ in range(1024)]
b = [rnd.uniform(-4, 4) for _ in range(1024)]
for op, f in [("add", lambda x, y: x + y), ("sub", lambda x, y: x - y), ("mul", lambda x, y: x * y)]:
    out, ok = SF.run_binary(coord, a, b, ctx=ctx, op=op, prebuilt=False)
    ref = [f(a[i], b[i]) for i in range(1024)]
    rel = max(abs(out[i] - ref[i]) for i in range(1024)) / (max(abs(r) for r in ref) + 1e-9)
    print(f"  {op:3s}: ok={ok}  max_rel_err={rel:.2e}  {'PASS' if rel < 2e-2 else 'FAIL'}")
