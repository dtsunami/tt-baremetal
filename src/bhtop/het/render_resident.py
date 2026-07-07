"""het.render_resident — drive the FUSED RESIDENT Gaussian-splat forward render across the Tensix grid.

The `resident_render_perf` kernel runs the whole `tensix.splat.render_ondevice` pipeline (6 MVMUL + 5
SFPU, 11 stages fused to 6) resident on one worker: host stages the per-tile operands once and rings the
doorbell once per 32-pixel group — no per-op ELF reload/reboot. This module is the host driver: build
operands, boot the kernel on a worker (or every worker), render a tile, assemble the RGB. Proven on
silicon (RESIDENT_GRID.md): full 16x16 tile = 51 dB vs the exact golden; N-worker grid renders in parallel.

The heavy lifting (the proven residency + inter-stage L1 dataflow) is in the kernel; here we only shuttle
operands and rings over exalens. Bit layout / addresses MUST match resident_render_perf.cpp.
"""
import math
import time

from ..tensix import matmul as MM, llk_run, splat as SP
from ..tensix.resident import boot_resident

# L1 map — must match kernels/tensix/llk/resident_render_perf/resident_render_perf.cpp
DB, DONE, HB = 0x16000, 0x16010, 0x16020
TELEM = 0x16080                                          # pack cycle stamps: [start, F1..F6 (grp0), end]
NG_ADDR = 0x160A0                                        # host-set pixel-group count (1 ring = whole tile)
STRIDE = 0x800                                           # per-group stride for phi[g] / OUT[g] (bf16 32x32)
STAGE_NAMES = ("F1 Vsq=sq(phi@psi)", "F2 ar=exp(@Ppair)", "F3 lpa=log(@Dop)",
               "F4 la=log1p(@Dnop)", "F5 w=exp(la@Stri+lpa)", "F6 C=@color")
A_PHI, A_PSI = 0x21000, 0x31000
A_PPAIR, A_DOP, A_DNOP, A_STRI, A_IDEN, A_COLOR = 0x60000, 0x60800, 0x61000, 0x61800, 0x62000, 0x62800
S_VSQ, S_AR, S_LPA, S_LA, S_W, OUT_C = 0x40000, 0x40800, 0x41000, 0x41800, 0x42000, 0x51000
POISON = 0xBADF00D5
SCRATCH = (S_VSQ, S_AR, S_LPA, S_LA, S_W, OUT_C)
RUNTIME_WORDS = [1, 1, 1, 1, 1, 128, 128, 0, 4, 4]      # single 32x32 bf16 tile, stock matmul ABI
KERNEL = "resident_render_perf"
# fp32 dest-acc + bf16 pack: unpack/math bf16 (5), dest fp32 (pack_src=0), pack_dst bf16 (5).
_FP32_FORMATS = (5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 0, 5)


def build(fp32_acc=False):
    """Compile the fused render kernel (ITERATIONS=32 so the SFPU covers the full 32x32 tile).
    fp32_acc=True: fp32 dest accumulation + fp32 SFPU (higher precision) with a bf16 pack of the result."""
    b = llk_run.build(KERNEL, run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=fp32_acc,
                      formats=(_FP32_FORMATS if fp32_acc else None),
                      overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    if not b["ok"]:
        raise RuntimeError("resident_render_perf build failed:\n" + b["log"][-2000:])
    return b


def _enc(flat):
    return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])


def build_operands(*, k=16, size=16, seed=5, gs=None, order=None):
    """The per-tile operands (depth-sorted): the const matrices (staged once) + per-group phi + golden.
    Mirrors tensix.splat.render_ondevice's operand construction. Returns (consts, phis, groups, gold, gs, order)."""
    assert 2 * k <= 32 and size <= 32
    if gs is None:
        gs = SP.scene_rgb(k=k, seed=seed, span=float(size))
    if order is None:
        order = sorted(range(k), key=lambda i: gs[i][9])
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, k)
    Stri = [Mcomb[r] for r in range(k)]                 # strict-upper K x K
    Iden = [Mcomb[k + r] for r in range(k)]             # identity  K x K
    psi_rows = [[psi[r][c] for c in range(2 * k)] for r in range(3)]
    consts = {A_PSI: SP._pad32(psi_rows), A_PPAIR: SP._pad32(Ppair), A_DOP: SP._pad32(Dop),
              A_DNOP: SP._pad32(Dnop), A_STRI: SP._pad32(Stri), A_IDEN: SP._pad32(Iden),
              A_COLOR: SP._pad32(color)}
    pixels = [(x, y) for y in range(size) for x in range(size)]
    groups = [pixels[i:i + 32] for i in range(0, len(pixels), 32)]
    phis = [SP._pad32([[float(x), float(y), 1.0] for (x, y) in g]) for g in groups]
    gold = SP._golden_render(gs, size)
    return consts, phis, groups, gold, gs, order


def boot(coord, *, ctx, device_id=0, phis=None, ng=None, consts=None):
    """Boot the resident render kernel on `coord`. phi[g] (pixel coords) are the SAME for every tile of a
    given size, so stage them once here (strided A_PHI + g*STRIDE) + write the group count NG. `consts`
    (the per-tile gaussian matrices) may be staged here too or later via stage_consts()."""
    from ttexalens.tt_exalens_lib import write_words_to_device as wr
    rdbg = boot_resident(KERNEL, coord, ctx=ctx, device_id=device_id, runtime_words=RUNTIME_WORDS, clear_words=48)
    if phis is not None:
        for g, flat in enumerate(phis):
            wr(coord, A_PHI + g * STRIDE, _enc(flat), device_id=device_id, context=ctx)
        wr(coord, NG_ADDR, [len(phis) if ng is None else ng], device_id=device_id, context=ctx)
    if consts is not None:
        stage_consts(coord, ctx=ctx, device_id=device_id, consts=consts)
    return rdbg


def stage_consts(coord, *, ctx, device_id=0, consts):
    """Stage the per-tile gaussian const matrices (psi/Ppair/Dop/Dnop/Stri/Iden/color) into L1."""
    from ttexalens.tt_exalens_lib import write_words_to_device as wr
    for addr, flat in consts.items():
        wr(coord, addr, _enc(flat), device_id=device_id, context=ctx)


def render_tile(coord, *, ctx, device_id=0, groups, size=16, timeout=6.0, telem=False):
    """Render a whole tile in ONE doorbell ring (the kernel loops all NG pixel-groups internally). The
    per-tile consts must already be staged (boot/stage_consts). Returns RGB [size*size][3]; if telem,
    also a telemetry dict (per-stage device cycles for group 0 + whole-tile device cycles + host ms)."""
    from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
    rgb = [[0.0, 0.0, 0.0] for _ in range(size * size)]
    ring = rd(coord, DONE, device_id=device_id, context=ctx) + 1
    wr(coord, DB, [ring], device_id=device_id, context=ctx)
    t0 = time.time()
    while time.time() - t0 < timeout and rd(coord, DONE, device_id=device_id, context=ctx) != ring:
        time.sleep(0.004)
    host_ms = (time.time() - t0) * 1e3
    p = 0
    for gi, g in enumerate(groups):
        c = MM.untilize32(MM.unpack_bf16_words(
            rds(coord, OUT_C + gi * STRIDE, word_count=512, device_id=device_id, context=ctx)))
        for r in range(len(g)):
            rgb[p] = [c[r * 32], c[r * 32 + 1], c[r * 32 + 2]]; p += 1
    if telem:
        ts = rds(coord, TELEM, word_count=8, device_id=device_id, context=ctx)
        t = {"stage_cycles": [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(6)],
             "group0_cycles": (ts[6] - ts[0]) & 0xFFFFFFFF,
             "tile_cycles": (ts[7] - ts[0]) & 0xFFFFFFFF,
             "host_ms": host_ms, "groups": len(groups)}
        return rgb, t
    return rgb


def psnr(rgb, gold):
    n = len(gold)
    mse = sum((rgb[p][ch] - gold[p][ch]) ** 2 for p in range(n) for ch in range(3)) / (n * 3)
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)


def render(coord, *, ctx, device_id=0, k=16, size=16, seed=5, gs=None, order=None,
           do_build=True, fp32_acc=False, telem=True, verbose=True):
    """Convenience: build operands + boot + stage + render one tile, return a render_ondevice-shaped dict
    (+ telemetry: per-stage device cycles and host ms/ring)."""
    if do_build:
        build(fp32_acc=fp32_acc)
    consts, phis, groups, gold, gs, order = build_operands(k=k, size=size, seed=seed, gs=gs, order=order)
    boot(coord, ctx=ctx, device_id=device_id, phis=phis, ng=len(groups), consts=consts)
    time.sleep(0.05)
    out = render_tile(coord, ctx=ctx, device_id=device_id, groups=groups, size=size, telem=telem)
    rgb, t = out if telem else (out, None)
    p = psnr(rgb, gold)
    res = {"ok": p >= 40.0, "psnr_db": p, "rgb": rgb, "golden": gold, "size": size, "gaussians": k,
           "groups": len(groups), "coord": str(coord), "order": list(order), "gs": gs, "telem": t}
    if verbose:
        print(f"[render_resident] {size}x{size}, {k} Gaussians, {len(groups)} groups — FUSED resident "
              f"({'fp32' if fp32_acc else 'bf16'} dest, 1 boot, 1 RING/tile, NO reload) "
              f"PSNR={p:.1f} dB -> {'PASS' if res['ok'] else 'CHECK'}")
        if t:
            print(f"[render_resident] whole tile = {t['tile_cycles']} device cyc "
                  f"({t['groups']} groups, group0={t['group0_cycles']} cyc); host {t['host_ms']:.1f} ms/ring")
            for nm, c in zip(STAGE_NAMES, t["stage_cycles"]):
                print(f"    {nm:24s} {c:6d} cyc")
    return res
