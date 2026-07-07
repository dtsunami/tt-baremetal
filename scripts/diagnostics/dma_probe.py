"""Probe whether tt-umd DMA actually WORKS on this Blackhole (ttexalens gates it off for BH in can_use_dma,
but the underlying dma_read/write_from_device may work — 4x1GB hugepages are present). Patch the gate on,
bulk-transfer to worker L1 + x280 GDDR, VERIFY correctness, and time DMA-on vs DMA-off (noc)."""
import sys, time, numpy as np
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens, umd_device
from ttexalens.tt_exalens_lib import write_to_device as wrb, read_from_device as rdb
from bhtop.tensix.loader import worker_coord

ctx = init_ttexalens()
# introspect the umd device
udev = None
try:
    udev = ctx.devices[0]._device if hasattr(ctx.devices[0], "_device") else None
except Exception:
    pass
print("can_use_dma (default):", getattr(umd_device.UmdDevice, "can_use_dma", None))

N = 256 * 1024                                   # 256 KiB
payload = bytes((i * 131 + 7) & 0xFF for i in range(N))
WX, WY = 11, 2
coord = worker_coord(ctx, WX, WY)
L1 = 0x20000                                     # valid L1 offset


def bench(tag, addr, use_dma):
    # toggle the BH DMA gate
    if use_dma:
        umd_device.UmdDevice.can_use_dma = property(lambda self: self._is_mmio_capable and not self._is_simulation)
    else:
        umd_device.UmdDevice.can_use_dma = property(lambda self: False)
    try:
        t = time.time(); wrb(coord, addr, payload, context=ctx); wt = time.time() - t
        t = time.time(); got = rdb(coord, addr, num_bytes=N, context=ctx); rt = time.time() - t
        ok = bytes(got) == payload
        print(f"  [{tag:14s}] write {N/wt/1e6:7.1f} MB/s  read {N/rt/1e6:7.1f} MB/s  correct={ok}")
    except Exception as e:
        print(f"  [{tag:14s}] ERROR: {type(e).__name__}: {str(e)[:120]}")


print(f"=== worker L1 ({WX},{WY}) @0x{L1:x}, {N//1024}KiB ===")
bench("noc (default)", L1, False)
bench("DMA (patched)", L1, True)
