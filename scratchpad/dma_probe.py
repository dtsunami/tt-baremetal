"""Is the gt-upload slowness the exalens transfer (DMA not active) or host-side Python marshaling? Measure both
write paths + a read, on a safe reset GDDR scratch addr (no kernel, no hang risk). 20 MB = a 1600px gt image."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import (write_words_to_device, write_to_device,
                                      read_words_from_device, read_word_from_device)
from bhtop.l2cpu import L2cpu

ctx = init_ttexalens()
ctx.use_4B_mode = False                     # what grid_engine sets (TT_DMA_READBACK)
print("use_4B_mode:", ctx.use_4B_mode, "| dma_write_threshold:", getattr(ctx, "dma_write_threshold", "?"),
      "| dma_read_threshold:", getattr(ctx, "dma_read_threshold", "?"))
dev = L2cpu(ctx=ctx); loc = dev.loc[0]
d0 = dev.dev.devices[0] if hasattr(dev.dev, "devices") else None
print("L2CPU loc:", loc, "| can_use_dma:", getattr(getattr(dev.dev, "_umd_device", dev.dev), "can_use_dma", "?"))

ADDR = 0x30100000                            # PARAM scratch (nothing running post-reset)
N = 1600 * 1056 * 3                          # 20 MB, a 1600px gt
words = [int(x) for x in np.random.randint(0, 2**31, N, dtype=np.uint32)]
buf   = np.asarray(words, np.uint32).tobytes()
MB = len(buf) / 1e6
print(f"\npayload: {N:,} words = {MB:.1f} MB\n")

# sanity: card responds?
try:
    _ = read_word_from_device(loc, ADDR, context=ctx, safe_mode=False); print("card responds to a read: OK\n")
except Exception as e:
    print("card NOT responding:", e); sys.exit(1)

def timed(name, fn):
    t = time.time()
    try: fn(); dt = time.time() - t; print(f"{name:44s} {dt*1000:8.0f} ms   {MB/dt:7.1f} MB/s")
    except Exception as e: print(f"{name:44s} FAILED: {type(e).__name__}: {str(e)[:80]}")
    return

# 1) current grid_engine path: write_words_to_device(int list) -> b"".join marshaling -> noc_write
timed("write_words_to_device(int list)  [current]", lambda: write_words_to_device(loc, ADDR, words, context=ctx, noc_id=0, safe_mode=False))
# 2) bytes path: write_to_device(bytes) -> noc_write directly (skips the join; DMAs if active)
timed("write_to_device(bytes)           [proposed]", lambda: write_to_device(loc, ADDR, buf, context=ctx, noc_id=0, safe_mode=False))
# 3) read for reference (DMA readback is known-working)
timed("read_words_from_device (readback)", lambda: read_words_from_device(loc, ADDR, word_count=N, context=ctx, safe_mode=False))

# verify byte path landed correctly
rb = read_words_from_device(loc, ADDR, word_count=8, context=ctx, safe_mode=False)
print("\nbyte-path readback matches:", list(rb) == words[:8])
