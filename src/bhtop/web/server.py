"""
bhtop-web — FastAPI server exposing live Blackhole NoC telemetry over REST + WebSocket.

All device access funnels through a single DeviceManager (one worker thread), so the
chip never sees concurrent PCIe access. The Svelte frontend (built to frontend/dist)
is served as static files in production; in dev it runs on the Vite server and proxies
/api + /ws here.
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .device import DeviceManager, CARD_PATH
from .schemas import InjectRequest, KernelRunRequest
from ..patterns import PATTERN_INFO

app = FastAPI(title="bhtop-web")
dm = DeviceManager()

DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


@app.on_event("startup")
async def _startup():
    await dm.start()


@app.get("/api/floorplan")
async def floorplan():
    return dm.floorplan_model()


@app.get("/api/status")
async def status():
    return dm.status()


@app.get("/api/card.png")
async def card_png():
    if not os.path.exists(CARD_PATH):
        raise HTTPException(404, f"card image not found at {CARD_PATH}")
    return FileResponse(CARD_PATH, media_type="image/png")


@app.get("/api/tile/{x}/{y}")
async def tile(x: int, y: int):
    d = await dm.tile_detail(x, y)
    if d is None:
        raise HTTPException(404, f"no tile at noc0 ({x},{y})")
    return d


@app.get("/api/inject/patterns")
async def inject_patterns():
    return PATTERN_INFO


@app.post("/api/inject")
async def inject(req: InjectRequest):
    if dm.reset_needed:
        raise HTTPException(409, "NoC hang pending — run `tt-smi -r 0` and restart the server")
    if dm.mode == "busy":
        raise HTTPException(409, "a tt-metal kernel owns the device — wait for it to finish")
    return await dm.inject(req.src, req.pattern, req.length, req.fires, req.stream)


@app.post("/api/inject/stop")
async def inject_stop():
    return await dm.inject_stop()


@app.get("/api/kernels")
async def kernels():
    return await dm.kernels()


@app.post("/api/kernels/run")
async def kernels_run(req: KernelRunRequest):
    if dm.reset_needed:
        raise HTTPException(409, "NoC hang pending — run `tt-smi -r 0` and restart the server")
    return await dm.run_kernel(req.name, req.timeout)


@app.get("/api/kernels/last")
async def kernels_last():
    return dm.kernel_last()


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    q = await dm.subscribe()
    try:
        while True:
            frame = await q.get()
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        pass
    finally:
        dm.unsubscribe(q)


# Serve the built frontend last so /api and /ws take precedence (only if built).
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="bhtop web telemetry server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not DIST.exists():
        print(f"[bhtop-web] frontend not built ({DIST}); API + WS only. "
              f"Run `cd frontend && npm install && npm run build`, or use the Vite dev server.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
