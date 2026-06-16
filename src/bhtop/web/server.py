"""
bhtop-web — FastAPI server exposing live Blackhole NoC telemetry over REST + WebSocket.

All device access funnels through a single DeviceManager (one worker thread), so the
chip never sees concurrent PCIe access. The Svelte frontend (built to frontend/dist)
is served as static files in production; in dev it runs on the Vite server and proxies
/api + /ws here.
"""
import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .device import DeviceManager, CARD_PATH
from .schemas import (InjectRequest, KernelRunRequest, LabWriteRequest,
                      LabPathRequest, LabBuildRequest, L2DeployRequest,
                      L2CompileRequest, L2TileRequest, L2WriteRequest,
                      L2NewRequest, L2PokeRequest, TlabRunRequest, CopyRequest)
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
    return await dm.run_kernel(req.name, req.timeout,
                               req.dprint_cores if req.dprint else None)


@app.get("/api/kernels/last")
async def kernels_last():
    return dm.kernel_last()


# ---- tlab: Tensix Compute Lab (run a compute example, per-engine occupancy) ----
@app.get("/api/tlab/examples")
async def tlab_examples():
    return await dm.tlab_examples()


@app.post("/api/tlab/run")
async def tlab_run(req: TlabRunRequest):
    if dm.reset_needed:
        raise HTTPException(409, "NoC hang pending — run `tt-smi -r 0` and restart the server")
    return await dm.tlab_run(req.name, req.timeout)


@app.get("/api/tlab/last")
async def tlab_last():
    return dm.tlab_last()


@app.get("/api/tlab/status")
async def tlab_status():
    return dm.tlab_status()


@app.get("/api/tlab/disasm")
async def tlab_disasm():
    return await dm.tlab_disasm()


@app.get("/api/running")
async def running():
    """tt-metal kernels live now, keyed by JIT build hash (source<->hash<->program<->coords).
    Drives the device tree's running/stale badges."""
    return await dm.running()


@app.post("/api/tlab/file/duplicate")
async def tlab_file_duplicate(req: CopyRequest):
    try:
        return await dm.tlab_copy(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/tlab/files")
async def tlab_files(example: str):
    return await dm.tlab_files(example)


@app.get("/api/tlab/file")
async def tlab_file(path: str):
    try:
        return await dm.tlab_read(path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/file")
async def tlab_file_write(req: LabWriteRequest):
    try:
        return await dm.tlab_write(req.path, req.content)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/file/revert")
async def tlab_file_revert(req: LabPathRequest):
    try:
        return await dm.tlab_revert(req.path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tlab/docs")
async def tlab_docs():
    return await dm.tlab_docs_index()


@app.get("/api/tlab/doc/{doc_id}")
async def tlab_doc(doc_id: str):
    try:
        return await dm.tlab_doc(doc_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ---- kernel lab: edit / build / docs -------------------------------------
@app.get("/api/lab/projects")
async def lab_projects():
    return await dm.lab_projects()


@app.get("/api/lab/files")
async def lab_files(project: str):
    return await dm.lab_files(project)


@app.get("/api/lab/file")
async def lab_file(path: str):
    try:
        return await dm.lab_read(path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/file")
async def lab_file_write(req: LabWriteRequest):
    try:
        return await dm.lab_write(req.path, req.content)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/file/revert")
async def lab_file_revert(req: LabPathRequest):
    try:
        return await dm.lab_revert(req.path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/file/duplicate")
async def lab_file_duplicate(req: CopyRequest):
    try:
        return await dm.lab_copy(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/build")
async def lab_build(req: LabBuildRequest):
    if dm.mode == "busy":
        raise HTTPException(409, "a tt-metal kernel owns the device — wait for it to finish")
    return await dm.lab_build(req.target)


@app.get("/api/lab/build/last")
async def lab_build_last():
    return dm.build_last()


@app.get("/api/lab/docs")
async def lab_docs():
    return await dm.lab_docs_index()


@app.get("/api/lab/doc/{doc_id}")
async def lab_doc(doc_id: str):
    try:
        return await dm.lab_doc(doc_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/lab/uarch/{name}")
async def lab_uarch(name: str):
    from . import lab
    p = lab.uarch_path(name)
    if not p:
        raise HTTPException(404, f"no uarch image {name}")
    return FileResponse(p)


# ---- tt-isa-documentation browser (live from GitHub, cached) --------------
@app.get("/api/isa/tree")
async def isa_tree():
    from . import isa
    try:
        return await asyncio.to_thread(isa.tree)
    except Exception as e:
        raise HTTPException(502, f"ISA docs fetch failed: {e}")


@app.get("/api/isa/doc")
async def isa_doc(path: str):
    from . import isa
    try:
        return await asyncio.to_thread(isa.doc, path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"ISA doc fetch failed: {e}")


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


# ---- L2CPU cockpit: develop / deploy / observe ----------------------------
@app.get("/api/l2/tiles")
async def l2_tiles():
    return await dm.l2_tiles()


@app.get("/api/l2/regs")
async def l2_regs(tile: int):
    return await dm.l2_regs(tile)


@app.get("/api/l2/arch")
async def l2_arch(tile: int, hart: int):
    return await dm.l2_arch(tile, hart)


@app.get("/api/l2/files")
async def l2_files():
    return await dm.l2_files()


@app.get("/api/l2/examples")
async def l2_examples():
    return await dm.l2_examples()


@app.get("/api/l2/file")
async def l2_file(name: str):
    try:
        return await dm.l2_read(name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/file")
async def l2_file_write(req: L2WriteRequest):
    try:
        return await dm.l2_write(req.name, req.content)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/file/new")
async def l2_file_new(req: L2NewRequest):
    try:
        return await dm.l2_new(req.name, req.lang)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/file/duplicate")
async def l2_file_duplicate(req: CopyRequest):
    try:
        return await dm.l2_copy(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/file/delete")
async def l2_file_delete(req: L2WriteRequest):
    try:
        return await dm.l2_delete(req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/file/rename")
async def l2_file_rename(req: CopyRequest):
    try:
        return await dm.l2_rename(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/compile")
async def l2_compile(req: L2CompileRequest):
    return await dm.l2_compile(req.content, req.lang, req.addr)


@app.post("/api/l2/bringup")
async def l2_bringup(req: L2TileRequest):
    return await dm.l2_bringup(req.tile)


@app.get("/api/l2/bringup/last")
async def l2_bringup_last():
    return dm.l2_bringup_last()


@app.post("/api/l2/deploy")
async def l2_deploy(req: L2DeployRequest):
    return await dm.l2_deploy(req.tile, req.hart, req.content, req.lang, req.addr, req.name)


@app.post("/api/l2/deploy_all")
async def l2_deploy_all(req: L2DeployRequest):
    return await dm.l2_deploy_all(req.tile, req.content, req.lang, req.addr, req.name)


@app.post("/api/l2/park_all")
async def l2_park_all(req: L2TileRequest):
    return await dm.l2_park_all(req.tile)


@app.post("/api/l2/tele/zero")
async def l2_tele_zero(req: L2TileRequest):
    return await dm.l2_zero_tele(req.tile, req.hart)


@app.post("/api/l2/poke")
async def l2_poke(req: L2PokeRequest):
    return await dm.l2_poke(req.tile, req.addr, req.val)


@app.get("/api/l2/docs")
async def l2_docs():
    return await dm.l2_docs_index()


@app.get("/api/l2/doc/{doc_id}")
async def l2_doc(doc_id: str):
    try:
        return await dm.l2_doc(doc_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.websocket("/ws/l2cpu")
async def ws_l2cpu(websocket: WebSocket):
    await websocket.accept()
    q = await dm.l2_subscribe()

    async def sender():                      # push telemetry frames to the client
        while True:
            await websocket.send_json(await q.get())

    async def receiver():                    # client picks the focused tile + rate
        while True:
            msg = await websocket.receive_json()
            dm.l2_select(msg.get("tile"), msg.get("hz"))

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        # whichever finishes first (a disconnect surfaces on the receiver) ends both
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for tk in pending:
            tk.cancel()
        for tk in done:
            tk.exception()                   # retrieve so it's not flagged unhandled
    finally:
        dm.l2_unsubscribe(q)


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
