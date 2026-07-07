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
                      L2NewRequest, L2PokeRequest, L2CommandRequest, L2FreqRequest,
                      L2FolderRequest, L2ParamsRequest, KernelConfigRequest,
                      KernelMergeRequest, TlabRunRequest, TlabExampleRequest, CopyRequest,
                      TensixRtaRequest, TensixGoRequest, TensixLoopRequest,
                      TensixBlParamRequest, TensixBlStageRequest, TensixBlExecRequest,
                      TensixBlHaltRequest, TensixBlCompileRequest, TensixBlSourceRequest,
                      TensixBlLaunchRequest, LlkBuildRequest, LlkRunRequest, LlkTestAllRequest)
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


# ---- Tensix launch cockpit: read/poke a worker's runtime args in L1 + re-go ----
@app.get("/api/tensix/launch")
async def tensix_launch(x: int, y: int, index: int | None = None):
    try:
        return await dm.tensix_launch(x, y, index)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tensix/rta")
async def tensix_rta(req: TensixRtaRequest):
    try:
        return await dm.tensix_write_rta(req.x, req.y, req.proc, req.values, req.arg_offset, req.index)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tensix/go")
async def tensix_go(req: TensixGoRequest):
    try:
        return await dm.tensix_go(req.x, req.y, req.signal)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tensix/peek")
async def tensix_peek(x: int, y: int, addr: int, n: int = 8):
    try:
        return await dm.tensix_peek(x, y, addr, n)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tensix/scan")
async def tensix_scan():
    return await dm.tensix_scan()


@app.post("/api/tensix/loop")
async def tensix_loop(req: TensixLoopRequest):
    try:
        return await dm.tensix_loop(req.x, req.y, req.on, req.hz, req.force)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tensix/loop")
async def tensix_loop_status():
    return dm.tensix_loop_status()


# ---- resident bootloader cockpit: deploy + hot-swap code overlays, live telemetry ----
@app.get("/api/tensix/bl/overlays")
async def tensix_bl_overlays():
    return dm.tensix_bl_overlays()

@app.get("/api/tensix/bl/scan")
async def tensix_bl_scan():
    return await dm.tensix_bl_scan()

@app.get("/api/tensix/bl/status")
async def tensix_bl_status(x: int, y: int):
    try:
        return await dm.tensix_bl_status(x, y)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/tensix/bl/param")
async def tensix_bl_param(req: TensixBlParamRequest):
    try:
        return await dm.tensix_bl_param(req.x, req.y, req.index, req.value)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/tensix/bl/stage")
async def tensix_bl_stage(req: TensixBlStageRequest):
    try:
        return await dm.tensix_bl_stage(req.x, req.y, req.overlay, req.slot)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))

@app.post("/api/tensix/bl/exec")
async def tensix_bl_exec(req: TensixBlExecRequest):
    try:
        return await dm.tensix_bl_exec(req.x, req.y, req.slot, req.wait, req.timeout, req.force)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/tensix/bl/halt")
async def tensix_bl_halt(req: TensixBlHaltRequest):
    try:
        return await dm.tensix_bl_halt(req.x, req.y)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/tensix/bl/compile")
async def tensix_bl_compile(req: TensixBlCompileRequest):
    return await dm.tensix_bl_compile(req.name, req.source)

@app.get("/api/tensix/bl/source")
async def tensix_bl_source(name: str):
    try:
        return dm.tensix_bl_source(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

@app.post("/api/tensix/bl/source")
async def tensix_bl_save_source(req: TensixBlSourceRequest):
    return await dm.tensix_bl_save_source(req.name, req.source)

@app.get("/api/tensix/bl/launch")
async def tensix_bl_launch_status():
    return dm.bl_launch_status()

@app.post("/api/tensix/bl/launch")
async def tensix_bl_launch(req: TensixBlLaunchRequest):
    return await dm.tensix_bl_launch(req.grid)

@app.post("/api/tensix/bl/launch/stop")
async def tensix_bl_launch_stop():
    return await dm.tensix_bl_launch_stop()


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


@app.get("/api/tlab/buildlog")
async def tlab_buildlog():
    return await dm.tlab_build_log()


@app.post("/api/tlab/rebuild")
async def tlab_rebuild():
    return await dm.tlab_rebuild()


@app.get("/api/tlab/recipe")
async def tlab_recipe(example: str):
    return await dm.tlab_recipe(example)


@app.post("/api/tlab/extract")
async def tlab_extract(req: TlabExampleRequest):
    try:
        return await dm.tlab_extract(req.example)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/build")
async def tlab_build_standalone(req: TlabExampleRequest):
    try:
        return await dm.tlab_build_standalone(req.example)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/running")
async def running():
    """tt-metal kernels live now, keyed by JIT build hash (source<->hash<->program<->coords).
    Drives the device tree's running/stale badges."""
    return await dm.running()


@app.get("/api/telemetry")
async def telemetry():
    """Latest sampled NoC frame, for HTTP polling clients (simpler + more robust than the WS:
    the poll loop already samples the device at `hz` and caches the frame, so this just returns
    the cache — no per-request device touch). The /ws/telemetry WebSocket remains for push use."""
    return dm.last_frame() or {"ts": 0, "mode": dm.mode, "tiles": {}, "dram": {}}


# ---- UI defaults: persist the chip-view style/calibration to a tracked repo file so a good
# pathfinding layout can be committed to git and shared (vs. per-browser localStorage) --------
import json
UICONF = DIST.parent / "ui-defaults.json"   # ~/bhtop/frontend/ui-defaults.json (tracked)


@app.get("/api/uiconfig")
async def uiconfig_get():
    try:
        return json.loads(UICONF.read_text())
    except Exception:
        return {}


@app.post("/api/uiconfig")
async def uiconfig_set(cfg: dict):
    UICONF.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return {"ok": True, "path": str(UICONF)}


@app.post("/api/tlab/file/duplicate")
async def tlab_file_duplicate(req: CopyRequest):
    try:
        return await dm.tlab_copy(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/tlab/files")
async def tlab_files(example: str):
    return await dm.tlab_files(example)


@app.get("/api/tlab/tree")
async def tlab_tree():
    return await dm.tlab_tree()


@app.get("/api/tlab/params")
async def tlab_params(key: str):
    try:
        return await dm.tlab_params(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tlab/config")
async def tlab_config(key: str):
    try:
        return await dm.tlab_config_get(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/config")
async def tlab_config_save(req: KernelConfigRequest):
    try:
        return await dm.tlab_config_put(req.key, req.text)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/merge")
async def tlab_merge(req: KernelMergeRequest):
    try:
        return await dm.tlab_merge_params(req.key)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/tlab/restore")
async def tlab_restore():
    return await dm.tlab_restore()


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


@app.get("/api/lab/tree")
async def lab_tree():
    return await dm.lab_tree()


@app.get("/api/lab/params")
async def lab_params(key: str):
    try:
        return await dm.lab_params(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/lab/config")
async def lab_config(key: str):
    try:
        return await dm.lab_config_get(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/config")
async def lab_config_save(req: KernelConfigRequest):
    try:
        return await dm.lab_config_put(req.key, req.text)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/merge")
async def lab_merge(req: KernelMergeRequest):
    try:
        return await dm.lab_merge_params(req.key)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/lab/restore")
async def lab_restore():
    return await dm.lab_restore()


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


# ---- Tensix ISA (assembly.yaml): decoded opcodes + per-operand bit-fields for the cockpit's
# instruction tooltips + searchable reference panel. Pure host (parsed + cached), no device. ----
@app.get("/api/tensix/isa")
async def tensix_isa():
    from ..tensix import isa as tensix_isa_mod
    return await asyncio.to_thread(tensix_isa_mod.load)


@app.get("/api/tensix/isa/{mnemonic}")
async def tensix_isa_one(mnemonic: str):
    from ..tensix import isa as tensix_isa_mod
    info = await asyncio.to_thread(tensix_isa_mod.one, mnemonic)
    if not info:
        raise HTTPException(404, f"unknown Tensix instruction {mnemonic!r}")
    return info


# ---- LLK perf kernels: tt-llk's tests/sources/*_perf.cpp, built on llk_lib, imported into
# folder-per-kernel canon with per-thread (unpack/math/pack) metadata. Pure host (reads canon). ----
@app.get("/api/tensix/llk")
async def tensix_llk():
    from ..tensix import llk
    return await asyncio.to_thread(llk.load)


@app.get("/api/tensix/llk/{name}")
async def tensix_llk_source(name: str):
    from ..tensix import llk
    try:
        return {"name": name, "source": await asyncio.to_thread(llk.source, name), "lang": "cpp"}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/tensix/llk/build")
async def tensix_llk_build(req: LlkBuildRequest):
    return await dm.llk_build(req.name, req.run_type)


@app.get("/api/tensix/llk/{name}/disasm")
async def tensix_llk_disasm(name: str):
    return await dm.llk_disasm(name)


@app.get("/api/tensix/bl/disasm")
async def tensix_bl_disasm(name: str):
    return await dm.overlay_disasm(name)


@app.post("/api/tensix/llk/run")
async def tensix_llk_run(req: LlkRunRequest):
    if dm.reset_needed:
        raise HTTPException(409, "NoC hang pending — run `tt-smi -r 0` and restart the server")
    try:
        return await dm.llk_run(req.name, req.x, req.y, req.tile_cnt, req.timeout, req.run_type)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tensix/llk/test_all")
async def tensix_llk_test_all(req: LlkTestAllRequest):
    """Build every LLK kernel once + run each on every core (default all Tensix cores); background
    job. Poll /api/tensix/llk/test_all/last for live progress + the final pass/fail overview."""
    if dm.reset_needed:
        raise HTTPException(409, "NoC hang pending — run `tt-smi -r 0` and restart the server")
    return await dm.llk_test_all(req.cores, req.run_type, req.tile_cnt, req.timeout)


@app.get("/api/tensix/llk/test_all/last")
async def tensix_llk_test_all_last():
    return dm.llk_test_last()


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


@app.get("/api/l2/tree")
async def l2_tree():
    return await dm.l2_tree()


@app.get("/api/l2/params")
async def l2_params(key: str):
    try:
        return await dm.l2_params(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/params")
async def l2_params_save(req: L2ParamsRequest):
    try:
        return await dm.l2_save_params(req.key, req.values)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/l2/config")
async def l2_config(key: str):
    try:
        return await dm.l2_config_get(key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/config")
async def l2_config_save(req: KernelConfigRequest):
    try:
        return await dm.l2_config_put(req.key, req.text)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/merge")
async def l2_merge(req: KernelMergeRequest):
    try:
        return await dm.l2_merge_params(req.key)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/folder/new")
async def l2_folder_new(req: L2FolderRequest):
    try:
        return await dm.l2_folder_new(req.path)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/folder/duplicate")
async def l2_folder_duplicate(req: CopyRequest):
    try:
        return await dm.l2_folder_dup(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/folder/rename")
async def l2_folder_rename(req: CopyRequest):
    try:
        return await dm.l2_folder_rename(req.src, req.name)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/folder/delete")
async def l2_folder_delete(req: L2FolderRequest):
    try:
        return await dm.l2_folder_delete(req.path)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/l2/regenerate")
async def l2_regenerate():
    return await dm.l2_regenerate()


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
    return await dm.l2_compile(req.content, req.lang, req.addr, req.defines)


@app.post("/api/l2/bringup")
async def l2_bringup(req: L2TileRequest):
    return await dm.l2_bringup(req.tile)


@app.get("/api/l2/bringup/last")
async def l2_bringup_last():
    return dm.l2_bringup_last()


@app.post("/api/l2/deploy")
async def l2_deploy(req: L2DeployRequest):
    return await dm.l2_deploy(req.tile, req.hart, req.content, req.lang, req.addr, req.name, req.defines)


@app.post("/api/l2/deploy_all")
async def l2_deploy_all(req: L2DeployRequest):
    return await dm.l2_deploy_all(req.tile, req.content, req.lang, req.addr, req.name, req.defines, req.harts)


@app.post("/api/l2/park_all")
async def l2_park_all(req: L2TileRequest):
    return await dm.l2_park_all(req.tile)


@app.post("/api/l2/tele/zero")
async def l2_tele_zero(req: L2TileRequest):
    return await dm.l2_zero_tele(req.tile, req.hart)


@app.post("/api/l2/poke")
async def l2_poke(req: L2PokeRequest):
    return await dm.l2_poke(req.tile, req.addr, req.val)


@app.post("/api/l2/cmd")
async def l2_cmd(req: L2CommandRequest):
    """Ring a hart's command mailbox (the 'keypress' — steer the virus, set seed, mutate)."""
    return await dm.l2_cmd(req.tile, req.hart, req.op, req.arg0, req.arg1)


@app.get("/api/l2/vec")
async def l2_vec(tile: int, hart: int, ew: int = 32):
    """Decode a hart's vector-register dump (v0..v31 + vector CSRs); needs bh_dump_vec()."""
    return await dm.l2_vec(tile, hart, ew)


@app.get("/api/l2/power")
async def l2_power():
    """Live board power / current / temperature (ARC telemetry)."""
    return await dm.l2_power()


@app.get("/api/l2/clocks")
async def l2_clocks():
    """Core (l2cpuclk) vs uncore (axiclk/arcclk) vs Tensix (aiclk) frequencies."""
    return await dm.l2_clocks()


@app.post("/api/l2/freq")
async def l2_freq(req: L2FreqRequest):
    """Set the L2CPU CORE PLL (verified points only; uncore is the transport, not settable)."""
    try:
        return await dm.l2_freq(req.mhz)
    except ValueError as e:
        raise HTTPException(400, str(e))


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


@app.websocket("/ws/bootloader")
async def ws_bootloader(websocket: WebSocket):
    await websocket.accept()
    q = await dm.bl_subscribe()

    async def sender():
        while True:
            await websocket.send_json(await q.get())

    async def receiver():                    # client picks the focused core + rate
        while True:
            msg = await websocket.receive_json()
            dm.bl_select(msg.get("x"), msg.get("y"), msg.get("hz"))

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for tk in pending:
            tk.cancel()
        for tk in done:
            tk.exception()
    finally:
        dm.bl_unsubscribe(q)


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
