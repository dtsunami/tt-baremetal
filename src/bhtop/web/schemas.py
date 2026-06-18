"""Request bodies for the bhtop-web API (responses are plain dicts from DeviceManager)."""
from pydantic import BaseModel


class InjectRequest(BaseModel):
    src: list[int]            # [x, y] noc0 coords of the source tensix tile
    pattern: str = "gddr6_write"
    length: int = 0x40000     # bytes per pair per fire (clamped to MAX_LEN by the injector)
    fires: int = 3
    stream: bool = True       # keep re-firing each poll tick so traffic is sustained


class KernelRunRequest(BaseModel):
    name: str                 # gtest name from /api/kernels
    timeout: int = 1800       # first run JIT-compiles kernels — can take >15 min
    dprint: bool = False      # enable on-device DPRINT capture (slower; floods on big grids)
    dprint_cores: str = "0,0" # TT_METAL_DPRINT_CORES value when dprint is on


class LabWriteRequest(BaseModel):
    path: str                 # project-relative source path from /api/lab/files
    content: str


class LabPathRequest(BaseModel):
    path: str                 # project-relative source path (revert)


class LabBuildRequest(BaseModel):
    target: str = "unit_tests_data_movement"


# ---- L2CPU cockpit ----
class L2DeployRequest(BaseModel):
    tile: int                 # 0..3
    hart: int = 0             # single-hart deploy (/api/l2/deploy)
    harts: list[int] | None = None  # subset for deploy_all (None = all 4); any grouping
    content: str              # kernel source (the editor buffer)
    lang: str = "c"           # c | asm | rust
    addr: int = 0x30008000    # load address (user-code window, above the data blocks)
    name: str = ""            # source filename, recorded so the UI shows what's on each hart
    defines: dict = {}        # define-kind meta-params injected at compile (name -> int|hex)


class L2CompileRequest(BaseModel):
    content: str
    lang: str = "c"
    addr: int = 0x30008000
    defines: dict = {}        # define-kind meta-params injected at compile (name -> int|hex)


class L2CommandRequest(BaseModel):
    tile: int
    hart: int
    op: int                   # see regmap.CMD_OPS (10=select_class, 11=set_seed, 12=mutate, …)
    arg0: int = 0
    arg1: int = 0


class L2FreqRequest(BaseModel):
    mhz: int                  # L2CPU core PLL target — only verified points (200, 1750) allowed


class L2TileRequest(BaseModel):
    tile: int
    hart: int | None = None   # optional: tele/zero a single hart's window (None = all)


class L2WriteRequest(BaseModel):
    name: str                 # workspace filename from /api/l2/files
    content: str


class L2NewRequest(BaseModel):
    name: str
    lang: str = "c"


class L2PokeRequest(BaseModel):
    tile: int
    addr: int
    val: int


# ---- tlab: Tensix Compute Lab ----
class TlabRunRequest(BaseModel):
    name: str                 # compute example binary name from /api/tlab/examples
    timeout: int = 900        # JIT-compiles compute kernels on first run


class CopyRequest(BaseModel):
    src: str                  # existing file (path or name) to duplicate
    name: str                 # new file name for the copy (a fresh variation)


# ---- folder browser + per-kernel meta-params ----
class L2FolderRequest(BaseModel):
    path: str                 # workspace-relative folder path (new / delete)


class L2ParamsRequest(BaseModel):
    key: str                  # a file key inside the kernel folder whose kernel.json to update
    values: dict = {}         # {param_name: value} to persist as the kernel's new defaults


class KernelConfigRequest(BaseModel):
    key: str                  # a file key inside the kernel (resolves to its kernel.json)
    text: str                 # raw kernel.json text from the JSON editor (validated on write)


class KernelMergeRequest(BaseModel):
    key: str                  # a file key inside the kernel; parse its source(s) + merge params
