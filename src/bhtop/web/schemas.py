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
