"""
bhtop.tensix — drive Tensix worker kernels at the L1/firmware level over the NoC (tt-exalens),
the way bhtop.l2cpu drives the x280 harts. `abi` is the pure Blackhole launch-ABI map; `loader`
is the device-side reader/poker for live runtime-arg editing. See TENSIX_ABI.md.
"""
from . import abi
from . import bootloader
from .bootloader import Bootloader
from .loader import TensixLauncher, worker_coords

__all__ = ["abi", "bootloader", "Bootloader", "TensixLauncher", "worker_coords"]
