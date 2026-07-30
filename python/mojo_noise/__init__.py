"""Perlin and simplex noise implemented in Mojo.

The six public function names and scalar signatures match the ``noise``
package. As an extension, coordinates may be broadcastable NumPy arrays.
"""

from __future__ import annotations

import operator
import subprocess
from typing import Any

import numpy as np

from ._lib import lib

__version__ = "0.1.0"

_PERM_256 = np.array(
    [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68,
        175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111,
        229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
        102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187,
        208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100,
        109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38,
        147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17,
        182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
        70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
        108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
        251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51,
        145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205,
        93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61,
        156, 180,
    ],
    dtype=np.uint8,
)
_PERM = np.tile(_PERM_256, 4)
_PERM_ADDR = _PERM.ctypes.data
_GPU_MIN_FREE_MIB = 4000
_GPU_MAX_POINTS = 75_000_000
_PNOISE2_DEFAULT = None


def _int(value: Any) -> int:
    return operator.index(value)


def _common(octaves: Any, persistence: Any, lacunarity: Any) -> tuple[int, float, float]:
    octaves = _int(octaves)
    if octaves <= 0:
        raise ValueError("Expected octaves value > 0")
    return octaves, float(persistence), float(lacunarity)


def _base(value: Any) -> int:
    value = _int(value)
    if not 0 <= value <= 255:
        raise ValueError("base must be between 0 and 255")
    return value


def _period_int(value: Any, name: str) -> int:
    value = _int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _period_float(value: Any, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _scalar_coords(*coords: Any) -> bool:
    return all(
        np.isscalar(coord)
        or (isinstance(coord, np.ndarray) and coord.ndim == 0)
        for coord in coords
    )


def _gpu_has_headroom() -> bool:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode:
            return False
        free_mib = int(proc.stdout.splitlines()[0].split()[0])
        return free_mib >= _GPU_MIN_FREE_MIB
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return False


def _arrays(*coords: Any) -> tuple[list[np.ndarray], tuple[int, ...], bool]:
    scalar = all(np.ndim(coord) == 0 for coord in coords)
    broadcast = np.broadcast_arrays(
        *(np.asarray(coord, dtype=np.float32) for coord in coords)
    )
    arrays = [
        np.ascontiguousarray(coord, dtype=np.float32).reshape(-1)
        for coord in broadcast
    ]
    return arrays, broadcast[0].shape, scalar


def _finish(result: np.ndarray, shape: tuple[int, ...], scalar: bool):
    if scalar:
        return float(result[0])
    return result.reshape(shape)


def _result(arrays: list[np.ndarray], shape: tuple[int, ...]):
    result = np.empty(arrays[0].size, dtype=np.float32)
    if result.size == 0:
        return result.reshape(shape)
    return result


def pnoise1(
    x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0
):
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    repeat = _period_int(repeat, "repeat")
    base = _base(base)
    if _scalar_coords(x):
        return float(
            lib().mn_pnoise1(
                float(x), octaves, persistence, lacunarity,
                repeat, base, _PERM_ADDR,
            )
        )
    arrays, shape, scalar = _arrays(x)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    lib().mn_pnoise1_array(
        arrays[0].ctypes.data, result.ctypes.data, result.size, octaves,
        persistence, lacunarity, repeat, base, _PERM_ADDR,
    )
    return _finish(result, shape, scalar)


def pnoise2(
    x, y, octaves=1, persistence=0.5, lacunarity=2.0,
    repeatx=1024, repeaty=1024, base=0,
):
    global _PNOISE2_DEFAULT
    if (
        type(x) is float
        and type(y) is float
        and octaves == 1
        and persistence == 0.5
        and lacunarity == 2.0
        and repeatx == 1024
        and repeaty == 1024
        and base == 0
    ):
        if _PNOISE2_DEFAULT is None:
            _PNOISE2_DEFAULT = lib().mn_pnoise2
        return _PNOISE2_DEFAULT(
            x, y, 1, 0.5, 2.0, 1024.0, 1024.0, 0, _PERM_ADDR
        )
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    repeatx = _period_float(repeatx, "repeatx")
    repeaty = _period_float(repeaty, "repeaty")
    base = _base(base)
    if _scalar_coords(x, y):
        return float(
            lib().mn_pnoise2(
                float(x), float(y), octaves, persistence, lacunarity,
                repeatx, repeaty, base, _PERM_ADDR,
            )
        )
    arrays, shape, scalar = _arrays(x, y)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    lib().mn_pnoise2_array(
        arrays[0].ctypes.data, arrays[1].ctypes.data, result.ctypes.data,
        result.size, octaves, persistence, lacunarity, repeatx, repeaty,
        base, _PERM_ADDR,
    )
    return _finish(result, shape, scalar)


def pnoise3(
    x, y, z, octaves=1, persistence=0.5, lacunarity=2.0,
    repeatx=1024, repeaty=1024, repeatz=1024, base=0,
):
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    repeatx = _period_int(repeatx, "repeatx")
    repeaty = _period_int(repeaty, "repeaty")
    repeatz = _period_int(repeatz, "repeatz")
    base = _base(base)
    if _scalar_coords(x, y, z):
        return float(
            lib().mn_pnoise3(
                float(x), float(y), float(z), octaves,
                persistence, lacunarity, repeatx, repeaty, repeatz,
                base, _PERM_ADDR,
            )
        )
    arrays, shape, scalar = _arrays(x, y, z)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    lib().mn_pnoise3_array(
        *(array.ctypes.data for array in arrays), result.ctypes.data,
        result.size, octaves, persistence, lacunarity, repeatx, repeaty,
        repeatz, base, _PERM_ADDR,
    )
    return _finish(result, shape, scalar)


def snoise2(
    x, y, octaves=1, persistence=0.5, lacunarity=2.0,
    repeatx=None, repeaty=None, base=0.0,
):
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    has_repeatx = repeatx is not None
    has_repeaty = repeaty is not None
    rx = _period_float(repeatx, "repeatx") if has_repeatx else 1.0
    ry = _period_float(repeaty, "repeaty") if has_repeaty else 1.0
    base = float(base)
    args = (
        octaves, persistence, lacunarity, int(has_repeatx), rx,
        int(has_repeaty), ry, base, _PERM_ADDR,
    )
    if _scalar_coords(x, y):
        return float(
            lib().mn_snoise2(float(x), float(y), *args)
        )
    arrays, shape, scalar = _arrays(x, y)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    lib().mn_snoise2_array(
        arrays[0].ctypes.data, arrays[1].ctypes.data, result.ctypes.data,
        result.size, *args,
    )
    return _finish(result, shape, scalar)


def snoise3(
    x, y, z, octaves=1, persistence=0.5, lacunarity=2.0
):
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    if _scalar_coords(x, y, z):
        return float(
            lib().mn_snoise3(
                float(x), float(y), float(z), octaves,
                persistence, lacunarity, _PERM_ADDR,
            )
        )
    arrays, shape, scalar = _arrays(x, y, z)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    lib().mn_snoise3_array(
        *(array.ctypes.data for array in arrays), result.ctypes.data,
        result.size, octaves, persistence, lacunarity, _PERM_ADDR,
    )
    return _finish(result, shape, scalar)


def snoise4(
    x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0, device="cpu"
):
    octaves, persistence, lacunarity = _common(octaves, persistence, lacunarity)
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if _scalar_coords(x, y, z, w):
        return float(
            lib().mn_snoise4(
                float(x), float(y), float(z), float(w), octaves,
                persistence, lacunarity, _PERM_ADDR,
            )
        )
    arrays, shape, scalar = _arrays(x, y, z, w)
    result = _result(arrays, shape)
    if result.size == 0:
        return result
    if device == "gpu":
        if result.size > _GPU_MAX_POINTS:
            raise ValueError(
                f"GPU batches are limited to {_GPU_MAX_POINTS} points"
            )
        if not _gpu_has_headroom():
            raise RuntimeError("GPU unavailable or has insufficient free memory")
        succeeded = lib().mn_snoise4_array_gpu(
            *(array.ctypes.data for array in arrays), result.ctypes.data,
            result.size, octaves, persistence, lacunarity, _PERM_ADDR,
        )
        if succeeded:
            return _finish(result, shape, scalar)
        raise RuntimeError("GPU execution failed")
    lib().mn_snoise4_array(
        *(array.ctypes.data for array in arrays), result.ctypes.data,
        result.size, octaves, persistence, lacunarity, _PERM_ADDR,
    )
    return _finish(result, shape, scalar)


pnoise1_array = pnoise1
pnoise2_array = pnoise2
pnoise3_array = pnoise3
snoise2_array = snoise2
snoise3_array = snoise3
snoise4_array = snoise4

__all__ = [
    "pnoise1", "pnoise2", "pnoise3", "snoise2", "snoise3", "snoise4",
    "pnoise1_array", "pnoise2_array", "pnoise3_array",
    "snoise2_array", "snoise3_array", "snoise4_array",
]
