"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "noise.mojo")
LIB = os.environ.get("MOJO_NOISE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-noise.so"
)

I = ctypes.c_int64
F = ctypes.c_float

_SIGNATURES = {
    "mn_pnoise1": ([F, I, F, F, I, I, I], F),
    "mn_pnoise2": ([F, F, I, F, F, F, F, I, I], F),
    "mn_pnoise3": ([F, F, F, I, F, F, I, I, I, I, I], F),
    "mn_snoise2": ([F, F, I, F, F, I, F, I, F, F, I], F),
    "mn_snoise3": ([F, F, F, I, F, F, I], F),
    "mn_snoise4": ([F, F, F, F, I, F, F, I], F),
    "mn_pnoise1_array": ([I, I, I, I, F, F, I, I, I], None),
    "mn_pnoise2_array": ([I, I, I, I, I, F, F, F, F, I, I], None),
    "mn_pnoise3_array": ([I, I, I, I, I, I, F, F, I, I, I, I, I], None),
    "mn_snoise2_array": ([I, I, I, I, I, F, F, I, F, I, F, F, I], None),
    "mn_snoise3_array": ([I, I, I, I, I, I, F, F, I], None),
    "mn_snoise4_array": ([I, I, I, I, I, I, I, F, F, I], None),
    "mn_snoise4_array_gpu": ([I, I, I, I, I, I, I, F, F, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_NOISE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.exists(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo source at {SRC} and no shared library at {LIB}; "
            "set MOJO_NOISE_LIB to a built libmojo-noise.so"
        )
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib
