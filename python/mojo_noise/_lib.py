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


class _Cfg(ctypes.Structure):
    """Matches the CFG_* offsets read by the Mojo scalar entry points."""
    # Byte offsets must match CFG_* in src/noise.mojo.
    _fields_ = [
        ("octaves", I),      # 0
        ("persistence", F),  # 8
        ("lacunarity", F),   # 12
        ("has0", I),         # 16
        ("repeat0", F),      # 24
        ("has1", I),         # 32
        ("repeat1", F),      # 40
        ("repeat2", F),      # 44
        ("base", F),         # 48
        ("perm", I),         # 56
        ("repeat3", F),      # 64
    ]

_SIGNATURES = {
    "mn_pnoise1_cfg": ([F, ctypes.POINTER(_Cfg)], F),
    "mn_pnoise2_cfg": ([F, F, ctypes.POINTER(_Cfg)], F),
    "mn_pnoise3_cfg": ([F, F, F, ctypes.POINTER(_Cfg)], F),
    "mn_snoise2_cfg": ([F, F, ctypes.POINTER(_Cfg)], F),
    "mn_snoise3_cfg": ([F, F, F, ctypes.POINTER(_Cfg)], F),
    "mn_snoise4_cfg": ([F, F, F, F, ctypes.POINTER(_Cfg)], F),
    "mn_pnoise1_array": ([I, I, I, I, F, F, I, I, I], None),
    "mn_pnoise2_array": ([I, I, I, I, I, F, F, F, F, I, I], None),
    "mn_pnoise3_array": ([I, I, I, I, I, I, F, F, I, I, I, I, I], None),
    "mn_snoise2_array": ([I, I, I, I, I, F, F, I, F, I, F, F, I], None),
    "mn_snoise3_array": ([I, I, I, I, I, I, F, F, I], None),
    "mn_snoise4_array": ([I, I, I, I, I, I, I, F, F, I], None),
    "mn_snoise4_array_gpu": ([I, I, I, I, I, I, I, F, F, I], I),
}


def config(
    *,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base: float = 0.0,
    perm_addr: int,
    has0: int = 0,
    repeat0: float = 0.0,
    has1: int = 0,
    repeat1: float = 0.0,
    repeat2: float = 0.0,
    repeat3: float = 0.0,
) -> ctypes.POINTER(_Cfg):
    """Return a cached config struct for the scalar entry points.

    ctypes marshalling costs roughly half a microsecond per argument, so a
    scalar call is dominated by the ABI, not the kernel. Passing the constant
    fBm parameters as one struct the kernel reads directly cuts a 9-argument
    call to 3. The struct is only rewritten when a parameter actually
    changes, so the steady-state cost is one pointer argument.
    """
    global _cfg, _cfg_key
    key = (
        octaves, persistence, lacunarity, base, perm_addr, has0, repeat0,
        has1, repeat1, repeat2, repeat3,
    )
    if key != _cfg_key:
        _cfg = _Cfg(
            octaves, persistence, lacunarity, has0, repeat0, has1,
            repeat1, repeat2, base, perm_addr, repeat3,
        )
        _cfg_key = key
    return ctypes.pointer(_cfg)


_cfg: _Cfg | None = None
_cfg_key: tuple | None = None


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
