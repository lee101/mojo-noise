import os
import subprocess
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "python",
    ),
)


_PROBE = "import noise; noise.snoise4(0., 0., 0., 0., octaves=2, lacunarity=1.7)"


def _repair_snoise4_oracle() -> None:
    """Restore noise.snoise4 when the installed build corrupts memory.

    noise 1.2.2 ships py_noise4 calling PyArg_ParseTupleAndKeywords with the
    format "ffff|iff:snoise4" but only six destination pointers, so any call
    supplying lacunarity makes CPython read a seventh variadic argument that
    was never pushed and store the float through it. A fresh process dies with
    SIGSEGV; a long-lived one has its heap overwritten and then spins forever.
    Upstream master passes &lacunarity, so rebuild the same value from the
    oracle's own working four-argument noise4 plus fbm_noise4.
    """
    try:
        import noise
    except ImportError:
        return
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, check=False
    )
    if probe.returncode == 0:
        return
    from noise import _simplex

    def snoise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0):
        if octaves <= 0:
            raise ValueError("Expected octaves value > 0")
        if octaves == 1:
            return _simplex.noise4(x, y, z, w)
        f = np.float32
        freq = amp = norm = f(1.0)
        total = f(_simplex.noise4(x, y, z, w))
        for _ in range(1, octaves):
            freq *= f(lacunarity)
            amp *= f(persistence)
            norm += amp
            scaled = (f(coord * freq) for coord in (x, y, z, w))
            total += f(_simplex.noise4(*scaled)) * amp
        return float(total / norm)

    noise.snoise4 = snoise4


_repair_snoise4_oracle()
