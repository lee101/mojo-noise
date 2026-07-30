# mojo-noise

`mojo-noise` is a Mojo port of the compute-heavy noise generators from Casey
Duncan's Python [`noise`](https://pypi.org/project/noise/) package. It provides
drop-in scalar signatures for improved Perlin noise and simplex noise, plus
NumPy array support for generating complete textures and point clouds in one
FFI call.

The port targets `noise` 1.2.2 and deliberately preserves its float32
arithmetic, fixed permutation table, repeat handling, base offsets, and
normalized fractional Brownian motion (fBm) behavior.

## Coverage

Covered:

- `pnoise1`, `pnoise2`, and `pnoise3`
- `snoise2`, `snoise3`, and `snoise4`
- `octaves`, `persistence`, and `lacunarity`
- Perlin repeat periods and integer base offsets
- `snoise2` one-axis and two-axis tiling and floating-point base offsets
- Broadcastable NumPy coordinate arrays, also available through the explicit
  `*_array` aliases

Not covered:

- the upstream package's GLSL shader text helpers
- the older pure-Python `BaseNoise` and `TileableNoise` classes
- user-defined permutation tables or seeded tables, which are not part of the
  upstream six-function API

## Install

The repository pins the Mojo nightly used to build and test it:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-noise.so`. Pixi sets `PYTHONPATH=python` for
the repository tasks and commands.

## Usage

Scalar calls use the same names and argument order as `noise`:

```python
import mojo_noise as noise

value = noise.pnoise3(
    0.1, 0.2, 0.3,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    repeatx=64,
    repeaty=64,
    repeatz=64,
    base=7,
)
print(value)
```

Passing arrays evaluates a whole broadcast grid in one Mojo call:

```python
import numpy as np
import mojo_noise as noise

x = np.linspace(0, 8, 512, dtype=np.float32)[:, None]
y = np.linspace(0, 8, 512, dtype=np.float32)[None, :]
texture = noise.snoise2(x, y, octaves=4)
print(texture.shape, texture.dtype)  # (512, 512) float32
```

Large 4D simplex batches can explicitly use the GPU:

```python
cloud = noise.snoise4(x, y, z, w, octaves=3, device="gpu")
```

CPU is always the default. An explicit GPU request raises an error if the GPU
is unavailable, has insufficient free memory, the batch exceeds the configured
device-memory limit, or GPU execution fails.

Run these examples with `pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
72 logical CPUs, Linux 6.8.0-136-generic, glibc 2.39. Times are the best of
three runs and include the public Python wrapper. Array rows compare one
`mojo-noise` batch call with the scalar loop required by upstream `noise`
1.2.2.

| Kernel | mojo-noise | noise 1.2.2 | Speedup |
|---|---:|---:|---:|
| pnoise2 262k points, 1 octave | 20.86 ms | 280.54 ms | 13.45x |
| pnoise2 262k points, 6 octaves | 16.55 ms | 391.63 ms | 23.66x |
| snoise2 262k points, 1 octave | 18.21 ms | 301.37 ms | 16.55x |
| snoise3 200k points, 4 octaves | 101.83 ms | 514.67 ms | 5.05x |
| snoise4 100k points, 3 octaves | 14.41 ms | 179.36 ms | 12.44x |
| pnoise2 20k scalar Python calls | 78.58 ms | 5.59 ms | 0.07x |

The optional GPU path is benchmarked at a size large enough to amortize context
creation and transfers:

| GPU kernel | CPU | GPU | GPU speedup |
|---|---:|---:|---:|
| snoise4 1m points, 3 octaves | 85.29 ms | 38.18 ms | 2.23x |

The scalar result is intentionally included: upstream's direct CPython C
extension is much faster for repeated one-point calls. This port is useful
when coordinates can be submitted as arrays, allowing the FFI and array
conversion costs to be amortized.

## How it works

All kernels live in one Mojo compilation unit and are exported through a small
C ABI. Python loads the shared library with `ctypes`. Scalar coordinates cross
as C `float`; batches cross as integer addresses because Mojo exports cannot
use parametric pointer origins.

The Python wrapper broadcasts coordinates, converts them to C-contiguous
float32 arrays, and allocates one C-contiguous float32 result. Multi-octave
`pnoise2` and `snoise4` process SIMD-width point blocks with a scalar tail and
use up to 16 CPU workers only for arrays of at least 16384 points. The caller
owns every CPU buffer, while the fixed permutation table is held by the Python
module and passed as a read-only byte buffer address. Contiguous NumPy inputs
remain zero-copy across the CPU FFI boundary.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares scalar and randomized array results with the installed
upstream package, including fBm parameters, repeat periods, simplex tiling,
base offsets, broadcasting, signatures, and errors.
