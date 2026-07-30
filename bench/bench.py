"""Benchmark Mojo batch kernels against noise 1.2.2 scalar calls."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "python",
    ),
)

import mojo_noise as mojo  # noqa: E402
import noise as upstream  # noqa: E402


@dataclass
class Case:
    name: str
    ours: Callable
    theirs: Callable
    atol: float


def timeit(fn: Callable, repeat: int = 3) -> tuple[float, np.ndarray | float]:
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def reference_array(fn: Callable, coords: tuple[np.ndarray, ...], kwargs: dict):
    return np.fromiter(
        (
            fn(*(float(coord[i]) for coord in coords), **kwargs)
            for i in range(coords[0].size)
        ),
        dtype=np.float32,
        count=coords[0].size,
    )


def array_case(
    name: str,
    function_name: str,
    coords: tuple[np.ndarray, ...],
    kwargs: dict | None = None,
    atol: float = 3e-5,
) -> Case:
    kwargs = kwargs or {}
    ours_fn = getattr(mojo, function_name)
    theirs_fn = getattr(upstream, function_name)
    return Case(
        name,
        lambda: ours_fn(*coords, **kwargs),
        lambda: reference_array(theirs_fn, coords, kwargs),
        atol,
    )


def machine() -> str:
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model or 'unknown CPU'}; {os.cpu_count()} logical CPUs; {platform.platform()}"


def gpu_has_headroom() -> tuple[bool, str]:
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
            return False, "nvidia-smi unavailable"
        free_mib = int(proc.stdout.splitlines()[0].split()[0])
        if free_mib < 4000:
            return False, f"only {free_mib} MiB free"
        return True, f"{free_mib} MiB free"
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return False, "GPU memory could not be queried"


def cases() -> list[Case]:
    rng = np.random.default_rng(0)
    n2 = 512 * 512
    x2 = np.ascontiguousarray(rng.uniform(-32, 32, n2), dtype=np.float32)
    y2 = np.ascontiguousarray(rng.uniform(-32, 32, n2), dtype=np.float32)
    n3 = 200_000
    xyz = tuple(
        np.ascontiguousarray(rng.uniform(-16, 16, n3), dtype=np.float32)
        for _ in range(3)
    )
    n4 = 100_000
    xyzw = tuple(
        np.ascontiguousarray(rng.uniform(-8, 8, n4), dtype=np.float32)
        for _ in range(4)
    )
    scalar_points = tuple(zip(x2[:20_000], y2[:20_000]))
    return [
        array_case("pnoise2 262k points, 1 octave", "pnoise2", (x2, y2)),
        array_case(
            "pnoise2 262k points, 6 octaves",
            "pnoise2",
            (x2, y2),
            {"octaves": 6},
        ),
        array_case("snoise2 262k points, 1 octave", "snoise2", (x2, y2)),
        array_case(
            "snoise3 200k points, 4 octaves",
            "snoise3",
            xyz,
            {"octaves": 4},
        ),
        array_case(
            "snoise4 100k points, 3 octaves",
            "snoise4",
            xyzw,
            {"octaves": 3},
        ),
        Case(
            "pnoise2 20k scalar Python calls",
            lambda: sum(mojo.pnoise2(float(x), float(y)) for x, y in scalar_points),
            lambda: sum(
                upstream.pnoise2(float(x), float(y)) for x, y in scalar_points
            ),
            1e-4,
        ),
    ]


def benchmark_gpu() -> None:
    rng = np.random.default_rng(1)
    n = 1_000_000
    coords = tuple(
        np.ascontiguousarray(rng.uniform(-8, 8, n), dtype=np.float32)
        for _ in range(4)
    )
    cpu = lambda: mojo.snoise4(*coords, octaves=3)
    gpu = lambda: mojo.snoise4(*coords, octaves=3, device="gpu")
    cpu()
    gpu()
    cpu_time, expected = timeit(cpu)
    gpu_time, actual = timeit(gpu)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)
    print()
    print("| GPU kernel | CPU | GPU | GPU speedup |")
    print("|---|---:|---:|---:|")
    print(
        f"| snoise4 1m points, 3 octaves | {cpu_time * 1e3:.2f} ms | "
        f"{gpu_time * 1e3:.2f} ms | {cpu_time / gpu_time:.2f}x |"
    )


def main() -> None:
    print(f"Machine: {machine()}")
    gpu_ok, gpu_status = gpu_has_headroom()
    if not gpu_ok:
        print(f"GPU benchmark skipped: {gpu_status}.")
    print()
    print("| Kernel | mojo-noise | noise 1.2.2 | Speedup |")
    print("|---|---:|---:|---:|")
    for case in cases():
        case.ours()
        mojo_time, actual = timeit(case.ours)
        upstream_time, expected = timeit(case.theirs)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=case.atol)
        speedup = upstream_time / mojo_time
        print(
            f"| {case.name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {speedup:.2f}x |"
        )
    if gpu_ok:
        benchmark_gpu()


if __name__ == "__main__":
    main()
