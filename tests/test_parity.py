"""Numerical and behavioral parity with the noise 1.2.2 extension."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import mojo_noise as ours

reference = pytest.importorskip("noise")


FUNCTION_CASES = [
    ("pnoise1", (0.123,), {}),
    ("pnoise1", (-1.25,), {"octaves": 4, "repeat": 19}),
    ("pnoise1", (0.5,), {"base": 5}),
    ("pnoise2", (0.123, -0.45), {}),
    (
        "pnoise2",
        (-4.3, 2.1),
        {
            "octaves": 4,
            "persistence": 0.61,
            "lacunarity": 1.83,
            "repeatx": 17.5,
            "repeaty": 23.25,
        },
    ),
    ("pnoise2", (0.73, 0.27), {"base": 5}),
    ("pnoise3", (0.1, -0.2, 0.3), {}),
    (
        "pnoise3",
        (-4.3, 2.1, 5.6),
        {
            "octaves": 3,
            "persistence": 0.4,
            "lacunarity": 2.3,
            "repeatx": 17,
            "repeaty": 23,
            "repeatz": 31,
        },
    ),
    ("pnoise3", (0.1, 0.7, 0.33), {"base": 5}),
    ("snoise2", (0.123, -0.45), {}),
    (
        "snoise2",
        (-4.3, 2.1),
        {"octaves": 4, "persistence": 0.61, "lacunarity": 1.83, "base": 3.0},
    ),
    (
        "snoise2",
        (-4.3, 2.1),
        {"octaves": 3, "repeatx": 17, "repeaty": 23, "base": 2.0},
    ),
    ("snoise2", (2.5, -1.25), {"repeatx": 17}),
    ("snoise2", (2.5, -1.25), {"repeaty": 23}),
    ("snoise3", (0.1, -0.2, 0.3), {}),
    (
        "snoise3",
        (-4.3, 2.1, 5.6),
        {"octaves": 4, "persistence": 0.61, "lacunarity": 1.83},
    ),
    ("snoise4", (0.1, -0.2, 0.3, 0.4), {}),
    (
        "snoise4",
        (0.3, 0.6, 0.9, 1.2),
        {"octaves": 4, "persistence": 0.61},
    ),
]


@pytest.mark.parametrize(("name", "coords", "kwargs"), FUNCTION_CASES)
def test_scalar_parity(name, coords, kwargs):
    actual = getattr(ours, name)(*coords, **kwargs)
    expected = getattr(reference, name)(*coords, **kwargs)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, abs=3e-5)


@pytest.mark.parametrize(
    ("name", "dimensions", "kwargs", "atol"),
    [
        ("pnoise1", 1, {}, 3e-6),
        (
            "pnoise1",
            1,
            {"octaves": 5, "persistence": 0.61, "lacunarity": 1.83, "repeat": 71},
            3e-6,
        ),
        ("pnoise2", 2, {}, 3e-6),
        (
            "pnoise2",
            2,
            {
                "octaves": 5,
                "persistence": 0.61,
                "lacunarity": 1.83,
                "repeatx": 71.5,
                "repeaty": 83.25,
            },
            4e-6,
        ),
        ("pnoise3", 3, {}, 4e-6),
        (
            "pnoise3",
            3,
            {
                "octaves": 5,
                "persistence": 0.61,
                "lacunarity": 1.83,
                "repeatx": 71,
                "repeaty": 83,
                "repeatz": 97,
            },
            5e-6,
        ),
        ("snoise2", 2, {}, 4e-6),
        (
            "snoise2",
            2,
            {"octaves": 5, "persistence": 0.61, "lacunarity": 1.83, "base": 11.25},
            3e-5,
        ),
        (
            "snoise2",
            2,
            {
                "octaves": 3,
                "persistence": 0.61,
                "lacunarity": 1.83,
                "repeatx": 71,
                "repeaty": 83,
                "base": 2.5,
            },
            3e-5,
        ),
        ("snoise3", 3, {}, 5e-6),
        (
            "snoise3",
            3,
            {"octaves": 5, "persistence": 0.61, "lacunarity": 1.83},
            5e-6,
        ),
        ("snoise4", 4, {}, 5e-6),
        (
            "snoise4",
            4,
            {"octaves": 5, "persistence": 0.61},
            5e-6,
        ),
    ],
)
def test_array_parity(name, dimensions, kwargs, atol):
    rng = np.random.default_rng(42 + dimensions)
    coords = [
        rng.uniform(-25.0, 25.0, size=257).astype(np.float32)
        for _ in range(dimensions)
    ]
    actual = getattr(ours, name)(*coords, **kwargs)
    expected = np.fromiter(
        (
            getattr(reference, name)(
                *(float(coord[i]) for coord in coords), **kwargs
            )
            for i in range(coords[0].size)
        ),
        dtype=np.float32,
        count=coords[0].size,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def test_array_broadcasting_and_aliases():
    x = np.linspace(0.0, 2.0, 7, dtype=np.float32)[:, None]
    y = np.linspace(-1.0, 1.0, 9, dtype=np.float32)[None, :]
    actual = ours.pnoise2_array(x, y, octaves=3)
    assert actual.shape == (7, 9)
    expected = np.array(
        [
            [reference.pnoise2(float(xv), float(yv), octaves=3) for yv in y[0]]
            for xv in x[:, 0]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-6)


@pytest.mark.parametrize(
    ("name", "dimensions"),
    [
        ("pnoise1", 1),
        ("pnoise2", 2),
        ("pnoise3", 3),
        ("snoise2", 2),
        ("snoise3", 3),
        ("snoise4", 4),
    ],
)
def test_array_aliases_and_empty_inputs_do_not_enter_ffi(
    name, dimensions, monkeypatch
):
    alias = getattr(ours, f"{name}_array")
    assert alias is getattr(ours, name)
    monkeypatch.setattr(ours, "lib", lambda: pytest.fail("empty input entered FFI"))
    coords = [np.empty((0, 3), dtype=np.float64) for _ in range(dimensions)]
    actual = alias(*coords)
    assert actual.shape == (0, 3)
    assert actual.dtype == np.float32


@pytest.mark.parametrize(("name", "dimensions"), [("pnoise2", 2), ("snoise4", 4)])
def test_simd_tail(name, dimensions):
    rng = np.random.default_rng(100 + dimensions)
    coords = [
        rng.uniform(-3.0, 3.0, size=19).astype(np.float32)
        for _ in range(dimensions)
    ]
    actual = getattr(ours, name)(*coords, octaves=3)
    expected = np.fromiter(
        (
            getattr(reference, name)(
                *(float(coord[i]) for coord in coords), octaves=3
            )
            for i in range(19)
        ),
        dtype=np.float32,
        count=19,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)


@pytest.mark.parametrize(("name", "dimensions"), [("pnoise2", 2), ("snoise4", 4)])
def test_parallel_threshold(name, dimensions):
    rng = np.random.default_rng(200 + dimensions)
    coords = [
        rng.uniform(-3.0, 3.0, size=16384).astype(np.float32)
        for _ in range(dimensions)
    ]
    actual = getattr(ours, name)(*coords, octaves=2)
    expected = np.concatenate(
        [
            getattr(ours, name)(
                *(coord[start : start + 8192] for coord in coords),
                octaves=2,
            )
            for start in (0, 8192)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_snoise4_gpu_path():
    rng = np.random.default_rng(304)
    coords = [
        rng.uniform(-3.0, 3.0, size=10003).astype(np.float32)
        for _ in range(4)
    ]
    actual = ours.snoise4(*coords, octaves=3, device="gpu")
    expected = ours.snoise4(*coords, octaves=3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)


def test_snoise4_gpu_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(ours, "_gpu_has_headroom", lambda: False)
    coords = [
        np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        for _ in range(4)
    ]
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        ours.snoise4(*coords, octaves=2, device="gpu")


def test_snoise4_invalid_device():
    with pytest.raises(ValueError, match="device"):
        ours.snoise4(0.1, 0.2, 0.3, 0.4, device="other")


def test_snoise4_custom_lacunarity():
    coords = (0.3, 0.6, 0.9, 1.2)
    octaves = 4
    persistence = np.float32(0.61)
    lacunarity = np.float32(1.83)
    actual = ours.snoise4(
        *coords,
        octaves=octaves,
        persistence=float(persistence),
        lacunarity=float(lacunarity),
    )
    frequency = np.float32(1.0)
    amplitude = np.float32(1.0)
    amplitude_sum = np.float32(0.0)
    expected = np.float32(0.0)
    coords32 = tuple(np.float32(value) for value in coords)
    for _ in range(octaves):
        value = reference.snoise4(
            *(float(np.float32(coord * frequency)) for coord in coords32)
        )
        expected = np.float32(expected + np.float32(value) * amplitude)
        amplitude_sum = np.float32(amplitude_sum + amplitude)
        frequency = np.float32(frequency * lacunarity)
        amplitude = np.float32(amplitude * persistence)
    expected = np.float32(expected / amplitude_sum)
    assert actual == pytest.approx(float(expected), abs=5e-6)


def test_repeat_periods():
    x, y, z = 1.125, 2.25, 0.75
    assert ours.pnoise1(x, repeat=17) == pytest.approx(
        ours.pnoise1(x + 17, repeat=17), abs=3e-6
    )
    assert ours.pnoise2(x, y, repeatx=17, repeaty=23) == pytest.approx(
        ours.pnoise2(x + 17, y + 23, repeatx=17, repeaty=23), abs=3e-6
    )
    assert ours.pnoise3(
        x, y, z, repeatx=17, repeaty=23, repeatz=29
    ) == pytest.approx(
        ours.pnoise3(
            x + 17, y + 23, z + 29,
            repeatx=17, repeaty=23, repeatz=29,
        ),
        abs=4e-6,
    )
    assert ours.snoise2(
        x, y, repeatx=17, repeaty=23
    ) == pytest.approx(
        ours.snoise2(x + 17, y + 23, repeatx=17, repeaty=23),
        abs=3e-5,
    )


@pytest.mark.parametrize(
    "name,coords",
    [
        ("pnoise1", (0.1,)),
        ("pnoise2", (0.1, 0.2)),
        ("pnoise3", (0.1, 0.2, 0.3)),
        ("snoise2", (0.1, 0.2)),
        ("snoise3", (0.1, 0.2, 0.3)),
        ("snoise4", (0.1, 0.2, 0.3, 0.4)),
    ],
)
def test_invalid_octaves_match_upstream(name, coords):
    with pytest.raises(ValueError, match="octaves"):
        getattr(ours, name)(*coords, octaves=0)
    with pytest.raises(ValueError, match="octaves"):
        getattr(reference, name)(*coords, octaves=0)


def test_public_signatures():
    expected = {
        "pnoise1": ["x", "octaves", "persistence", "lacunarity", "repeat", "base"],
        "pnoise2": [
            "x", "y", "octaves", "persistence", "lacunarity",
            "repeatx", "repeaty", "base",
        ],
        "pnoise3": [
            "x", "y", "z", "octaves", "persistence", "lacunarity",
            "repeatx", "repeaty", "repeatz", "base",
        ],
        "snoise2": [
            "x", "y", "octaves", "persistence", "lacunarity",
            "repeatx", "repeaty", "base",
        ],
        "snoise3": ["x", "y", "z", "octaves", "persistence", "lacunarity"],
        "snoise4": [
            "x", "y", "z", "w", "octaves", "persistence", "lacunarity",
            "device",
        ],
    }
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(ours, name)).parameters) == parameters
