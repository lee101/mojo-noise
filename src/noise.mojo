"""Perlin improved noise and simplex noise kernels compatible with noise 1.2.2."""

from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import abs, floor
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 16384
comptime PARALLEL_GRAIN = 4096
comptime MAX_PARALLEL_WORKERS = 16


def perm(p: BPtr, i: Int) -> Int:
    return Int(p[i])


def lerp(t: Float32, a: Float32, b: Float32) -> Float32:
    return a + t * (b - a)


def fade(x: Float32) -> Float32:
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def fmod32(x: Float32, period: Float32) -> Float32:
    return x - Float32(Int(x / period)) * period


def grad1(hash: Int, x: Float32) -> Float32:
    var g = Float32((hash & 7) + 1)
    if hash & 8:
        g = -1.0
    return g * x


def grad2(hash: Int, x: Float32, y: Float32) -> Float32:
    var h = hash & 15
    if h == 0:
        return x + y
    if h == 1:
        return -x + y
    if h == 2:
        return x - y
    if h == 3:
        return -x - y
    if h == 4 or h == 6 or h == 12:
        return x
    if h == 5 or h == 7 or h == 13:
        return -x
    if h == 8 or h == 10 or h == 15:
        return y
    return -y


def grad3(hash: Int, x: Float32, y: Float32, z: Float32) -> Float32:
    var h = hash & 15
    if h == 0:
        return x + y
    if h == 1:
        return -x + y
    if h == 2:
        return x - y
    if h == 3:
        return -x - y
    if h == 4:
        return x + z
    if h == 5:
        return -x + z
    if h == 6 or h == 12:
        return x - z
    if h == 7 or h == 13:
        return -x - z
    if h == 8 or h == 15:
        return y + z
    if h == 9 or h == 14:
        return -y + z
    if h == 10:
        return y - z
    return -y - z


def grad4(hash: Int, x: Float32, y: Float32, z: Float32, w: Float32) -> Float32:
    var h = hash & 31
    var a: Float32
    var b: Float32
    var c: Float32
    if h < 8:
        a = y if h < 4 else -y
        b = z if (h & 3) < 2 else -z
        c = w if (h & 1) == 0 else -w
    elif h < 16:
        a = x if h < 12 else -x
        b = z if (h & 3) < 2 else -z
        c = w if (h & 1) == 0 else -w
    elif h < 24:
        a = x if h < 20 else -x
        b = y if (h & 3) < 2 else -y
        c = w if (h & 1) == 0 else -w
    else:
        a = x if h < 28 else -x
        b = y if (h & 3) < 2 else -y
        c = z if (h & 1) == 0 else -z
    return a + b + c


def perlin1_one(x_in: Float32, repeat: Int, base: Int, p: BPtr) -> Float32:
    var x = x_in
    var xf = floor(x)
    var cell = Int(xf)
    var i = cell - Int(Float32(cell) / Float32(repeat)) * repeat
    var next_cell = i + 1
    var ii = next_cell - Int(Float32(next_cell) / Float32(repeat)) * repeat
    i = (i & 255) + base
    ii = (ii & 255) + base
    x -= xf
    return lerp(fade(x), grad1(perm(p, i), x), grad1(perm(p, ii), x - 1.0)) * 0.4


def perlin2_one(
    x_in: Float32,
    y_in: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Int,
    p: BPtr,
) -> Float32:
    var x = x_in
    var y = y_in
    var i = Int(floor(fmod32(x, repeatx)))
    var j = Int(floor(fmod32(y, repeaty)))
    var ii = Int(fmod32(Float32(i + 1), repeatx))
    var jj = Int(fmod32(Float32(j + 1), repeaty))
    i = (i & 255) + base
    j = (j & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base
    x -= floor(x)
    y -= floor(y)
    var fx = fade(x)
    var fy = fade(y)
    var a = perm(p, i)
    var aa = perm(p, a + j)
    var ab = perm(p, a + jj)
    var b = perm(p, ii)
    var ba = perm(p, b + j)
    var bb = perm(p, b + jj)
    return lerp(
        fy,
        lerp(fx, grad2(perm(p, aa), x, y), grad2(perm(p, ba), x - 1.0, y)),
        lerp(fx, grad2(perm(p, ab), x, y - 1.0), grad2(perm(p, bb), x - 1.0, y - 1.0)),
    )


def perlin3_one(
    x_in: Float32,
    y_in: Float32,
    z_in: Float32,
    repeatx: Int,
    repeaty: Int,
    repeatz: Int,
    base: Int,
    p: BPtr,
) -> Float32:
    var x = x_in
    var y = y_in
    var z = z_in
    var i = Int(floor(fmod32(x, Float32(repeatx))))
    var j = Int(floor(fmod32(y, Float32(repeaty))))
    var k = Int(floor(fmod32(z, Float32(repeatz))))
    var ii = Int(fmod32(Float32(i + 1), Float32(repeatx)))
    var jj = Int(fmod32(Float32(j + 1), Float32(repeaty)))
    var kk = Int(fmod32(Float32(k + 1), Float32(repeatz)))
    i = (i & 255) + base
    j = (j & 255) + base
    k = (k & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base
    kk = (kk & 255) + base
    x -= floor(x)
    y -= floor(y)
    z -= floor(z)
    var fx = fade(x)
    var fy = fade(y)
    var fz = fade(z)
    var a = perm(p, i)
    var aa = perm(p, a + j)
    var ab = perm(p, a + jj)
    var b = perm(p, ii)
    var ba = perm(p, b + j)
    var bb = perm(p, b + jj)
    return lerp(
        fz,
        lerp(
            fy,
            lerp(fx, grad3(perm(p, aa + k), x, y, z), grad3(perm(p, ba + k), x - 1.0, y, z)),
            lerp(fx, grad3(perm(p, ab + k), x, y - 1.0, z), grad3(perm(p, bb + k), x - 1.0, y - 1.0, z)),
        ),
        lerp(
            fy,
            lerp(fx, grad3(perm(p, aa + kk), x, y, z - 1.0), grad3(perm(p, ba + kk), x - 1.0, y, z - 1.0)),
            lerp(fx, grad3(perm(p, ab + kk), x, y - 1.0, z - 1.0), grad3(perm(p, bb + kk), x - 1.0, y - 1.0, z - 1.0)),
        ),
    )


def simplex2_one(x: Float32, y: Float32, p: BPtr) -> Float32:
    var s = (x + y) * 0.3660254037844386
    var i = floor(x + s)
    var j = floor(y + s)
    var t = (i + j) * 0.21132486540518713
    var x0 = x - (i - t)
    var y0 = y - (j - t)
    var i1 = 1 if x0 > y0 else 0
    var j1 = 1 - i1
    var x1 = x0 - Float32(i1) + 0.21132486540518713
    var y1 = y0 - Float32(j1) + 0.21132486540518713
    var x2 = x0 - 1.0 + 2.0 * 0.21132486540518713
    var y2 = y0 - 1.0 + 2.0 * 0.21132486540518713
    var ii = Int(i) & 255
    var jj = Int(j) & 255
    var g0 = perm(p, ii + perm(p, jj)) % 12
    var g1 = perm(p, ii + i1 + perm(p, jj + j1)) % 12
    var g2 = perm(p, ii + 1 + perm(p, jj + 1)) % 12
    var n0: Float32 = 0.0
    var n1: Float32 = 0.0
    var n2: Float32 = 0.0
    var f0 = 0.5 - x0 * x0 - y0 * y0
    var f1 = 0.5 - x1 * x1 - y1 * y1
    var f2 = 0.5 - x2 * x2 - y2 * y2
    if f0 > 0.0:
        n0 = f0 * f0 * f0 * f0 * grad2(g0, x0, y0)
    if f1 > 0.0:
        n1 = f1 * f1 * f1 * f1 * grad2(g1, x1, y1)
    if f2 > 0.0:
        n2 = f2 * f2 * f2 * f2 * grad2(g2, x2, y2)
    return (n0 + n1 + n2) * 70.0


def simplex3_one(x: Float32, y: Float32, z: Float32, p: BPtr) -> Float32:
    var s = (x + y + z) * (1.0 / 3.0)
    var i = floor(x + s)
    var j = floor(y + s)
    var k = floor(z + s)
    var t = (i + j + k) * (1.0 / 6.0)
    var x0 = x - (i - t)
    var y0 = y - (j - t)
    var z0 = z - (k - t)
    var i1 = 0
    var j1 = 0
    var k1 = 0
    var i2 = 0
    var j2 = 0
    var k2 = 0
    if x0 >= y0:
        if y0 >= z0:
            i1 = 1
            i2 = 1
            j2 = 1
        elif x0 >= z0:
            i1 = 1
            i2 = 1
            k2 = 1
        else:
            k1 = 1
            i2 = 1
            k2 = 1
    else:
        if y0 < z0:
            k1 = 1
            j2 = 1
            k2 = 1
        elif x0 < z0:
            j1 = 1
            j2 = 1
            k2 = 1
        else:
            j1 = 1
            i2 = 1
            j2 = 1
    var x1 = x0 - Float32(i1) + 1.0 / 6.0
    var y1 = y0 - Float32(j1) + 1.0 / 6.0
    var z1 = z0 - Float32(k1) + 1.0 / 6.0
    var x2 = x0 - Float32(i2) + 2.0 / 6.0
    var y2 = y0 - Float32(j2) + 2.0 / 6.0
    var z2 = z0 - Float32(k2) + 2.0 / 6.0
    var x3 = x0 - 1.0 + 3.0 / 6.0
    var y3 = y0 - 1.0 + 3.0 / 6.0
    var z3 = z0 - 1.0 + 3.0 / 6.0
    var ii = Int(i) & 255
    var jj = Int(j) & 255
    var kk = Int(k) & 255
    var g0 = perm(p, ii + perm(p, jj + perm(p, kk))) % 12
    var g1 = perm(p, ii + i1 + perm(p, jj + j1 + perm(p, kk + k1))) % 12
    var g2 = perm(p, ii + i2 + perm(p, jj + j2 + perm(p, kk + k2))) % 12
    var g3 = perm(p, ii + 1 + perm(p, jj + 1 + perm(p, kk + 1))) % 12
    var n0: Float32 = 0.0
    var n1: Float32 = 0.0
    var n2: Float32 = 0.0
    var n3: Float32 = 0.0
    var f0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    var f1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    var f2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    var f3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if f0 > 0.0:
        n0 = f0 * f0 * f0 * f0 * grad3(g0, x0, y0, z0)
    if f1 > 0.0:
        n1 = f1 * f1 * f1 * f1 * grad3(g1, x1, y1, z1)
    if f2 > 0.0:
        n2 = f2 * f2 * f2 * f2 * grad3(g2, x2, y2, z2)
    if f3 > 0.0:
        n3 = f3 * f3 * f3 * f3 * grad3(g3, x3, y3, z3)
    return (n0 + n1 + n2 + n3) * 32.0


def simplex4_one(x: Float32, y: Float32, z: Float32, w: Float32, p: BPtr) -> Float32:
    var s = (x + y + z + w) * 0.30901699437494745
    var i = floor(x + s)
    var j = floor(y + s)
    var k = floor(z + s)
    var l = floor(w + s)
    var t = (i + j + k + l) * 0.1381966011250105
    var x0 = x - (i - t)
    var y0 = y - (j - t)
    var z0 = z - (k - t)
    var w0 = w - (l - t)
    var rx = 0
    var ry = 0
    var rz = 0
    var rw = 0
    if x0 > y0:
        rx += 1
    else:
        ry += 1
    if x0 > z0:
        rx += 1
    else:
        rz += 1
    if x0 > w0:
        rx += 1
    else:
        rw += 1
    if y0 > z0:
        ry += 1
    else:
        rz += 1
    if y0 > w0:
        ry += 1
    else:
        rw += 1
    if z0 > w0:
        rz += 1
    else:
        rw += 1
    var i1 = 1 if rx >= 3 else 0
    var j1 = 1 if ry >= 3 else 0
    var k1 = 1 if rz >= 3 else 0
    var l1 = 1 if rw >= 3 else 0
    var i2 = 1 if rx >= 2 else 0
    var j2 = 1 if ry >= 2 else 0
    var k2 = 1 if rz >= 2 else 0
    var l2 = 1 if rw >= 2 else 0
    var i3 = 1 if rx >= 1 else 0
    var j3 = 1 if ry >= 1 else 0
    var k3 = 1 if rz >= 1 else 0
    var l3 = 1 if rw >= 1 else 0
    var x1 = x0 - Float32(i1) + 0.1381966011250105
    var y1 = y0 - Float32(j1) + 0.1381966011250105
    var z1 = z0 - Float32(k1) + 0.1381966011250105
    var w1 = w0 - Float32(l1) + 0.1381966011250105
    var x2 = x0 - Float32(i2) + 2.0 * 0.1381966011250105
    var y2 = y0 - Float32(j2) + 2.0 * 0.1381966011250105
    var z2 = z0 - Float32(k2) + 2.0 * 0.1381966011250105
    var w2 = w0 - Float32(l2) + 2.0 * 0.1381966011250105
    var x3 = x0 - Float32(i3) + 3.0 * 0.1381966011250105
    var y3 = y0 - Float32(j3) + 3.0 * 0.1381966011250105
    var z3 = z0 - Float32(k3) + 3.0 * 0.1381966011250105
    var w3 = w0 - Float32(l3) + 3.0 * 0.1381966011250105
    var x4 = x0 - 1.0 + 4.0 * 0.1381966011250105
    var y4 = y0 - 1.0 + 4.0 * 0.1381966011250105
    var z4 = z0 - 1.0 + 4.0 * 0.1381966011250105
    var w4 = w0 - 1.0 + 4.0 * 0.1381966011250105
    var ii = Int(i) & 255
    var jj = Int(j) & 255
    var kk = Int(k) & 255
    var ll = Int(l) & 255
    var g0 = perm(p, ii + perm(p, jj + perm(p, kk + perm(p, ll)))) & 31
    var g1 = perm(p, ii + i1 + perm(p, jj + j1 + perm(p, kk + k1 + perm(p, ll + l1)))) & 31
    var g2 = perm(p, ii + i2 + perm(p, jj + j2 + perm(p, kk + k2 + perm(p, ll + l2)))) & 31
    var g3 = perm(p, ii + i3 + perm(p, jj + j3 + perm(p, kk + k3 + perm(p, ll + l3)))) & 31
    var g4 = perm(p, ii + 1 + perm(p, jj + 1 + perm(p, kk + 1 + perm(p, ll + 1)))) & 31
    var n0: Float32 = 0.0
    var n1: Float32 = 0.0
    var n2: Float32 = 0.0
    var n3: Float32 = 0.0
    var n4: Float32 = 0.0
    var f0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
    var f1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
    var f2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
    var f3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
    var f4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4
    if f0 >= 0.0:
        f0 *= f0
        n0 = f0 * f0 * grad4(g0, x0, y0, z0, w0)
    if f1 >= 0.0:
        f1 *= f1
        n1 = f1 * f1 * grad4(g1, x1, y1, z1, w1)
    if f2 >= 0.0:
        f2 *= f2
        n2 = f2 * f2 * grad4(g2, x2, y2, z2, w2)
    if f3 >= 0.0:
        f3 *= f3
        n3 = f3 * f3 * grad4(g3, x3, y3, z3, w3)
    if f4 >= 0.0:
        f4 *= f4
        n4 = f4 * f4 * grad4(g4, x4, y4, z4, w4)
    return 27.0 * (n0 + n1 + n2 + n3 + n4)


def perlin1_fbm(
    x: Float32, octaves: Int, persistence: Float32, lacunarity: Float32,
    repeat: Int, base: Int, p: BPtr,
) -> Float32:
    var freq: Float32 = 1.0
    var amp: Float32 = 1.0
    var amp_sum: Float32 = 0.0
    var total: Float32 = 0.0
    for _ in range(octaves):
        total += perlin1_one(x * freq, Int(Float32(repeat) * freq), base, p) * amp
        amp_sum += amp
        freq *= lacunarity
        amp *= persistence
    return total / amp_sum


def perlin2_fbm(
    x: Float32, y: Float32, octaves: Int, persistence: Float32, lacunarity: Float32,
    repeatx: Float32, repeaty: Float32, base: Int, p: BPtr,
) -> Float32:
    var freq: Float32 = 1.0
    var amp: Float32 = 1.0
    var amp_sum: Float32 = 0.0
    var total: Float32 = 0.0
    for _ in range(octaves):
        total += perlin2_one(x * freq, y * freq, repeatx * freq, repeaty * freq, base, p) * amp
        amp_sum += amp
        freq *= lacunarity
        amp *= persistence
    return total / amp_sum


def perlin3_fbm(
    x: Float32, y: Float32, z: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, repeatx: Int, repeaty: Int,
    repeatz: Int, base: Int, p: BPtr,
) -> Float32:
    var freq: Float32 = 1.0
    var amp: Float32 = 1.0
    var amp_sum: Float32 = 0.0
    var total: Float32 = 0.0
    for _ in range(octaves):
        total += perlin3_one(
            x * freq, y * freq, z * freq, Int(Float32(repeatx) * freq),
            Int(Float32(repeaty) * freq), Int(Float32(repeatz) * freq), base, p,
        ) * amp
        amp_sum += amp
        freq *= lacunarity
        amp *= persistence
    return total / amp_sum


def simplex3_fbm(
    x: Float32, y: Float32, z: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, p: BPtr,
) -> Float32:
    var freq: Float32 = 1.0
    var amp: Float32 = 1.0
    var amp_sum: Float32 = 0.0
    var total: Float32 = 0.0
    for _ in range(octaves):
        total += simplex3_one(x * freq, y * freq, z * freq, p) * amp
        amp_sum += amp
        freq *= lacunarity
        amp *= persistence
    return total / amp_sum


def simplex4_fbm(
    x: Float32, y: Float32, z: Float32, w: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, p: BPtr,
) -> Float32:
    var freq: Float32 = 1.0
    var amp: Float32 = 1.0
    var amp_sum: Float32 = 0.0
    var total: Float32 = 0.0
    for _ in range(octaves):
        total += simplex4_one(x * freq, y * freq, z * freq, w * freq, p) * amp
        amp_sum += amp
        freq *= lacunarity
        amp *= persistence
    return total / amp_sum


def fast_sin(x_in: Float32) -> Float32:
    var x = x_in
    var z = x + 25165824.0
    x -= z - 25165824.0
    var y = x - x * abs(x)
    return y * (3.1 + 3.6 * abs(y))


def simplex2_fbm(
    x_in: Float32, y_in: Float32, octaves: Int, persistence: Float32,
    lacunarity: Float32, has_repeatx: Int, repeatx: Float32,
    has_repeaty: Int, repeaty: Float32, base: Float32, p: BPtr,
) -> Float32:
    var x = x_in
    var y = y_in
    if has_repeatx == 0 and has_repeaty == 0:
        var freq: Float32 = 1.0
        var amp: Float32 = 1.0
        var amp_sum: Float32 = 0.0
        var total: Float32 = 0.0
        for _ in range(octaves):
            total += simplex2_one(x * freq + base, y * freq + base, p) * amp
            amp_sum += amp
            freq *= lacunarity
            amp *= persistence
        return total / amp_sum
    var z = base
    var w = base
    if has_repeaty:
        var yf = y * 2.0 / repeaty
        var yr = repeaty * 0.31830988618379067 * 0.5
        y = fast_sin(yf) * yr
        w += fast_sin(yf + 0.5) * yr
        if has_repeatx == 0:
            return simplex3_fbm(x, y, w, octaves, persistence, lacunarity, p)
    if has_repeatx:
        var xf = x * 2.0 / repeatx
        var xr = repeatx * 0.31830988618379067 * 0.5
        x = fast_sin(xf) * xr
        z += fast_sin(xf + 0.5) * xr
        if has_repeaty == 0:
            return simplex3_fbm(x, y, z, octaves, persistence, lacunarity, p)
    return simplex4_fbm(x, y, z, w, octaves, persistence, lacunarity, p)


def perlin2_range(
    x: FPtr, y: FPtr, result: FPtr, start: Int, end: Int,
    octaves: Int, persistence: Float32, lacunarity: Float32,
    repeatx: Float32, repeaty: Float32, base: Int, p: BPtr,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = start + ((end - start) // W) * W
    for i in range(start, vector_end, W):
        var xv = x.load[width=W](i)
        var yv = y.load[width=W](i)
        var values = SIMD[DType.float32, W]()
        comptime for lane in range(W):
            values[lane] = perlin2_fbm(
                xv[lane], yv[lane], octaves, persistence, lacunarity,
                repeatx, repeaty, base, p,
            )
        result.store(i, values)
    for i in range(vector_end, end):
        result[i] = perlin2_fbm(
            x[i], y[i], octaves, persistence, lacunarity,
            repeatx, repeaty, base, p,
        )


def simplex4_range(
    x: FPtr, y: FPtr, z: FPtr, w: FPtr, result: FPtr,
    start: Int, end: Int, octaves: Int, persistence: Float32,
    lacunarity: Float32, p: BPtr,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = start + ((end - start) // W) * W
    for i in range(start, vector_end, W):
        var xv = x.load[width=W](i)
        var yv = y.load[width=W](i)
        var zv = z.load[width=W](i)
        var wv = w.load[width=W](i)
        var values = SIMD[DType.float32, W]()
        comptime for lane in range(W):
            values[lane] = simplex4_fbm(
                xv[lane], yv[lane], zv[lane], wv[lane],
                octaves, persistence, lacunarity, p,
            )
        result.store(i, values)
    for i in range(vector_end, end):
        result[i] = simplex4_fbm(
            x[i], y[i], z[i], w[i], octaves, persistence, lacunarity, p,
        )


def simplex4_gpu(
    x: FPtr, y: FPtr, z: FPtr, w: FPtr, result: FPtr, n: Int,
    octaves: Int, persistence: Float32, lacunarity: Float32, p: BPtr,
):
    var i = global_idx.x
    if i < n:
        result[i] = simplex4_fbm(
            x[i], y[i], z[i], w[i], octaves, persistence, lacunarity, p,
        )


@export("mn_pnoise1")
def mn_pnoise1(
    x: Float32, octaves: Int, persistence: Float32, lacunarity: Float32,
    repeat: Int, base: Int, perm_addr: Int,
) abi("C") -> Float32:
    return perlin1_fbm(
        x, octaves, persistence, lacunarity, repeat, base,
        BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_pnoise2")
def mn_pnoise2(
    x: Float32, y: Float32, octaves: Int, persistence: Float32,
    lacunarity: Float32, repeatx: Float32, repeaty: Float32,
    base: Int, perm_addr: Int,
) abi("C") -> Float32:
    return perlin2_fbm(
        x, y, octaves, persistence, lacunarity, repeatx, repeaty, base,
        BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_pnoise3")
def mn_pnoise3(
    x: Float32, y: Float32, z: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, repeatx: Int, repeaty: Int,
    repeatz: Int, base: Int, perm_addr: Int,
) abi("C") -> Float32:
    return perlin3_fbm(
        x, y, z, octaves, persistence, lacunarity, repeatx, repeaty, repeatz,
        base, BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_snoise2")
def mn_snoise2(
    x: Float32, y: Float32, octaves: Int, persistence: Float32,
    lacunarity: Float32, has_repeatx: Int, repeatx: Float32,
    has_repeaty: Int, repeaty: Float32, base: Float32, perm_addr: Int,
) abi("C") -> Float32:
    return simplex2_fbm(
        x, y, octaves, persistence, lacunarity, has_repeatx, repeatx,
        has_repeaty, repeaty, base, BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_snoise3")
def mn_snoise3(
    x: Float32, y: Float32, z: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, perm_addr: Int,
) abi("C") -> Float32:
    return simplex3_fbm(
        x, y, z, octaves, persistence, lacunarity,
        BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_snoise4")
def mn_snoise4(
    x: Float32, y: Float32, z: Float32, w: Float32, octaves: Int,
    persistence: Float32, lacunarity: Float32, perm_addr: Int,
) abi("C") -> Float32:
    return simplex4_fbm(
        x, y, z, w, octaves, persistence, lacunarity,
        BPtr(unsafe_from_address=perm_addr),
    )


@export("mn_pnoise1_array")
def mn_pnoise1_array(
    x_addr: Int, result_addr: Int, n: Int, octaves: Int,
    persistence: Float32, lacunarity: Float32, repeat: Int, base: Int,
    perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    for i in range(n):
        result[i] = perlin1_fbm(x[i], octaves, persistence, lacunarity, repeat, base, p)


@export("mn_pnoise2_array")
def mn_pnoise2_array(
    x_addr: Int, y_addr: Int, result_addr: Int, n: Int, octaves: Int,
    persistence: Float32, lacunarity: Float32, repeatx: Float32,
    repeaty: Float32, base: Int, perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    if n < PARALLEL_THRESHOLD or octaves == 1:
        perlin2_range(
            x, y, result, 0, n, octaves, persistence, lacunarity,
            repeatx, repeaty, base, p,
        )
        return
    var num_tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    @always_inline
    def task(task_id: Int) {
        imm x, imm y, imm result, imm n, imm octaves,
        imm persistence, imm lacunarity, imm repeatx, imm repeaty,
        imm base, imm p,
    }:
        var start = task_id * PARALLEL_GRAIN
        var end = min(start + PARALLEL_GRAIN, n)
        perlin2_range(
            x, y, result, start, end, octaves, persistence, lacunarity,
            repeatx, repeaty, base, p,
        )

    parallelize(task, num_tasks, min(num_tasks, MAX_PARALLEL_WORKERS))


@export("mn_pnoise3_array")
def mn_pnoise3_array(
    x_addr: Int, y_addr: Int, z_addr: Int, result_addr: Int, n: Int,
    octaves: Int, persistence: Float32, lacunarity: Float32, repeatx: Int,
    repeaty: Int, repeatz: Int, base: Int, perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var z = FPtr(unsafe_from_address=z_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    for i in range(n):
        result[i] = perlin3_fbm(
            x[i], y[i], z[i], octaves, persistence, lacunarity,
            repeatx, repeaty, repeatz, base, p,
        )


@export("mn_snoise2_array")
def mn_snoise2_array(
    x_addr: Int, y_addr: Int, result_addr: Int, n: Int, octaves: Int,
    persistence: Float32, lacunarity: Float32, has_repeatx: Int,
    repeatx: Float32, has_repeaty: Int, repeaty: Float32,
    base: Float32, perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    for i in range(n):
        result[i] = simplex2_fbm(
            x[i], y[i], octaves, persistence, lacunarity,
            has_repeatx, repeatx, has_repeaty, repeaty, base, p,
        )


@export("mn_snoise3_array")
def mn_snoise3_array(
    x_addr: Int, y_addr: Int, z_addr: Int, result_addr: Int, n: Int,
    octaves: Int, persistence: Float32, lacunarity: Float32, perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var z = FPtr(unsafe_from_address=z_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    for i in range(n):
        result[i] = simplex3_fbm(
            x[i], y[i], z[i], octaves, persistence, lacunarity, p,
        )


@export("mn_snoise4_array")
def mn_snoise4_array(
    x_addr: Int, y_addr: Int, z_addr: Int, w_addr: Int, result_addr: Int,
    n: Int, octaves: Int, persistence: Float32, lacunarity: Float32,
    perm_addr: Int,
) abi("C"):
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var z = FPtr(unsafe_from_address=z_addr)
    var w = FPtr(unsafe_from_address=w_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var p = BPtr(unsafe_from_address=perm_addr)
    if n < PARALLEL_THRESHOLD or octaves == 1:
        simplex4_range(
            x, y, z, w, result, 0, n,
            octaves, persistence, lacunarity, p,
        )
        return
    var num_tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    @always_inline
    def task(task_id: Int) {
        imm x, imm y, imm z, imm w, imm result, imm n, imm octaves,
        imm persistence, imm lacunarity, imm p,
    }:
        var start = task_id * PARALLEL_GRAIN
        var end = min(start + PARALLEL_GRAIN, n)
        simplex4_range(
            x, y, z, w, result, start, end,
            octaves, persistence, lacunarity, p,
        )

    parallelize(task, num_tasks, min(num_tasks, MAX_PARALLEL_WORKERS))


@export("mn_snoise4_array_gpu")
def mn_snoise4_array_gpu(
    x_addr: Int, y_addr: Int, z_addr: Int, w_addr: Int, result_addr: Int,
    n: Int, octaves: Int, persistence: Float32, lacunarity: Float32,
    perm_addr: Int,
) abi("C") -> Int:
    try:
        with DeviceContext() as ctx:
            var x = FPtr(unsafe_from_address=x_addr)
            var y = FPtr(unsafe_from_address=y_addr)
            var z = FPtr(unsafe_from_address=z_addr)
            var w = FPtr(unsafe_from_address=w_addr)
            var result = FPtr(unsafe_from_address=result_addr)
            var p = BPtr(unsafe_from_address=perm_addr)
            var x_device = ctx.enqueue_create_buffer[DType.float32](n)
            var y_device = ctx.enqueue_create_buffer[DType.float32](n)
            var z_device = ctx.enqueue_create_buffer[DType.float32](n)
            var w_device = ctx.enqueue_create_buffer[DType.float32](n)
            var result_device = ctx.enqueue_create_buffer[DType.float32](n)
            var p_device = ctx.enqueue_create_buffer[DType.uint8](1024)
            ctx.enqueue_copy(x_device, x)
            ctx.enqueue_copy(y_device, y)
            ctx.enqueue_copy(z_device, z)
            ctx.enqueue_copy(w_device, w)
            ctx.enqueue_copy(p_device, p)
            comptime block_size = 256
            ctx.enqueue_function[simplex4_gpu](
                x_device, y_device, z_device, w_device, result_device, n,
                octaves, persistence, lacunarity, p_device,
                grid_dim=(n + block_size - 1) // block_size,
                block_dim=block_size,
            )
            ctx.enqueue_copy(result, result_device)
            ctx.synchronize()
        return 1
    except:
        return 0
