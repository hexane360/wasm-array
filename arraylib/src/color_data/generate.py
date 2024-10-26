#!/usr/bin/env python3

from pathlib import Path

import numpy
import matplotlib

out_folder = Path(__file__).parent

def output_cmap(name: str, data: numpy.ndarray):
    data = data.astype(numpy.float32).reshape(256, 3)
    data.tofile(out_folder / f"{name}.raw")


named_colormaps = ('magma', 'inferno', 'cividis', 'viridis', 'plasma')

for colormap in named_colormaps:
    arr = numpy.array(matplotlib.colormaps[colormap].colors)  # type: ignore
    output_cmap(colormap, arr)


ts = numpy.linspace(0., 1., 256, endpoint=True)

def sinebow(t: numpy.ndarray) -> numpy.ndarray:
    t = 0.5 - t
    return numpy.stack([
        numpy.sin(numpy.pi * (t + 0/3.))**2,
        numpy.sin(numpy.pi * (t + 1/3.))**2,
        numpy.sin(numpy.pi * (t + 2/3.))**2,
    ], axis=-1)

def hsv_to_rgb(h: numpy.ndarray, s: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray:
    h, s, v = numpy.broadcast_arrays(h, s, v)
    r, g, b = numpy.empty_like(h), numpy.empty_like(h), numpy.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    return numpy.stack([r, g, b], axis=-1)


output_cmap('sinebow', sinebow(ts))
output_cmap('hue', hsv_to_rgb(ts, numpy.array(1.0), numpy.array(1.0)))