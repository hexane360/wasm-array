#!/usr/bin/env python3

from pathlib import Path

import numpy
import matplotlib

colormaps = ('magma', 'inferno', 'cividis', 'viridis', 'plasma')


out_folder = Path(__file__).parent

for colormap in colormaps:
    arr = numpy.array(matplotlib.colormaps[colormap].colors)  # type: ignore
    arr = arr.astype(numpy.float32).reshape(-1, 3)
    print(arr.shape)
    arr.tofile(out_folder / f"{colormap}.raw")