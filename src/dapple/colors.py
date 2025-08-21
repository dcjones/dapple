
from cmap import Color
import numpy as np
from numpy.typing import NDArray
from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence


# RGBA encoded colors stored in a [n, 4] array
@dataclass
class Colors:
    values: NDArray[np.float64]

@singledispatch
def color(value) -> Colors:
    raise NotImplementedError(f"Type {type(value)} can't be converted to colors.")

# TODO: Actually the logic for this may be too tricky for singledispatch

@color.register(list)
def _(value) -> Colors:
    n = len(value)
    values = np.zeros((n, 4), dtype=np.float64)
    for i, v in enumerate(value):
        rgba = Color(v).rgba
        values[i,:] = [rgba.r, rgba.g, rgba.b, rgba.a]
    return Colors(values)

@color.register(str)
def _(value) -> Colors:
    return color(Color(value))

@color.register(Color)
def _(value) -> Colors:
    rgba = value.rgba
    return Colors(np.array([[rgba.r, rgba.g, rgba.b, rgba.a]], dtype=np.float64))
