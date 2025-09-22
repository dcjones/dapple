
from cmap import Color
import numpy as np
from numpy.typing import NDArray
from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Optional
from .coordinates import Serializable

# RGBA encoded colors stored in a [n, 4] array
@dataclass
class Colors(Serializable):
    values: NDArray[np.float64]

    def __len__(self) -> int:
        return self.values.shape[0]

    def isscalar(self) -> bool:
        return self.values.shape[0] == 1

    def assert_scalar(self):
        if not self.isscalar():
            raise ValueError(f"Scalar color expected but found {len(self)} lengths.")

    def scalar_value(self):
        if not self.isscalar():
            raise ValueError(f"Scalar color expected but found {len(self)} lengths.")

    def repeat_scalar(self, n: int) -> 'Colors':
        self.assert_scalar()
        return Colors(np.tile(self.values, (n, 1)))

    def __iter__(self):
        for value in self.values:
            yield Colors(np.array([value]))

    def serialize(self) -> None | str | list[str]:
        if self.isscalar():
            return Color(self.values.squeeze()).hex
        else:
            return [Color(self.values[i,:]).hex for i in range(len(self))]

@singledispatch
def color(value) -> Colors:
    raise NotImplementedError(f"Type {type(value)} can't be converted to colors.")

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
