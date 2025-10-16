from cmap import Color, Colormap
from basic_colormath import rgb_to_lab, rgbs_to_lab, get_delta_e_matrix_lab
import numpy as np
from numpy.typing import NDArray
from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Optional, cast
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

    def scalar_value(self) -> NDArray[np.float64]:
        if not self.isscalar():
            raise ValueError(f"Scalar color expected but found {len(self)} lengths.")

        return self.values[0, :]

    def repeat_scalar(self, n: int) -> "Colors":
        self.assert_scalar()
        return Colors(np.tile(self.values, (n, 1)))

    def __iter__(self):
        for value in self.values:
            yield Colors(np.array([value]))

    def serialize(self) -> None | str | list[str]:
        if self.isscalar():
            return Color(self.values.squeeze()).hex
        else:
            return [Color(self.values[i, :]).hex for i in range(len(self))]


@singledispatch
def color(value) -> Colors:
    raise NotImplementedError(f"Type {type(value)} can't be converted to colors.")


@color.register(list)
def _(value) -> Colors:
    n = len(value)
    values = np.zeros((n, 4), dtype=np.float64)
    for i, v in enumerate(value):
        rgba = Color(v).rgba
        values[i, :] = [rgba.r, rgba.g, rgba.b, rgba.a]
    return Colors(values)


@color.register(str)
def _(value) -> Colors:
    return color(Color(value))


@color.register(Color)
def _(value) -> Colors:
    rgba = value.rgba
    return Colors(np.array([[rgba.r, rgba.g, rgba.b, rgba.a]], dtype=np.float64))


def distinguishable_colors(
    k: int, background: Colors | None = None, n: int = 5000
) -> Colormap:
    """
    Generate k distinguishable colors using a greedy algorithm.
    """

    if background is None:
        background = color("white")

    background_rgb = np.astype(background.scalar_value(), np.float32)
    background_lab = np.asarray(
        rgb_to_lab(tuple(255 * background_rgb[0:3])), dtype=np.float32
    )

    nsteps_per_channel = int(np.ceil(np.pow(n, 1 / 3)))
    step_size = 255.0 / nsteps_per_channel

    steps = np.arange(0.0, 255.0, step_size, dtype=np.float32)
    candidates_rgb = np.reshape(np.stack(np.meshgrid(steps, steps, steps), -1), (-1, 3))
    candidates_lab = rgbs_to_lab(candidates_rgb).astype(np.float32)

    candidate_indexes = distinguishable_colors_from_candidates(
        k, candidates_lab, background_lab
    )

    colors = candidates_rgb[candidate_indexes, :]
    colors = np.concatenate([colors / 255, np.ones((k, 1), dtype=np.float32)], axis=-1)

    return Colormap(colors.astype(np.float64))


# Greedy approximation of the max-min dispersion problem. We want to choose k
# colors that maximize the minimum pairwise distance between any two colors.
def distinguishable_colors_from_candidates(
    k: int, candidates: NDArray[np.float32], seed: NDArray[np.float32]
) -> NDArray[np.int32]:
    n = int(candidates.shape[0])
    colors = np.zeros((k + 1, 3), dtype=np.float32)
    colors[0, :] = seed
    color_indexes = np.zeros(k, np.int32)

    diffs = np.full((k, n), np.inf, dtype=np.float32)

    for i in range(1, k + 1):
        diffs[i - 1, :] = get_delta_e_matrix_lab(candidates, colors[(i - 1) : i, :])[
            :, 0
        ]
        j = cast(int, diffs.min(axis=0).argmax())
        colors[i] = candidates[j, :]
        color_indexes[i - 1] = j

    return color_indexes
