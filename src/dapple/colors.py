from dataclasses import dataclass
from functools import singledispatch
from typing import Optional, Sequence, cast

import numpy as np
from basic_colormath import (
    get_delta_e_matrix,
    get_delta_e_matrix_lab,
    rgb_to_lab,
    rgbs_to_lab,
)
from cmap import Color, Colormap
from numpy.typing import NDArray

from .coordinates import Serializable


# RGBA encoded colors stored in a [n, 4] array
@dataclass
class Colors(Serializable):
    values: NDArray[np.float32]

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, key) -> "Colors":
        if isinstance(key, (np.ndarray, list)):
            return Colors(self.values[key])
        return Colors(
            self.values[key : key + 1] if isinstance(key, int) else self.values[key]
        )

    def isscalar(self) -> bool:
        return self.values.shape[0] == 1

    def assert_scalar(self):
        if not self.isscalar():
            raise ValueError(f"Scalar color expected but found {len(self)} lengths.")

    def scalar_value(self) -> NDArray[np.float32]:
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

    def is_light(self):
        """
        Classify a color as light or dark based on the relative delte e between white and black.
        """
        de_black_white = get_delta_e_matrix(
            255.0 * self.values[:, 0:3],
            255.0 * np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        )

        return de_black_white[:, 1] <= de_black_white[:, 0]

    def modulate_lightness(self, delta: float = 0.1):
        """
        Lighten or darken the color to produce a color that is similar but with a perceptible tweak.
        """

        lab = rgb_to_oklab(self.values[:, 0:3])
        lab[:, 0] += delta
        lab[self.is_light()] -= delta
        rgb = oklab_to_rgb(lab)
        return Colors(np.concat([rgb.clip(0.0, 1.0), self.values[:, 3:4]], axis=-1))


@singledispatch
def color(value) -> Colors:
    raise NotImplementedError(f"Type {type(value)} can't be converted to colors.")


@color.register(list)
def _(value) -> Colors:
    n = len(value)
    values = np.zeros((n, 4), dtype=np.float32)
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
    return Colors(np.array([[rgba.r, rgba.g, rgba.b, rgba.a]], dtype=np.float32))


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
    colors = np.concatenate(
        [colors / 255.0, np.ones((k, 1), dtype=np.float32)], axis=-1
    )

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


_RGB_TO_XYZ = np.array(
    [
        [0.41245645, 0.35757607, 0.18043749],
        [0.21267284, 0.71515214, 0.072175],
        [0.0193339, 0.11919203, 0.9503041],
    ],
    dtype=np.float32,
)

_RGB_TO_XYZ_INV = np.array(
    [
        [3.240454, -1.5371385, -0.4985314],
        [-0.96926594, 1.8760108, 0.041556],
        [0.05564343, -0.20402591, 1.0572252],
    ],
    dtype=np.float32,
)


def linearize_srgb(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    output = rgb / 12.92
    mask = rgb > 0.04045
    output[mask] = np.pow((rgb[mask] + 0.055) / 1.055, 2.4)
    return output


def linearize_srgb_inv(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    output = rgb * 12.92
    mask = rgb > 0.0031308
    output[mask] = 1.055 * np.pow(rgb[mask], 1 / 2.4) - 0.055
    return output


def rgb_to_xyz(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    return linearize_srgb(rgb) @ _RGB_TO_XYZ.transpose()


def xyz_to_rgb(xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    return linearize_srgb_inv(xyz @ _RGB_TO_XYZ_INV.transpose())


_OKLAB_M1 = np.array(
    [
        [0.818933, 0.36186674, -0.12885971],
        [0.03298454, 0.9293119, 0.03614564],
        [0.0482003, 0.26436627, 0.6338517],
    ],
    dtype=np.float32,
)

_OKLAB_M1_INV = np.array(
    [
        [1.2270138, -0.5578, 0.28125614],
        [-0.04058018, 1.1122569, -0.07167668],
        [-0.07638128, -0.42148197, 1.5861632],
    ],
    dtype=np.float32,
)

_OKLAB_M2 = np.array(
    [
        [0.21045426, 0.7936178, -0.00407205],
        [1.9779985, -2.4285922, 0.4505937],
        [0.02590404, 0.78277177, -0.80867577],
    ],
    dtype=np.float32,
)

_OKLAB_M2_INV = np.array(
    [
        [1.0, 0.39633778, 0.21580376],
        [1.0, -0.10556135, -0.06385417],
        [1.0, -0.08948418, -1.2914855],
    ],
    dtype=np.float32,
)


def xyz_to_oklab(xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    lms = xyz @ _OKLAB_M1.transpose()
    lab = np.cbrt(lms) @ _OKLAB_M2.transpose()
    return lab


def oklab_to_xyz(lab: NDArray[np.float32]) -> NDArray[np.float32]:
    lms = np.power(lab @ _OKLAB_M2_INV.transpose(), 3)
    xyz = lms @ _OKLAB_M1_INV.transpose()
    return xyz


def oklab_to_oklch(oklab: NDArray[np.float32]) -> NDArray[np.float32]:
    oklch = np.empty_like(oklab)
    oklch[:, 0] = oklab[:, 0]
    oklch[:, 1] = np.sqrt(oklab[:, 1] ** 2 + oklab[:, 2] ** 2)
    oklch[:, 2] = np.arctan2(oklab[:, 2], oklab[:, 1]) * 180 / np.pi
    return oklch


def oklch_to_oklab(oklch: NDArray[np.float32]) -> NDArray[np.float32]:
    oklab = np.empty_like(oklch)
    oklab[:, 0] = oklch[:, 0]
    oklab[:, 1] = oklch[:, 1] * np.cos(oklch[:, 2] * np.pi / 180)
    oklab[:, 2] = oklch[:, 1] * np.sin(oklch[:, 2] * np.pi / 180)
    return oklab


def rgb_to_oklab(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    xyz = rgb_to_xyz(rgb)
    return xyz_to_oklab(xyz)


def rgb_to_oklab(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    xyz = rgb_to_xyz(rgb)
    return xyz_to_oklab(xyz)


def oklab_to_rgb(oklab: NDArray[np.float32]) -> NDArray[np.float32]:
    xyz = oklab_to_xyz(oklab)
    return xyz_to_rgb(xyz)
