"""Geometry helpers for two-dimensional histograms."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from .heatmap import heatmap
from .rasterized_heatmap import rasterized_heatmap

__all__ = ["histogram2d"]


def _normalize_bins(bins: int | Sequence[int]) -> tuple[int, int]:
    """
    Normalize the `bins` argument into a pair of positive integers.

    Args:
        bins: Either a single integer applied to both axes or a sequence of two integers.

    Returns:
        A tuple ``(bins_x, bins_y)`` with strictly positive integers.

    Raises:
        TypeError: If bins are not integers or a pair of integers.
        ValueError: If any bin count is non-positive or a sequence has the wrong length.
    """
    if isinstance(bins, Integral):
        bins_x = bins_y = int(bins)
    else:
        if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
            raise TypeError("bins must be an integer or a pair of integers")
        if len(bins) != 2:
            raise ValueError("bins sequence must contain exactly two integers")

        bins_x, bins_y = bins
        if not isinstance(bins_x, Integral) or not isinstance(bins_y, Integral):
            raise TypeError("bins sequence must contain integers")

        bins_x = int(bins_x)
        bins_y = int(bins_y)

    if bins_x <= 0 or bins_y <= 0:
        raise ValueError("bin counts must be positive")

    return bins_x, bins_y


def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: int | Sequence[int] = 10,
    *,
    rasterize: bool = False,
):
    """
    Render a two-dimensional histogram as a heatmap-based geometry element.

    Args:
        x: Sequence of x coordinates for input samples.
        y: Sequence of y coordinates for input samples.
        bins: Number of bins for each axis. Either a single integer applied to both
            dimensions or a pair ``(bins_x, bins_y)``. Defaults to ``10``.
        rasterize: When ``True``, returns a rasterized heatmap for improved performance
            with large bin counts. Defaults to ``False``.

    Returns:
        A heatmap or rasterized heatmap element representing the binned counts.

    Raises:
        ValueError: If ``x`` and ``y`` differ in length.
        TypeError: If ``bins`` is not an integer or a pair of integers.
    """
    x_array = np.asarray(x, dtype=np.float64).ravel()
    y_array = np.asarray(y, dtype=np.float64).ravel()

    if x_array.shape != y_array.shape:
        raise ValueError("x and y must contain the same number of elements")

    bins_x, bins_y = _normalize_bins(bins)

    counts, x_edges, y_edges = np.histogram2d(
        x_array,
        y_array,
        bins=[bins_x, bins_y],
    )

    x0 = x_edges[0:-1]
    x1 = x_edges[1:]

    y0 = y_edges[0:-1]
    y1 = y_edges[1:]

    geometry_fn = rasterized_heatmap if rasterize else heatmap
    return geometry_fn(counts.T, x0=x0, x1=x1, y0=y0, y1=y1)
