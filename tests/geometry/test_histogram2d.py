from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dapple.geometry.histogram2d import histogram2d
from dapple.geometry.rasterized_heatmap import RasterizedHeatmap


class _SupportsReshape(Protocol):
    def reshape(self, *shape: int) -> NDArray[np.float64]: ...


class _HasValues(Protocol):
    values: _SupportsReshape


class _BinaryOp(Protocol):
    a: _HasValues


class TestHistogram2DGeometry:
    def test_histogram2d_defaults_to_heatmap_counts(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = rng.normal(size=200)

        element = histogram2d(x, y)

        assert element.tag == "dapple:bar"

        expected_counts, _, _ = np.histogram2d(x, y, bins=10)
        fill_attr = cast(_HasValues, element.attrib["fill"])
        raw_values = fill_attr.values
        actual: NDArray[np.float64] = np.asarray(raw_values, dtype=np.float64)

        np.testing.assert_allclose(actual, expected_counts.T.ravel())

    def test_histogram2d_coordinates_align_with_bin_centers(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-3.0, 3.0, size=128)
        y = rng.uniform(-1.5, 1.5, size=128)
        bins = (4, 5)

        element = histogram2d(x, y, bins=bins)

        _, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        expected_x = 0.5 * (x_edges[:-1] + x_edges[1:])
        expected_y = 0.5 * (y_edges[:-1] + y_edges[1:])

        x_expr = cast(_BinaryOp, element.attrib["x"])
        y_expr = cast(_BinaryOp, element.attrib["y"])

        x_matrix: NDArray[np.float64] = np.asarray(
            x_expr.a.values, dtype=np.float64
        ).reshape(bins[1], bins[0])
        y_matrix: NDArray[np.float64] = np.asarray(
            y_expr.a.values, dtype=np.float64
        ).reshape(bins[1], bins[0])

        x_values: NDArray[np.float64] = np.asarray(x_matrix[0], dtype=np.float64)
        y_values: NDArray[np.float64] = np.asarray(y_matrix[:, 0], dtype=np.float64)

        np.testing.assert_allclose(x_values, expected_x)
        np.testing.assert_allclose(y_values, expected_y)

    def test_histogram2d_accepts_bin_tuple(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0.5, 0.6, 0.7, 0.8])
        bins = (2, 3)

        element = histogram2d(x, y, bins=bins)

        expected_counts, _, _ = np.histogram2d(x, y, bins=bins)
        fill_attr = cast(_HasValues, element.attrib["fill"])
        raw_values = fill_attr.values
        actual: NDArray[np.float64] = np.asarray(raw_values, dtype=np.float64)

        assert actual.shape[0] == bins[0] * bins[1]
        np.testing.assert_allclose(
            actual.reshape(bins[1], bins[0]),
            expected_counts.T,
        )

    def test_histogram2d_rasterize_switches_geometry(self):
        x = np.linspace(0.0, 1.0, 25)
        y = np.linspace(2.0, 3.0, 25)

        element = histogram2d(x, y, bins=5, rasterize=True)

        assert isinstance(element, RasterizedHeatmap)

    def test_histogram2d_requires_matching_input_lengths(self):
        with pytest.raises(
            ValueError, match="x and y must contain the same number of elements"
        ):
            _ = histogram2d([0.0, 1.0], [0.5])

    def test_histogram2d_rejects_invalid_bin_sequence_length(self):
        with pytest.raises(
            ValueError, match="bins sequence must contain exactly two integers"
        ):
            _ = histogram2d([0.0, 1.0], [0.0, 1.0], bins=[4])

    def test_histogram2d_rejects_non_integer_bin_counts(self):
        invalid_bins = cast(Sequence[int], (3.5, 2))
        with pytest.raises(TypeError, match="bins sequence must contain integers"):
            _ = histogram2d([0.0, 1.0], [0.0, 1.0], bins=invalid_bins)
