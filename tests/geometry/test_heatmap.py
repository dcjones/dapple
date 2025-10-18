import numpy as np
import pytest

from dapple.geometry.heatmap import heatmap
from dapple.geometry.bars import Bar
from dapple.scales import UnscaledValues, UnscaledBinaryOp
from dapple.coordinates import CtxLengths


class TestHeatmapGeometry:
    """Tests for the heatmap geometry helper."""

    def test_returns_vectorized_rects(self):
        """Heatmap produces a vectorized rect element with expected attributes."""
        data = [[0.1, 0.2], [0.3, 0.4]]

        elem = heatmap(data)

        assert isinstance(elem, Bar)
        assert elem.tag == "dapple:bar"
        assert set(elem.attrib.keys()).issuperset({"x", "y", "width", "height", "fill"})
        assert isinstance(elem.attrib["x"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["y"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["width"], CtxLengths)
        assert isinstance(elem.attrib["height"], CtxLengths)
        assert isinstance(elem.attrib["fill"], UnscaledValues)

    def test_default_positions(self):
        """Default x/y spacing yields unit squares centered on integer grid."""
        data = [[1, 2], [3, 4]]

        elem = heatmap(data)

        x_expr = elem.attrib["x"]
        y_expr = elem.attrib["y"]

        centers_x = np.asarray(x_expr.a.values, dtype=np.float64)
        centers_y = np.asarray(y_expr.a.values, dtype=np.float64)
        offset_x = x_expr.b.values[0]
        offset_y = y_expr.b.values[0]

        np.testing.assert_allclose(
            centers_x - offset_x, [-0.5, 0.5, -0.5, 0.5], atol=1e-9
        )
        np.testing.assert_allclose(
            centers_y - offset_y, [-0.5, -0.5, 0.5, 0.5], atol=1e-9
        )
        np.testing.assert_allclose(elem.attrib["width"].values, [1.0])
        np.testing.assert_allclose(elem.attrib["height"].values, [1.0])

    def test_custom_positions(self):
        """Supplied x/y coordinates are converted into appropriate intervals."""
        data = [[1, 2, 3], [4, 5, 6]]
        x = [0.0, 2.0, 5.0]
        y = [0.0, 10.0]

        elem = heatmap(data, x=x, y=y)

        x_expr = elem.attrib["x"]
        y_expr = elem.attrib["y"]

        centers_x = np.asarray(x_expr.a.values, dtype=np.float64)
        centers_y = np.asarray(y_expr.a.values, dtype=np.float64)
        offset_x = x_expr.b.values[0]
        offset_y = y_expr.b.values[0]

        np.testing.assert_allclose(
            centers_x - offset_x,
            [-0.5, 1.5, 4.5, -0.5, 1.5, 4.5],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            centers_y - offset_y,
            [-0.5, -0.5, -0.5, 9.5, 9.5, 9.5],
            atol=1e-9,
        )
        np.testing.assert_allclose(elem.attrib["width"].values, [1.0])
        np.testing.assert_allclose(elem.attrib["height"].values, [1.0])

    def test_exclude_diagonal(self):
        """Diagonal entries are removed when exclude_diagonal is True."""
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        elem = heatmap(data, exclude_diagonal=True)

        fill_vals = np.asarray(elem.attrib["fill"].values)
        assert fill_vals.shape[0] == 6
        np.testing.assert_array_equal(
            fill_vals, [1, 2, 3, 5, 6, 7]
        )  # Off-diagonal entries only

    def test_numpy_array_input(self):
        """Heatmap accepts numpy arrays directly."""
        data = np.arange(9).reshape(3, 3)

        elem = heatmap(data)

        fill_vals = np.asarray(elem.attrib["fill"].values)
        np.testing.assert_array_equal(fill_vals, data.reshape(-1))

    def test_invalid_color_shape(self):
        """Non-matrix color inputs raise helpful errors."""
        with pytest.raises(ValueError, match="must be a 2D matrix"):
            heatmap([1, 2, 3])

    def test_position_length_validation(self):
        """Mismatched x/y lengths raise ValueError."""
        data = [[1, 2], [3, 4]]

        with pytest.raises(ValueError, match="Expected 2 x positions"):
            heatmap(data, x=[0.0])

        with pytest.raises(ValueError, match="Expected 2 y positions"):
            heatmap(data, y=[0.0])
