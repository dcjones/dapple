import numpy as np
import pytest

from dapple.coordinates import CtxLengths
from dapple.geometry.bars import Bar
from dapple.geometry.heatmap import Heatmap, heatmap
from dapple.scales import UnscaledBinaryOp, UnscaledValues


class TestHeatmapGeometry:
    """Tests for the heatmap geometry helper."""

    def test_returns_vectorized_rects(self):
        """Heatmap produces a vectorized rect element with expected attributes."""
        data = [[0.1, 0.2], [0.3, 0.4]]

        elem = heatmap(data)

        assert isinstance(elem, Heatmap)
        assert elem.tag == "dapple:heatmap"
        assert set(elem.attrib.keys()).issuperset({"x0", "y0", "x1", "y1", "fill"})
        assert isinstance(elem.attrib["x0"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["y0"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["x1"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["y1"], UnscaledBinaryOp)
        assert isinstance(elem.attrib["fill"], UnscaledValues)

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

        with pytest.raises(
            ValueError,
            match="x arguments must have the same length as the number of columns",
        ):
            heatmap(data, x=[0.0])

        with pytest.raises(
            ValueError,
            match="y arguments must have the same length as the number of rows",
        ):
            heatmap(data, y=[0.0])
