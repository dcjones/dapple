import pytest
import numpy as np
from dapple.geometry.bars import vertical_bars, horizontal_bars, Bar
from dapple.coordinates import (
    cxv,
    cyv,
    mm,
    CoordSet,
    ResolveContext,
    AbsLengths,
)
from dapple.scales import ScaleSet
from dapple.occupancy import Occupancy
from dapple.config import ConfigKey


class TestVerticalBars:
    """Test vertical_bars function."""

    def test_vertical_bars_with_x_centers(self):
        """Test vertical bars with centered x positions."""
        x = [1, 2, 3]
        y = [10, 20, 15]

        bars = vertical_bars(x=x, y=y)

        assert isinstance(bars, Bar)
        assert bars.tag == "dapple:bar"
        assert "x" in bars.attrib
        assert "y" in bars.attrib
        assert "width" in bars.attrib
        assert "height" in bars.attrib

    def test_vertical_bars_with_xmin_xmax(self):
        """Test vertical bars with explicit x ranges."""
        y = [10, 20, 15]
        xmin = [0.5, 1.5, 2.5]
        xmax = [1.5, 2.5, 3.5]

        bars = vertical_bars(y=y, xmin=xmin, xmax=xmax)

        assert isinstance(bars, Bar)
        assert bars.tag == "dapple:bar"

    def test_vertical_bars_with_color(self):
        """Test vertical bars with explicit color."""
        x = [1, 2, 3]
        y = [10, 20, 15]
        color = ["red", "blue", "green"]

        bars = vertical_bars(x=x, y=y, color=color)

        assert "fill" in bars.attrib
        # Should have color parameters
        assert bars.attrib["fill"] is not None

    def test_vertical_bars_default_color(self):
        """Test that default color is barcolor ConfigKey."""
        x = [1, 2, 3]
        y = [10, 20, 15]

        bars = vertical_bars(x=x, y=y)

        assert isinstance(bars.attrib["fill"], ConfigKey)
        assert bars.attrib["fill"].key == "barcolor"

    def test_vertical_bars_missing_y_raises(self):
        """Test that missing y parameter raises ValueError."""
        with pytest.raises(ValueError, match="y parameter is required"):
            vertical_bars(x=[1, 2, 3])

    def test_vertical_bars_both_x_and_xmin_xmax_raises(self):
        """Test that specifying both x and xmin/xmax raises ValueError."""
        with pytest.raises(ValueError, match="Specify either x or"):
            vertical_bars(
                x=[1, 2, 3], y=[10, 20, 15], xmin=[0.5, 1.5, 2.5], xmax=[1.5, 2.5, 3.5]
            )

    def test_vertical_bars_neither_x_nor_xmin_xmax_raises(self):
        """Test that specifying neither x nor xmin/xmax raises ValueError."""
        with pytest.raises(ValueError, match="Must specify either x or both"):
            vertical_bars(y=[10, 20, 15])

    def test_vertical_bars_negative_heights(self):
        """Test vertical bars with negative heights."""
        x = [1, 2, 3]
        y = [-10, 20, -5]

        bars = vertical_bars(x=x, y=y)

        # Should create bars, negative heights will be handled during resolve
        assert isinstance(bars, Bar)


class TestHorizontalBars:
    """Test horizontal_bars function."""

    def test_horizontal_bars_with_y_centers(self):
        """Test horizontal bars with centered y positions."""
        y = [1, 2, 3]
        x = [10, 20, 15]

        bars = horizontal_bars(y=y, x=x)

        assert isinstance(bars, Bar)
        assert bars.tag == "dapple:bar"
        assert "x" in bars.attrib
        assert "y" in bars.attrib
        assert "width" in bars.attrib
        assert "height" in bars.attrib

    def test_horizontal_bars_with_ymin_ymax(self):
        """Test horizontal bars with explicit y ranges."""
        x = [10, 20, 15]
        ymin = [0.5, 1.5, 2.5]
        ymax = [1.5, 2.5, 3.5]

        bars = horizontal_bars(x=x, ymin=ymin, ymax=ymax)

        assert isinstance(bars, Bar)
        assert bars.tag == "dapple:bar"

    def test_horizontal_bars_with_color(self):
        """Test horizontal bars with explicit color."""
        y = [1, 2, 3]
        x = [10, 20, 15]
        color = ["red", "blue", "green"]

        bars = horizontal_bars(y=y, x=x, color=color)

        assert "fill" in bars.attrib
        assert bars.attrib["fill"] is not None

    def test_horizontal_bars_default_color(self):
        """Test that default color is barcolor ConfigKey."""
        y = [1, 2, 3]
        x = [10, 20, 15]

        bars = horizontal_bars(y=y, x=x)

        assert isinstance(bars.attrib["fill"], ConfigKey)
        assert bars.attrib["fill"].key == "barcolor"

    def test_horizontal_bars_missing_x_raises(self):
        """Test that missing x parameter raises ValueError."""
        with pytest.raises(ValueError, match="x parameter is required"):
            horizontal_bars(y=[1, 2, 3])

    def test_horizontal_bars_both_y_and_ymin_ymax_raises(self):
        """Test that specifying both y and ymin/ymax raises ValueError."""
        with pytest.raises(ValueError, match="Specify either y or"):
            horizontal_bars(
                y=[1, 2, 3], x=[10, 20, 15], ymin=[0.5, 1.5, 2.5], ymax=[1.5, 2.5, 3.5]
            )

    def test_horizontal_bars_neither_y_nor_ymin_ymax_raises(self):
        """Test that specifying neither y nor ymin/ymax raises ValueError."""
        with pytest.raises(ValueError, match="Must specify either y or both"):
            horizontal_bars(x=[10, 20, 15])

    def test_horizontal_bars_negative_widths(self):
        """Test horizontal bars with negative widths."""
        y = [1, 2, 3]
        x = [-10, 20, -5]

        bars = horizontal_bars(y=y, x=x)

        # Should create bars, negative widths will be handled during resolve
        assert isinstance(bars, Bar)


class TestBarElement:
    """Test Bar element basic structure."""

    def test_bar_element_creation(self):
        """Test that Bar element can be created with attributes."""
        bar = Bar(x=mm(10), y=mm(20), width=mm(5), height=mm(30), fill="red")

        assert bar.tag == "dapple:bar"
        assert bar.attrib["x"].scalar_value() == 10.0
        assert bar.attrib["y"].scalar_value() == 20.0
        assert bar.attrib["width"].scalar_value() == 5.0
        assert bar.attrib["height"].scalar_value() == 30.0
        assert bar.attrib["fill"] == "red"

    def test_bar_element_stores_negative_dimensions(self):
        """Test that Bar element accepts negative dimensions."""
        bar = Bar(x=mm(10), y=mm(20), width=mm(-5), height=mm(-30), fill="blue")

        # Should store the values as given (resolution will handle corrections)
        assert bar.attrib["width"].scalar_value() == -5.0
        assert bar.attrib["height"].scalar_value() == -30.0
