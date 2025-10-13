import pytest
from dapple.geometry.ticks import XTicks, YTicks, xticks, yticks
from dapple.coordinates import ResolveContext, mm
from dapple.config import Config
from dapple.layout import Position


def test_xticks_creation():
    """Test that XTicks can be created with default parameters."""
    x_ticks = XTicks()
    assert x_ticks.tag == "dapple:xticks"
    assert "dapple:position" in x_ticks.attrib


def test_yticks_creation():
    """Test that YTicks can be created with default parameters."""
    y_ticks = YTicks()
    assert y_ticks.tag == "dapple:yticks"
    assert "dapple:position" in y_ticks.attrib


def test_xticks_function():
    """Test that the xticks function creates an XTicks instance."""
    x_ticks = xticks()
    assert isinstance(x_ticks, XTicks)
    assert x_ticks.tag == "dapple:xticks"


def test_yticks_function():
    """Test that the yticks function creates a YTicks instance."""
    y_ticks = yticks()
    assert isinstance(y_ticks, YTicks)
    assert y_ticks.tag == "dapple:yticks"


def test_xticks_custom_parameters():
    """Test that XTicks accepts custom styling parameters."""
    x_ticks = XTicks(stroke="#ff0000", stroke_width=mm(1.0), tick_length=mm(3.0))
    assert x_ticks.attrib["stroke"] == "#ff0000"
    assert x_ticks.attrib["stroke-width"] == mm(1.0)
    assert x_ticks.attrib["tick_length"] == mm(3.0)


def test_yticks_custom_parameters():
    """Test that YTicks accepts custom styling parameters."""
    y_ticks = YTicks(stroke="#00ff00", stroke_width=mm(0.5), tick_length=mm(4.0))
    assert y_ticks.attrib["stroke"] == "#00ff00"
    assert y_ticks.attrib["stroke-width"] == mm(0.5)
    assert y_ticks.attrib["tick_length"] == mm(4.0)


def test_xticks_integration():
    """Test that XTicks works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and x ticks
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xticks())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_yticks_integration():
    """Test that YTicks works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and y ticks
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), yticks())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_both_ticks_integration():
    """Test that both XTicks and YTicks can be used together."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and both ticks
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xticks(), yticks())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"
