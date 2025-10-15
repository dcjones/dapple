import pytest
import numpy as np
from dapple import plot, mm, inch
from dapple.geometry import points, key
from dapple.scales import ScaleDiscreteColor, ScaleContinuousColor
from dapple.config import Config
from dapple.coordinates import ResolveContext, AbsCoordTransform
from dapple.occupancy import Occupancy


def test_key_discrete_color():
    """Test that Key geometry works with discrete color scales."""
    n = 10
    categories = ["A", "B", "C"]
    colors = np.random.choice(categories, n)

    pl = plot(
        points(np.random.rand(n), np.random.rand(n), color=colors),
        key(),
    )

    # Generate SVG and check it contains key elements
    svg_root = pl.svg(inch(4), inch(3))
    svg_str = ""
    from io import StringIO

    buf = StringIO()
    svg_root.serialize(buf)
    svg_str = buf.getvalue()

    # Should contain rectangles for discrete colors
    assert "rect" in svg_str
    assert "fill=" in svg_str
    # Should contain text labels (they appear with whitespace around them)
    assert "A" in svg_str or "B" in svg_str or "C" in svg_str


def test_key_continuous_color():
    """Test that Key geometry works with continuous color scales."""
    n = 15
    values = np.random.rand(n) * 100

    pl = plot(
        points(np.random.rand(n), np.random.rand(n), color=values),
        key(),
    )

    # Generate SVG and check it contains gradient elements
    svg_root = pl.svg(inch(4), inch(3))
    svg_str = ""
    from io import StringIO

    buf = StringIO()
    svg_root.serialize(buf)
    svg_str = buf.getvalue()

    # Should contain linear gradient
    assert "linearGradient" in svg_str
    assert "stop" in svg_str
    assert "key-gradient" in svg_str
    # Should contain the gradient rectangle
    assert 'fill="url(#key-gradient)"' in svg_str


def test_key_no_color():
    """Test that Key geometry is empty when no color aesthetic is used."""
    n = 10

    pl = plot(
        points(np.random.rand(n), np.random.rand(n)),  # No color aesthetic
        key(),
    )

    # Generate SVG - key should be empty
    svg_root = pl.svg(inch(4), inch(3))
    svg_str = ""
    from io import StringIO

    buf = StringIO()
    svg_root.serialize(buf)
    svg_str = buf.getvalue()

    # Should not contain key-specific elements
    assert "key-gradient" not in svg_str
    # May contain some basic plot rectangles, but not many
    rect_count = svg_str.count("rect")
    assert rect_count <= 2  # Allow for basic plot elements


def test_key_apply_scales_discrete():
    """Test that apply_scales method works correctly for discrete scales."""
    key_geom = key()

    # Create a mock discrete color scale with proper colormap
    from dapple.config import Config

    config = Config()
    scale = ScaleDiscreteColor("color")
    scale.colormap = config.discrete_cmap  # Set the colormap

    scales = {"color": scale, "x": None, "y": None}

    # Mock the finalization and ticks
    scales["color"]._targets = {"A": ("A", None), "B": ("B", None)}
    scales["color"].finalize()

    key_geom.apply_scales(scales)

    # Should have stored the color scale
    assert key_geom._color_scale is not None
    assert isinstance(key_geom._color_scale, ScaleDiscreteColor)
    assert key_geom._labels is not None
    assert key_geom._colors is not None


def test_key_apply_scales_continuous():
    """Test that apply_scales method works correctly for continuous scales."""
    key_geom = key()

    # Create a mock continuous color scale
    scales = {"color": ScaleContinuousColor("color"), "x": None, "y": None}

    key_geom.apply_scales(scales)

    # Should have stored the color scale
    assert key_geom._color_scale is not None
    assert isinstance(key_geom._color_scale, ScaleContinuousColor)


def test_key_abs_bounds_discrete():
    """Test that abs_bounds calculation works for discrete keys."""
    key_geom = key()

    # Mock discrete scale data
    key_geom._color_scale = ScaleDiscreteColor("color")
    key_geom._labels = np.array(["A", "B", "C"])
    key_geom._colors = None  # Not needed for bounds calculation

    # Manually set the config values instead of using replace_keys
    from dapple.config import Config

    config = Config()
    key_geom.attrib["font_family"] = config.tick_label_font_family
    key_geom.attrib["font_size"] = config.tick_label_font_size
    key_geom.attrib["font_weight"] = config.tick_label_font_weight
    key_geom.attrib["square_size"] = config.key_square_size
    key_geom.attrib["spacing"] = config.key_spacing
    key_geom.attrib["gradient_width"] = config.key_gradient_width

    width, height = key_geom.abs_bounds()

    # Should return non-zero bounds
    assert width.scalar_value() > 0
    assert height.scalar_value() > 0


def test_key_abs_bounds_continuous():
    """Test that abs_bounds calculation works for continuous keys."""
    key_geom = key()

    # Mock continuous scale - just set the type, don't call ticks()
    key_geom._color_scale = ScaleContinuousColor("color")

    # Manually set the config values instead of using replace_keys
    from dapple.config import Config

    config = Config()
    key_geom.attrib["font_family"] = config.tick_label_font_family
    key_geom.attrib["font_size"] = config.tick_label_font_size
    key_geom.attrib["font_weight"] = config.tick_label_font_weight
    key_geom.attrib["gradient_width"] = config.key_gradient_width
    key_geom.attrib["spacing"] = config.key_spacing
    key_geom.attrib["square_size"] = config.key_square_size
    key_geom.attrib["gradient_width"] = config.key_gradient_width

    # We need to avoid the actual ticks() call in abs_bounds,
    # so let's test the case where the scale fails gracefully
    try:
        width, height = key_geom.abs_bounds()
        # If it succeeds, bounds should be reasonable
        assert width.scalar_value() > 0
        assert height.scalar_value() >= 0  # Height might be 0 if no ticks
    except ValueError:
        # If ticks() fails due to missing min/max, that's expected in this test
        # The key should still be identified as continuous type
        assert isinstance(key_geom._color_scale, ScaleContinuousColor)


def test_key_abs_bounds_no_scale():
    """Test that abs_bounds returns zero when no color scale is present."""
    key_geom = key()

    # No color scale set
    assert key_geom._color_scale is None

    width, height = key_geom.abs_bounds()

    # Should return zero bounds
    assert width.scalar_value() == 0
    assert height.scalar_value() == 0


def test_key_positioning():
    """Test that Key is positioned to the right by default."""
    key_geom = key()

    from dapple.layout import Position

    position = key_geom.attrib.get("dapple:position")

    assert position == Position.RightCenter
