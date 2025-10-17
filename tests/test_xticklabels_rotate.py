"""Tests for XTickLabels rotation feature."""

import pytest
import numpy as np
from dapple.geometry.labels import XTickLabels, xticklabels
from dapple.coordinates import mm
from dapple.scales import ScaleContinuousLength, UnscaledValues, TickCoverage
from dapple.coordinates import CtxLenType
from dapple.config import ChooseTicksParams


def test_xticklabels_rotate_parameter():
    """Test that XTickLabels accepts rotate parameter."""
    # Default (no rotation)
    xtl = XTickLabels()
    assert xtl.attrib["dapple:rotate"] is False

    # With rotation
    xtl_rotated = XTickLabels(rotate=True)
    assert xtl_rotated.attrib["dapple:rotate"] is True

    # Using helper function
    xtl_func = xticklabels(rotate=True)
    assert xtl_func.attrib["dapple:rotate"] is True


def test_xticklabels_rotate_bounds_swap():
    """Test that rotation swaps width and height in abs_bounds."""
    # Create tick labels with known font
    xtl_normal = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=False)
    xtl_rotated = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=True)

    # Create a scale with some data
    tick_params = ChooseTicksParams(
        k_min=2,
        k_max=10,
        k_ideal=5,
        granularity_weight=1 / 4,
        simplicity_weight=1 / 6,
        coverage_weight=1 / 2,
        niceness_weight=1 / 4,
    )
    x_scale = ScaleContinuousLength("x", TickCoverage.StrictSub, tick_params)
    unscaled_values = UnscaledValues(
        "x", np.array([1.0, 2.5, 4.0, 6.25]), CtxLenType.Pos
    )
    x_scale.fit_values(unscaled_values)

    scales = {"x": x_scale}

    # Apply scales to both
    xtl_normal.apply_scales(scales)
    xtl_rotated.apply_scales(scales)

    # Get bounds
    normal_width, normal_height = xtl_normal.abs_bounds()
    rotated_width, rotated_height = xtl_rotated.abs_bounds()

    # When rotated, width and height should be swapped
    assert rotated_width.scalar_value() == pytest.approx(normal_height.scalar_value())
    assert rotated_height.scalar_value() == pytest.approx(normal_width.scalar_value())


def test_xticklabels_rotate_text_anchor():
    """Test that rotation changes text-anchor attribute."""
    xtl_normal = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=False)
    xtl_rotated = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=True)

    # Create a scale
    tick_params = ChooseTicksParams(
        k_min=2,
        k_max=10,
        k_ideal=5,
        granularity_weight=1 / 4,
        simplicity_weight=1 / 6,
        coverage_weight=1 / 2,
        niceness_weight=1 / 4,
    )
    x_scale = ScaleContinuousLength("x", TickCoverage.StrictSub, tick_params)
    unscaled_values = UnscaledValues("x", np.array([1.0, 2.5, 4.0]), CtxLenType.Pos)
    x_scale.fit_values(unscaled_values)

    scales = {"x": x_scale}

    # Apply scales
    xtl_normal.apply_scales(scales)
    xtl_rotated.apply_scales(scales)

    # Check text-anchor in root element
    assert xtl_normal.root is not None
    assert xtl_rotated.root is not None

    assert xtl_normal.root.attrib["text-anchor"] == "middle"
    assert xtl_rotated.root.attrib["text-anchor"] == "end"


def test_xticklabels_rotate_transform_attribute():
    """Test that rotation adds transform attribute to text elements."""
    from dapple.geometry.labels import RotateTransforms

    xtl_normal = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=False)
    xtl_rotated = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=True)

    # Create a scale
    tick_params = ChooseTicksParams(
        k_min=2,
        k_max=10,
        k_ideal=5,
        granularity_weight=1 / 4,
        simplicity_weight=1 / 6,
        coverage_weight=1 / 2,
        niceness_weight=1 / 4,
    )
    x_scale = ScaleContinuousLength("x", TickCoverage.StrictSub, tick_params)
    unscaled_values = UnscaledValues("x", np.array([1.0, 2.5, 4.0]), CtxLenType.Pos)
    x_scale.fit_values(unscaled_values)

    scales = {"x": x_scale}

    # Apply scales
    xtl_normal.apply_scales(scales)
    xtl_rotated.apply_scales(scales)

    # Check that rotated version has transform on child text element
    assert xtl_rotated.root is not None
    assert len(xtl_rotated.root) > 0
    text_element = xtl_rotated.root[0]
    assert "transform" in text_element.attrib
    # The transform should be a RotateTransforms object with -90 degree rotation
    assert isinstance(text_element.attrib["transform"], RotateTransforms)
    assert text_element.attrib["transform"].angle == -90

    # Normal version should not have transform
    assert xtl_normal.root is not None
    assert len(xtl_normal.root) > 0
    normal_text = xtl_normal.root[0]
    assert "transform" not in normal_text.attrib


def test_xticklabels_rotate_update_bounds():
    """Test that update_bounds correctly handles rotation."""
    from dapple.coordinates import CoordBounds

    xtl_rotated = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0), rotate=True)

    # Create a scale
    tick_params = ChooseTicksParams(
        k_min=2,
        k_max=10,
        k_ideal=5,
        granularity_weight=1 / 4,
        simplicity_weight=1 / 6,
        coverage_weight=1 / 2,
        niceness_weight=1 / 4,
    )
    x_scale = ScaleContinuousLength("x", TickCoverage.StrictSub, tick_params)
    unscaled_values = UnscaledValues("x", np.array([1.0, 5.0, 10.0]), CtxLenType.Pos)
    x_scale.fit_values(unscaled_values)

    scales = {"x": x_scale}
    xtl_rotated.apply_scales(scales)

    # Create bounds and update them
    bounds = CoordBounds()
    xtl_rotated.update_bounds(bounds)

    # Should have updated bounds based on rotated text (height becomes the extent in x-direction)
    # Just verify the method runs without error - bounds are stored internally
    assert len(bounds.constraints) > 0
