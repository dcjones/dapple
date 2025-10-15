import pytest
from dapple.geometry.labels import (
    XLabel,
    YLabel,
    Title,
    XTickLabels,
    YTickLabels,
    xlabel,
    ylabel,
    title,
    xticklabels,
    yticklabels,
)
from dapple.coordinates import ResolveContext, mm
from dapple.config import Config
from dapple.layout import Position


def test_xlabel_creation():
    """Test that XLabel can be created with required text parameter."""
    x_label = XLabel("X Axis")
    assert x_label.tag == "dapple:xlabel"
    assert x_label.attrib["text"] == "X Axis"
    assert "dapple:position" in x_label.attrib
    assert x_label.attrib["dapple:position"] == Position.BottomCenter


def test_ylabel_creation():
    """Test that YLabel can be created with required text parameter."""
    y_label = YLabel("Y Axis")
    assert y_label.tag == "dapple:ylabel"
    assert y_label.attrib["text"] == "Y Axis"
    assert "dapple:position" in y_label.attrib
    assert y_label.attrib["dapple:position"] == Position.LeftCenter


def test_title_creation():
    """Test that Title can be created with required text parameter."""
    plot_title = Title("Plot Title")
    assert plot_title.tag == "dapple:title"
    assert plot_title.attrib["text"] == "Plot Title"
    assert "dapple:position" in plot_title.attrib
    assert plot_title.attrib["dapple:position"] == Position.TopCenter


def test_xlabel_function():
    """Test that the xlabel function creates an XLabel instance."""
    x_label = xlabel("X Axis")
    assert isinstance(x_label, XLabel)
    assert x_label.tag == "dapple:xlabel"
    assert x_label.attrib["text"] == "X Axis"


def test_ylabel_function():
    """Test that the ylabel function creates a YLabel instance."""
    y_label = ylabel("Y Axis")
    assert isinstance(y_label, YLabel)
    assert y_label.tag == "dapple:ylabel"
    assert y_label.attrib["text"] == "Y Axis"


def test_title_function():
    """Test that the title function creates a Title instance."""
    plot_title = title("Plot Title")
    assert isinstance(plot_title, Title)
    assert plot_title.tag == "dapple:title"
    assert plot_title.attrib["text"] == "Plot Title"


def test_xlabel_custom_parameters():
    """Test that XLabel accepts custom styling parameters."""
    x_label = XLabel(
        "Custom X Label",
        font_family="Arial",
        font_size=mm(4.0),
        font_weight="bold",
        fill="#ff0000",
    )
    assert x_label.attrib["text"] == "Custom X Label"
    assert x_label.attrib["font_family"] == "Arial"
    assert x_label.attrib["font_size"] == mm(4.0)
    assert x_label.attrib["font_weight"] == "bold"
    assert x_label.attrib["fill"] == "#ff0000"


def test_ylabel_custom_parameters():
    """Test that YLabel accepts custom styling parameters."""
    y_label = YLabel(
        "Custom Y Label",
        font_family="Times",
        font_size=mm(5.0),
        font_weight="300",
        fill="#00ff00",
    )
    assert y_label.attrib["text"] == "Custom Y Label"
    assert y_label.attrib["font_family"] == "Times"
    assert y_label.attrib["font_size"] == mm(5.0)
    assert y_label.attrib["font_weight"] == "300"
    assert y_label.attrib["fill"] == "#00ff00"


def test_title_custom_parameters():
    """Test that Title accepts custom styling parameters."""
    plot_title = Title(
        "Custom Title",
        font_family="Helvetica",
        font_size=mm(6.0),
        font_weight="600",
        fill="#0000ff",
    )
    assert plot_title.attrib["text"] == "Custom Title"
    assert plot_title.attrib["font_family"] == "Helvetica"
    assert plot_title.attrib["font_size"] == mm(6.0)
    assert plot_title.attrib["font_weight"] == "600"
    assert plot_title.attrib["fill"] == "#0000ff"


def test_xlabel_abs_bounds():
    """Test that XLabel computes reasonable absolute bounds."""
    x_label = XLabel(
        "Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000"
    )
    width_bound, height_bound = x_label.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_ylabel_abs_bounds():
    """Test that YLabel computes reasonable absolute bounds."""
    y_label = YLabel(
        "Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000"
    )
    width_bound, height_bound = y_label.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_title_abs_bounds():
    """Test that Title computes reasonable absolute bounds."""
    plot_title = YLabel(
        "Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000"
    )
    width_bound, height_bound = plot_title.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_different_text_lengths_affect_bounds():
    """Test that longer text results in different bounds."""
    short_label = XLabel("X", font_family="DejaVu Sans", font_size=mm(4.0))
    long_label = XLabel(
        "Very Long X Axis Label", font_family="DejaVu Sans", font_size=mm(4.0)
    )

    short_width, short_height = short_label.abs_bounds()
    long_width, long_height = long_label.abs_bounds()

    # Width bounds should be the same (both are 0 for xlabel)
    assert short_width.scalar_value() < long_width.scalar_value()
    # Heights might be similar (same font size), but both should be positive
    assert short_height.scalar_value() > 0.0
    assert long_height.scalar_value() > 0.0


def test_ylabel_different_text_lengths_affect_bounds():
    """Test that longer text results in different bounds for YLabel."""
    short_label = YLabel("Y", font_family="DejaVu Sans", font_size=mm(4.0))
    long_label = YLabel(
        "Very Long Y Axis Label", font_family="DejaVu Sans", font_size=mm(4.0)
    )

    short_width, short_height = short_label.abs_bounds()
    long_width, long_height = long_label.abs_bounds()

    # Both should have height bound of 0
    assert short_height.scalar_value() < long_height.scalar_value()

    # Width bounds should be positive and similar (both use text height as width when rotated)
    assert short_width.scalar_value() > 0.0
    assert long_width.scalar_value() > 0.0


def test_font_size_affects_bounds():
    """Test that different font sizes affect the bounds."""
    small_label = XLabel("Test", font_family="DejaVu Sans", font_size=mm(2.0))
    large_label = XLabel("Test", font_family="DejaVu Sans", font_size=mm(8.0))

    small_width, small_height = small_label.abs_bounds()
    large_width, large_height = large_label.abs_bounds()

    assert small_width.scalar_value() < large_width.scalar_value()
    assert large_height.scalar_value() > small_height.scalar_value()


def test_xlabel_integration():
    """Test that XLabel works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and x label
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xlabel("X Axis Label"))

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_ylabel_integration():
    """Test that YLabel works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and y label
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), ylabel("Y Axis Label"))

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_title_integration():
    """Test that Title works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and title
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), title("Plot Title"))

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_all_labels_integration():
    """Test that all labels can be used together."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and all labels
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(
        points(x_data, y_data),
        title("Complete Plot"),
        xlabel("X Values"),
        ylabel("Y Values"),
    )

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_empty_text_handling():
    """Test that empty text is handled gracefully."""
    empty_label = XLabel("", "DejaVu Sans", font_size=mm(4.0))
    width_bound, height_bound = empty_label.abs_bounds()

    # Should not crash and should return reasonable bounds
    assert width_bound.scalar_value() >= 0.0
    assert height_bound.scalar_value() >= 0.0


def test_unicode_text_handling():
    """Test that Unicode text is handled correctly."""
    unicode_label = XLabel(
        "Åxis Lábel with ñ and 中文", "DejaVu Sans", font_size=mm(4.0)
    )
    width_bound, height_bound = unicode_label.abs_bounds()

    # Should not crash and should return positive bounds
    assert width_bound.scalar_value() >= 0.0
    assert height_bound.scalar_value() > 0.0


def test_xticklabels_creation():
    """Test that XTickLabels can be created with default parameters."""
    x_tick_labels = XTickLabels()
    assert x_tick_labels.tag == "dapple:xticklabels"
    assert "dapple:position" in x_tick_labels.attrib
    assert x_tick_labels.attrib["dapple:position"] == Position.BottomLeft


def test_yticklabels_creation():
    """Test that YTickLabels can be created with default parameters."""
    y_tick_labels = YTickLabels()
    assert y_tick_labels.tag == "dapple:yticklabels"
    assert "dapple:position" in y_tick_labels.attrib
    assert y_tick_labels.attrib["dapple:position"] == Position.LeftTop


def test_xticklabels_function():
    """Test that the xticklabels function creates an XTickLabels instance."""
    x_tick_labels = xticklabels()
    assert isinstance(x_tick_labels, XTickLabels)
    assert x_tick_labels.tag == "dapple:xticklabels"


def test_yticklabels_function():
    """Test that the yticklabels function creates a YTickLabels instance."""
    y_tick_labels = yticklabels()
    assert isinstance(y_tick_labels, YTickLabels)
    assert y_tick_labels.tag == "dapple:yticklabels"


def test_xticklabels_custom_parameters():
    """Test that XTickLabels accepts custom styling parameters."""
    x_tick_labels = XTickLabels(
        font_family="Arial", font_size=mm(3.0), font_weight="bold", fill="#ff0000"
    )
    assert x_tick_labels.attrib["font_family"] == "Arial"
    assert x_tick_labels.attrib["font_size"] == mm(3.0)
    assert x_tick_labels.attrib["font_weight"] == "bold"
    assert x_tick_labels.attrib["fill"] == "#ff0000"


def test_yticklabels_custom_parameters():
    """Test that YTickLabels accepts custom styling parameters."""
    y_tick_labels = YTickLabels(
        font_family="Times", font_size=mm(2.0), font_weight="500", fill="#00ff00"
    )
    assert y_tick_labels.attrib["font_family"] == "Times"
    assert y_tick_labels.attrib["font_size"] == mm(2.0)
    assert y_tick_labels.attrib["font_weight"] == "500"
    assert y_tick_labels.attrib["fill"] == "#00ff00"


def test_xticklabels_abs_bounds():
    """Test that XTickLabels computes reasonable absolute bounds."""
    x_tick_labels = XTickLabels(
        font_family="DejaVu Sans", font_size=mm(3.0), fill="#333333"
    )
    width_bound, height_bound = x_tick_labels.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_yticklabels_abs_bounds():
    """Test that YTickLabels computes reasonable absolute bounds."""
    y_tick_labels = YTickLabels(
        font_family="DejaVu Sans", font_size=mm(3.0), fill="#333333"
    )
    width_bound, height_bound = y_tick_labels.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_xticklabels_integration():
    """Test that XTickLabels works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and x tick labels
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xticklabels())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_yticklabels_integration():
    """Test that YTickLabels works in a complete plotting context."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and y tick labels
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), yticklabels())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_both_ticklabels_integration():
    """Test that both XTickLabels and YTickLabels can be used together."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create a simple plot with data and both tick labels
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xticklabels(), yticklabels())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_ticklabels_with_ticks_integration():
    """Test that tick labels work well with tick marks."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points
    from dapple.geometry.ticks import xticks, yticks

    # Create a simple plot with data, ticks, and tick labels
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(points(x_data, y_data), xticks(), yticks(), xticklabels(), yticklabels())

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_complete_plot_with_all_elements():
    """Test a complete plot with all label types and tick elements."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points
    from dapple.geometry.ticks import xticks, yticks

    # Create a comprehensive plot with all elements
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 1, 5, 3])

    pl = plot(
        points(x_data, y_data),
        title("Complete Plot with All Labels"),
        xlabel("X Values"),
        ylabel("Y Values"),
        xticks(),
        yticks(),
        xticklabels(),
        yticklabels(),
    )

    # Should be able to generate SVG without errors
    svg_root = pl.svg(inch(4), inch(3))
    assert svg_root is not None
    assert svg_root.tag == "svg"


def test_ticklabels_font_size_affects_bounds():
    """Test that different font sizes affect the bounds of tick labels."""
    small_labels = XTickLabels(font_family="DejaVu Sans", font_size=mm(1.0))
    large_labels = XTickLabels(font_family="DejaVu Sans", font_size=mm(5.0))

    small_width, small_height = small_labels.abs_bounds()
    large_width, large_height = large_labels.abs_bounds()

    assert small_width.scalar_value() < large_width.scalar_value()
    assert small_height.scalar_value() < large_height.scalar_value()


def test_xticklabels_apply_scales():
    """Test that XTickLabels.apply_scales stores tick labels for precise bounds."""
    import numpy as np
    from dapple.scales import ScaleContinuousLength, UnscaledValues, TickCoverage
    from dapple.coordinates import CtxLenType
    from dapple.config import ChooseTicksParams

    # Create XTickLabels instance
    x_tick_labels = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0))

    # Initially, no tick labels are stored
    assert x_tick_labels.tick_labels is None

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

    # Create a mock ScaleSet
    scales = {"x": x_scale}

    # Apply scales
    x_tick_labels.apply_scales(scales)

    # Now tick labels should be stored
    assert x_tick_labels.tick_labels is not None
    assert len(x_tick_labels.tick_labels) > 0

    # Check that bounds calculation uses the actual labels
    width_with_labels, height_with_labels = x_tick_labels.abs_bounds()

    # Create a new instance without applied scales for comparison
    x_tick_labels_no_scales = XTickLabels(font_family="DejaVu Sans", font_size=mm(3.0))
    width_without_labels, height_without_labels = x_tick_labels_no_scales.abs_bounds()

    # The bounds might be different since one uses actual labels, other uses estimate
    assert width_with_labels.scalar_value() > 0.0
    assert height_with_labels.scalar_value() > 0.0


def test_yticklabels_apply_scales():
    """Test that YTickLabels.apply_scales stores tick labels for precise bounds."""
    import numpy as np
    from dapple.scales import ScaleContinuousLength, UnscaledValues, TickCoverage
    from dapple.coordinates import CtxLenType
    from dapple.config import ChooseTicksParams

    # Create YTickLabels instance
    y_tick_labels = YTickLabels(font_family="DejaVu Sans", font_size=mm(3.0))

    # Initially, no tick labels are stored
    assert y_tick_labels.tick_labels is None

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
    y_scale = ScaleContinuousLength("y", TickCoverage.StrictSub, tick_params)
    unscaled_values = UnscaledValues(
        "y", np.array([10.0, 25.5, 40.0, 62.25]), CtxLenType.Pos
    )
    y_scale.fit_values(unscaled_values)

    # Create a mock ScaleSet
    scales = {"y": y_scale}

    # Apply scales
    y_tick_labels.apply_scales(scales)

    # Now tick labels should be stored
    assert y_tick_labels.tick_labels is not None
    assert len(y_tick_labels.tick_labels) > 0

    # Check that bounds calculation uses the actual labels
    width_with_labels, height_with_labels = y_tick_labels.abs_bounds()

    # Create a new instance without applied scales for comparison
    y_tick_labels_no_scales = YTickLabels(font_family="DejaVu Sans", font_size=mm(3.0))
    width_without_labels, height_without_labels = y_tick_labels_no_scales.abs_bounds()

    # The bounds might be different since one uses actual labels, other uses estimate
    assert width_with_labels.scalar_value() > 0.0
    assert height_with_labels.scalar_value() > 0.0


def test_ticklabels_bounds_precision_with_long_labels():
    """Test that apply_scales provides more precise bounds for longer labels."""
    import numpy as np
    from dapple.scales import ScaleContinuousLength, UnscaledValues, TickCoverage
    from dapple.coordinates import CtxLenType
    from dapple.config import ChooseTicksParams

    # Create YTickLabels instance
    y_tick_labels = YTickLabels(font_family="DejaVu Sans", font_size=mm(3.0))

    # Create a scale that will generate longer labels
    tick_params = ChooseTicksParams(
        k_min=2,
        k_max=10,
        k_ideal=5,
        granularity_weight=1 / 4,
        simplicity_weight=1 / 6,
        coverage_weight=1 / 2,
        niceness_weight=1 / 4,
    )
    y_scale = ScaleContinuousLength("y", TickCoverage.StrictSub, tick_params)
    # Use values that will create labels like "1000.0", "2000.0", etc.
    unscaled_values = UnscaledValues(
        "y", np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]), CtxLenType.Pos
    )
    y_scale.fit_values(unscaled_values)

    # Get bounds before and after applying scales
    width_before, height_before = y_tick_labels.abs_bounds()

    # Apply scales
    scales = {"y": y_scale}
    y_tick_labels.apply_scales(scales)

    # Get bounds after applying scales
    width_after, height_after = y_tick_labels.abs_bounds()

    # With longer actual labels, the width should be larger than the "0.00" estimate
    assert width_after.scalar_value() > width_before.scalar_value()
    # Height might be similar since font size is the same
    assert height_after.scalar_value() > 0.0
    assert height_before.scalar_value() > 0.0


def test_ticklabels_apply_scales_missing_scale():
    """Test that apply_scales handles gracefully when the required scale is missing."""
    # Create instances with actual values instead of ConfigKeys
    x_tick_labels = XTickLabels(
        font_family="DejaVu Sans", font_size=mm(3.0), fill="#333333"
    )
    y_tick_labels = YTickLabels(
        font_family="DejaVu Sans", font_size=mm(3.0), fill="#333333"
    )

    # Apply scales with missing required scales
    empty_scales = {}
    x_tick_labels.apply_scales(empty_scales)
    y_tick_labels.apply_scales(empty_scales)

    # Should not crash and tick labels should remain None
    assert x_tick_labels.tick_labels is None
    assert y_tick_labels.tick_labels is None

    # Bounds should still work (using fallback)
    x_width, x_height = x_tick_labels.abs_bounds()
    y_width, y_height = y_tick_labels.abs_bounds()

    assert x_width.scalar_value() > 0.0
    assert x_height.scalar_value() > 0.0
    assert y_width.scalar_value() > 0.0
    assert y_height.scalar_value() > 0.0


def test_ticklabels_integration_with_apply_scales():
    """Test that tick labels work in complete plotting context with apply_scales."""
    import numpy as np
    from dapple import plot, inch
    from dapple.geometry.points import points

    # Create data that will generate specific tick labels
    x_data = np.array(
        [100, 200, 300, 400, 500]
    )  # Will likely generate nice round ticks
    y_data = np.array([1000, 2000, 3000, 4000, 5000])  # Will generate larger labels

    # Create tick label instances with actual values instead of ConfigKeys
    x_labels = xticklabels(font_family="DejaVu Sans", font_size=mm(2.0), fill="#333333")
    y_labels = yticklabels(font_family="DejaVu Sans", font_size=mm(2.0), fill="#333333")

    # Check initial bounds (should use estimates)
    x_width_before, x_height_before = x_labels.abs_bounds()
    y_width_before, y_height_before = y_labels.abs_bounds()

    # Create plot (this will trigger apply_scales during resolution)
    pl = plot(points(x_data, y_data), x_labels, y_labels)

    # Generate SVG (this will call apply_scales internally)
    svg_root = pl.svg(inch(5), inch(4))
    assert svg_root is not None

    # The bounds calculation should now be more precise
    # Note: In a real integration, apply_scales would be called during plot resolution
    # but we can't easily test that here without more complex setup
