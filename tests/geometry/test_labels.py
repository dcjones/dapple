import pytest
from dapple.geometry.labels import XLabel, YLabel, Title, xlabel, ylabel, title
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
        fill="#ff0000"
    )
    assert x_label.attrib["text"] == "Custom X Label"
    assert x_label.attrib["font_family"] == "Arial"
    assert x_label.attrib["font_size"] == mm(4.0)
    assert x_label.attrib["fill"] == "#ff0000"


def test_ylabel_custom_parameters():
    """Test that YLabel accepts custom styling parameters."""
    y_label = YLabel(
        "Custom Y Label",
        font_family="Times",
        font_size=mm(5.0),
        fill="#00ff00"
    )
    assert y_label.attrib["text"] == "Custom Y Label"
    assert y_label.attrib["font_family"] == "Times"
    assert y_label.attrib["font_size"] == mm(5.0)
    assert y_label.attrib["fill"] == "#00ff00"


def test_title_custom_parameters():
    """Test that Title accepts custom styling parameters."""
    plot_title = Title(
        "Custom Title",
        font_family="Helvetica",
        font_size=mm(6.0),
        fill="#0000ff"
    )
    assert plot_title.attrib["text"] == "Custom Title"
    assert plot_title.attrib["font_family"] == "Helvetica"
    assert plot_title.attrib["font_size"] == mm(6.0)
    assert plot_title.attrib["fill"] == "#0000ff"


def test_xlabel_abs_bounds():
    """Test that XLabel computes reasonable absolute bounds."""
    x_label = XLabel("Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000")
    width_bound, height_bound = x_label.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_ylabel_abs_bounds():
    """Test that YLabel computes reasonable absolute bounds."""
    y_label = YLabel("Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000")
    width_bound, height_bound = y_label.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_title_abs_bounds():
    """Test that Title computes reasonable absolute bounds."""
    plot_title = YLabel("Test Label", font_family="DejaVu Sans", font_size=mm(4.0), fill="#ff0000")
    width_bound, height_bound = plot_title.abs_bounds()

    assert width_bound.scalar_value() > 0.0
    assert height_bound.scalar_value() > 0.0


def test_different_text_lengths_affect_bounds():
    """Test that longer text results in different bounds."""
    short_label = XLabel("X", font_family="DejaVu Sans", font_size=mm(4.0))
    long_label = XLabel("Very Long X Axis Label", font_family="DejaVu Sans", font_size=mm(4.0))

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
    long_label = YLabel("Very Long Y Axis Label", font_family="DejaVu Sans", font_size=mm(4.0))

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

    pl = plot(
        points(x_data, y_data),
        xlabel("X Axis Label")
    )

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

    pl = plot(
        points(x_data, y_data),
        ylabel("Y Axis Label")
    )

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

    pl = plot(
        points(x_data, y_data),
        title("Plot Title")
    )

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
        ylabel("Y Values")
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
    unicode_label = XLabel("Åxis Lábel with ñ and 中文", "DejaVu Sans", font_size=mm(4.0))
    width_bound, height_bound = unicode_label.abs_bounds()

    # Should not crash and should return positive bounds
    assert width_bound.scalar_value() >= 0.0
    assert height_bound.scalar_value() > 0.0
