import io

import dapple as dpl


def _render_svg(element) -> str:
    # Build a simple plot, append the provided element, and render to SVG
    p = dpl.plot()
    p.append(element)

    out = io.StringIO()
    p.svg(dpl.inch(3), dpl.inch(3), output=out)
    return out.getvalue()


def test_vertical_violin_svg_with_categorical_groups():
    # Non-numeric x groups; numeric y data with >= 2 samples per group for KDE
    x = ["A", "A", "B", "B", "B", "C", "C"]
    y = [1.0, 1.5, 2.2, 2.0, 1.9, 0.7, 0.9]

    svg = _render_svg(dpl.vertical_violin(x=x, y=y))

    # Basic sanity checks on the SVG
    assert "<svg" in svg
    # Should resolve to one or more path elements (not custom dapple tags)
    assert "<path" in svg
    # No dapple-specific tags should remain post-resolve
    assert "dapple:" not in svg


def test_horizontal_violin_svg_with_categorical_groups():
    # Non-numeric y groups; numeric x data with >= 2 samples per group for KDE
    y = ["K", "K", "L", "L", "M", "M"]
    x = [2.1, 1.7, 3.0, 2.8, 1.3, 2.5]

    svg = _render_svg(dpl.horizontal_violin(x=x, y=y))

    # Basic sanity checks on the SVG
    assert "<svg" in svg
    assert "<path" in svg
    assert "dapple:" not in svg
