import io
import math

import numpy as np
import pytest

from dapple.colors import Colors
from dapple.config import Config
from dapple.coordinates import Lengths
from dapple.elements import Element
from dapple.geometry import horizontal_violin, vertical_violin, violin
from dapple.scales import (
    ScaleContinuousColor,
    ScaleContinuousLength,
    ScaleDiscreteColor,
    ScaleDiscreteLength,
    ScaleSet,
    UnscaledExpr,
    UnscaledValues,
)


def _collect_unscaled_units(el: Element) -> set[str]:
    """
    Traverse UnscaledExpr attributes and collect all unit names found in UnscaledValues.
    """
    units: set[str] = set()

    def visit_values(values: UnscaledValues):
        units.add(values.unit)

    def visit_attr(_key: str, expr: UnscaledExpr):
        expr.accept_visitor(visit_values)

    el.traverse_attributes(visit_attr, UnscaledExpr)
    return units


def _build_scales_for_element(el: Element) -> ScaleSet:
    """
    Mimic the Plot.resolve scale discovery:
    - Traverse UnscaledValues to determine if each unit is all-numeric
    - Create appropriate scales for x/y/color based on numeric/discrete
    """
    all_numeric: dict[str, bool] = {}

    def track_values(values: UnscaledValues):
        u = values.unit
        is_numeric = values.all_numeric()
        all_numeric[u] = all_numeric.get(u, True) and is_numeric

    def visitor(_key: str, expr: UnscaledExpr):
        expr.accept_visitor(track_values)

    el.traverse_attributes(visitor, UnscaledExpr)

    scales: ScaleSet = ScaleSet()
    for unit, numeric in all_numeric.items():
        if unit == "color":
            scales[unit] = (
                ScaleContinuousColor(unit) if numeric else ScaleDiscreteColor(unit)
            )
        elif unit == "shape":
            # shape not used here
            continue
        else:
            scales[unit] = (
                ScaleContinuousLength(unit) if numeric else ScaleDiscreteLength(unit)
            )

    # Resolve any ConfigKey defaults on the scales (e.g., colormaps)
    Config().replace_keys(scales)

    # Fit and finalize scales
    def fit_visitor(_key: str, expr: UnscaledExpr):
        expr.accept_fit(scales)

    el.traverse_attributes(fit_visitor, UnscaledExpr)

    for s in scales.values():
        s.finalize()

    return scales


def _scale_element(el: Element) -> Element:
    """
    Apply scaling to an element by rewriting UnscaledExpr attributes
    into Lengths/Colors using the constructed ScaleSet.
    """
    scales = _build_scales_for_element(el)

    def scale_attr(_key: str, expr: UnscaledExpr):
        return expr.accept_scale(scales)

    return el.rewrite_attributes(scale_attr, UnscaledExpr)


def _iter_paths(el: Element):
    """
    Normalize return to an iterator over Path-like children:
    - If el is a container 'g', iterate children
    - If el is a single path-like element, yield it
    """
    if el.tag == "g":
        for child in el:
            yield from _iter_paths(child)
    else:
        yield el


class TestVerticalViolin:
    def test_vertical_violin_with_non_numeric_x_groups(self):
        # Non-numeric x groups; numeric y data
        x = ["A", "A", "B", "B", "B", "C"]
        y = [1.0, 1.5, 2.2, 2.0, 1.9, 0.7]

        el = vertical_violin(x=x, y=y)
        # Expect either a container with multiple paths or a single path if one group
        paths = list(_iter_paths(el))
        assert len(paths) >= 1
        for p in paths:
            assert p.tag in ("dapple:path", "dapple:violin_quartile", "line")

        # Pre-scale: ensure units include x, y, and color
        units = _collect_unscaled_units(el)
        assert "x" in units
        assert "y" in units
        # Color unit may not be present pre-scale; only assert positional units

        # After scaling, x and y should be Lengths (CtxLengths)
        scaled = _scale_element(el)
        for p in _iter_paths(scaled):
            if p.tag == "dapple:path":
                assert isinstance(p.attrib.get("x"), Lengths)
                assert isinstance(p.attrib.get("y"), Lengths)
            elif p.tag == "dapple:violin_quartile":
                assert isinstance(p.attrib.get("center"), Lengths)
                assert isinstance(p.attrib.get("q_low"), Lengths)
                assert isinstance(p.attrib.get("q_high"), Lengths)
                assert isinstance(p.attrib.get("q_med"), Lengths)
            elif p.tag == "line":
                assert isinstance(p.attrib.get("x1"), Lengths)
                assert isinstance(p.attrib.get("y1"), Lengths)
                assert isinstance(p.attrib.get("x2"), Lengths)
                assert isinstance(p.attrib.get("y2"), Lengths)

        path_colors = []
        overlay_colors = []
        for p in _iter_paths(scaled):
            if p.tag == "dapple:path" and isinstance(p.attrib.get("fill"), Colors):
                path_colors.append(p.attrib["fill"])
            elif p.tag == "dapple:violin_quartile" and isinstance(
                p.attrib.get("fill"), Colors
            ):
                overlay_colors.append(p.attrib["fill"])

        assert len(overlay_colors) == len(path_colors)
        for base, box in zip(path_colors, overlay_colors):
            expected = base.modulate_lightness(0.4)
            assert np.allclose(box.values, expected.values)

    def test_vertical_violin_single_group_returns_path(self):
        x = ["Z"] * 20
        y = np.linspace(0.0, 1.0, 20)
        el = vertical_violin(x=x, y=y)
        # Single group => Path directly (pre-resolve may be a custom violin element)
        assert el.tag in ("g", "dapple:path")

        units = _collect_unscaled_units(el)
        assert "x" in units and "y" in units

        scaled = _scale_element(el)
        if scaled.tag == "dapple:path":
            assert isinstance(scaled.attrib.get("x"), Lengths)
            assert isinstance(scaled.attrib.get("y"), Lengths)
        else:
            children = list(_iter_paths(scaled))
            assert any(child.tag == "dapple:path" for child in children)

    def test_violin_alias_equivalence(self):
        # violin is an alias for vertical_violin
        x = ["G1", "G2", "G1", "G2"]
        y = [0.2, 0.8, 0.4, 0.6]
        a = vertical_violin(x=x, y=y)
        b = violin(x=x, y=y)

        # Not necessarily identical object trees but both yield paths with same units present
        units_a = _collect_unscaled_units(a)
        units_b = _collect_unscaled_units(b)
        assert "x" in units_a and "y" in units_a
        assert "x" in units_b and "y" in units_b

    def test_vertical_violin_broadcasting(self):
        y = [1.0, 2.0, 3.0, 4.0]

        # Scalar str x
        el_scalar = vertical_violin(x="Group", y=y)
        assert el_scalar is not None

        # Scalar int x
        el_int = vertical_violin(x=1, y=y)
        assert el_int is not None

        # Single element list x
        el_list = vertical_violin(x=["Group"], y=y)
        assert el_list is not None


class TestHorizontalViolin:
    def test_horizontal_violin_with_non_numeric_y_groups(self):
        # Non-numeric y groups; numeric x data
        y = ["K", "K", "L", "L", "K", "L"]
        x = [2.1, 1.7, 3.0, 2.8, 1.3, 2.5]

        el = horizontal_violin(x=x, y=y)
        paths = list(_iter_paths(el))
        assert len(paths) >= 1
        for p in paths:
            assert p.tag in ("dapple:path", "dapple:violin_quartile", "line")

        units = _collect_unscaled_units(el)
        assert "x" in units
        assert "y" in units
        # Color unit may not be present pre-scale; only assert positional units

        scaled = _scale_element(el)
        for p in _iter_paths(scaled):
            if p.tag == "dapple:path":
                assert isinstance(p.attrib.get("x"), Lengths)
                assert isinstance(p.attrib.get("y"), Lengths)
            elif p.tag == "dapple:violin_quartile":
                assert isinstance(p.attrib.get("center"), Lengths)
                assert isinstance(p.attrib.get("q_low"), Lengths)
                assert isinstance(p.attrib.get("q_high"), Lengths)
                assert isinstance(p.attrib.get("q_med"), Lengths)
            elif p.tag == "line":
                assert isinstance(p.attrib.get("x1"), Lengths)
                assert isinstance(p.attrib.get("y1"), Lengths)
                assert isinstance(p.attrib.get("x2"), Lengths)
                assert isinstance(p.attrib.get("y2"), Lengths)

        path_colors = []
        overlay_colors = []
        for p in _iter_paths(scaled):
            if p.tag == "dapple:path" and isinstance(p.attrib.get("fill"), Colors):
                path_colors.append(p.attrib["fill"])
            elif p.tag == "dapple:violin_quartile" and isinstance(
                p.attrib.get("fill"), Colors
            ):
                overlay_colors.append(p.attrib["fill"])

        assert len(overlay_colors) == len(path_colors)
        for base, box in zip(path_colors, overlay_colors):
            expected = base.modulate_lightness(0.12)
            assert np.allclose(box.values, expected.values)

    def test_horizontal_violin_single_group_returns_path(self):
        y = ["Only"] * 16
        x = np.sin(np.linspace(0.0, 2 * math.pi, 16))
        el = horizontal_violin(x=x, y=y)
        assert el.tag in ("g", "dapple:path")

        units = _collect_unscaled_units(el)
        assert "x" in units and "y" in units

        scaled = _scale_element(el)
        if scaled.tag == "dapple:path":
            assert isinstance(scaled.attrib.get("x"), Lengths)
            assert isinstance(scaled.attrib.get("y"), Lengths)
        else:
            children = list(_iter_paths(scaled))
            assert any(child.tag == "dapple:path" for child in children)

    def test_horizontal_violin_broadcasting(self):
        x = [1.0, 2.0, 3.0, 4.0]

        # Scalar str y
        el_scalar = horizontal_violin(x=x, y="Group")
        assert el_scalar is not None

        # Scalar int y
        el_int = horizontal_violin(x=x, y=1)
        assert el_int is not None

        # Single element list y
        el_list = horizontal_violin(x=x, y=["Group"])
        assert el_list is not None
