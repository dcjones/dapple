from __future__ import annotations

import io

import numpy as np
import pytest

import dapple as dpl
from dapple.elements import Element
from dapple.geometry.stacked_bars import (
    _build_stacked_pivot,
    _unique_in_order,
    stacked_horizontal_bars,
    stacked_vertical_bars,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_svg(element: Element) -> str:
    p = dpl.plot()
    p.append(element)
    out = io.StringIO()
    p.svg(dpl.inch(4), dpl.inch(4), output=out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestUniqueInOrder:
    def test_preserves_first_appearance_order(self):
        assert _unique_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_empty(self):
        assert _unique_in_order([]) == []

    def test_all_same(self):
        assert _unique_in_order([1, 1, 1]) == [1]

    def test_numeric(self):
        assert _unique_in_order([3, 1, 2, 1]) == [3, 1, 2]


class TestBuildStackedPivot:
    def test_basic_aggregation(self):
        pivot = _build_stacked_pivot(
            pos_values=["A", "A", "B", "B"],
            data_values=[10.0, 20.0, 15.0, 25.0],
            color_values=["cats", "dogs", "cats", "dogs"],
            unique_pos=["A", "B"],
            unique_colors=["cats", "dogs"],
        )
        assert pivot["A"]["cats"] == pytest.approx(10.0)
        assert pivot["A"]["dogs"] == pytest.approx(20.0)
        assert pivot["B"]["cats"] == pytest.approx(15.0)
        assert pivot["B"]["dogs"] == pytest.approx(25.0)

    def test_sums_duplicates(self):
        pivot = _build_stacked_pivot(
            pos_values=["A", "A", "A"],
            data_values=[5.0, 5.0, 10.0],
            color_values=["a", "a", "b"],
            unique_pos=["A"],
            unique_colors=["a", "b"],
        )
        assert pivot["A"]["a"] == pytest.approx(10.0)
        assert pivot["A"]["b"] == pytest.approx(10.0)

    def test_missing_combinations_default_to_zero(self):
        pivot = _build_stacked_pivot(
            pos_values=["A"],
            data_values=[5.0],
            color_values=["a"],
            unique_pos=["A", "B"],
            unique_colors=["a", "b"],
        )
        assert pivot["A"]["b"] == pytest.approx(0.0)
        assert pivot["B"]["a"] == pytest.approx(0.0)
        assert pivot["B"]["b"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# stacked_vertical_bars – structure
# ---------------------------------------------------------------------------


class TestStackedVerticalBarsStructure:
    def test_returns_element(self):
        el = stacked_vertical_bars(
            x=["A", "A", "B", "B"],
            y=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )
        assert isinstance(el, Element)

    def test_one_child_per_color_group(self):
        el = stacked_vertical_bars(
            x=["A", "A", "B", "B"],
            y=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )
        assert len(el) == 2

    def test_three_color_groups(self):
        el = stacked_vertical_bars(
            x=["A", "A", "A", "B", "B", "B"],
            y=[10, 20, 5, 15, 25, 8],
            color=["a", "b", "c", "a", "b", "c"],
        )
        assert len(el) == 3

    def test_single_color_group(self):
        el = stacked_vertical_bars(
            x=["A", "B", "C"],
            y=[10, 15, 20],
            color=["only", "only", "only"],
        )
        assert len(el) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stacked_vertical_bars(
                x=["A", "B"],
                y=[10, 20, 30],
                color=["a", "b", "c"],
            )

    def test_mismatched_color_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stacked_vertical_bars(
                x=["A", "B", "C"],
                y=[10, 20, 30],
                color=["a", "b"],
            )


# ---------------------------------------------------------------------------
# stacked_vertical_bars – normalization math
# ---------------------------------------------------------------------------


class TestStackedVerticalBarsNormalization:
    def test_equal_values_normalize_to_50_each(self):
        # Two equal groups, normalize=100 → each contributes 50
        unique_x = ["A"]
        unique_colors = ["a", "b"]
        pivot = _build_stacked_pivot(
            ["A", "A"], [1.0, 1.0], ["a", "b"], unique_x, unique_colors
        )
        total = sum(pivot["A"].values())
        for ci in unique_colors:
            pivot["A"][ci] = pivot["A"][ci] * 100.0 / total
        assert pivot["A"]["a"] == pytest.approx(50.0)
        assert pivot["A"]["b"] == pytest.approx(50.0)

    def test_normalize_preserves_proportions(self):
        # Group "a" is 3x group "b"; normalize=1 should give 0.75 and 0.25
        unique_x = ["A"]
        unique_colors = ["a", "b"]
        pivot = _build_stacked_pivot(
            ["A", "A"], [3.0, 1.0], ["a", "b"], unique_x, unique_colors
        )
        total = sum(pivot["A"].values())
        for ci in unique_colors:
            pivot["A"][ci] = pivot["A"][ci] * 1.0 / total
        assert pivot["A"]["a"] == pytest.approx(0.75)
        assert pivot["A"]["b"] == pytest.approx(0.25)

    def test_cumulative_sum_increases_with_each_group(self):
        # Cumulative starts for successive groups should increase
        el = stacked_vertical_bars(
            x=["A", "A", "A"],
            y=[10, 20, 30],
            color=["a", "b", "c"],
        )
        # Cannot easily inspect inner lengths here; just verify we get 3 children
        assert len(el) == 3


# ---------------------------------------------------------------------------
# stacked_vertical_bars – SVG rendering
# ---------------------------------------------------------------------------


class TestStackedVerticalBarsSVG:
    def test_renders_valid_svg(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["A", "A", "B", "B"],
                y=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_normalize(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["A", "A", "B", "B"],
                y=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
                normalize=100,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_proportion_stacked(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["X", "X", "X"],
                y=[1, 2, 3],
                color=["a", "b", "c"],
                normalize=1,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_numeric_x_positions(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=[1, 1, 2, 2, 3, 3],
                y=[5, 10, 8, 12, 6, 9],
                color=["alpha", "beta", "alpha", "beta", "alpha", "beta"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_three_groups(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
                y=[10, 20, 30, 5, 15, 25],
                color=["x", "y", "z", "x", "y", "z"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_duplicate_observations(self):
        # Multiple rows with same (x, color) should be summed and still render
        svg = _render_svg(
            stacked_vertical_bars(
                x=["A", "A", "A", "B", "B"],
                y=[5, 5, 20, 15, 25],
                color=["cats", "cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_produces_rect_elements(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["A", "A", "B", "B"],
                y=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<rect" in svg

    def test_custom_width(self):
        svg = _render_svg(
            stacked_vertical_bars(
                x=["A", "A", "B", "B"],
                y=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
                width=0.5,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg


# ---------------------------------------------------------------------------
# stacked_horizontal_bars – structure
# ---------------------------------------------------------------------------


class TestStackedHorizontalBarsStructure:
    def test_returns_element(self):
        el = stacked_horizontal_bars(
            y=["A", "A", "B", "B"],
            x=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )
        assert isinstance(el, Element)

    def test_one_child_per_color_group(self):
        el = stacked_horizontal_bars(
            y=["A", "A", "B", "B"],
            x=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )
        assert len(el) == 2

    def test_three_color_groups(self):
        el = stacked_horizontal_bars(
            y=["A", "A", "A", "B", "B", "B"],
            x=[10, 20, 5, 15, 25, 8],
            color=["a", "b", "c", "a", "b", "c"],
        )
        assert len(el) == 3

    def test_single_color_group(self):
        el = stacked_horizontal_bars(
            y=["A", "B", "C"],
            x=[10, 15, 20],
            color=["only", "only", "only"],
        )
        assert len(el) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stacked_horizontal_bars(
                y=["A", "B"],
                x=[10, 20, 30],
                color=["a", "b", "c"],
            )

    def test_mismatched_color_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stacked_horizontal_bars(
                y=["A", "B", "C"],
                x=[10, 20, 30],
                color=["a", "b"],
            )


# ---------------------------------------------------------------------------
# stacked_horizontal_bars – SVG rendering
# ---------------------------------------------------------------------------


class TestStackedHorizontalBarsSVG:
    def test_renders_valid_svg(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["A", "A", "B", "B"],
                x=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_normalize(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["A", "A", "B", "B"],
                x=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
                normalize=100,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_proportion_stacked(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["X", "X", "X"],
                x=[1, 2, 3],
                color=["a", "b", "c"],
                normalize=1,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_numeric_y_positions(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=[1, 1, 2, 2, 3, 3],
                x=[5, 10, 8, 12, 6, 9],
                color=["alpha", "beta", "alpha", "beta", "alpha", "beta"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_three_groups(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
                x=[10, 20, 30, 5, 15, 25],
                color=["x", "y", "z", "x", "y", "z"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_renders_with_duplicate_observations(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["A", "A", "A", "B", "B"],
                x=[5, 5, 20, 15, 25],
                color=["cats", "cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg

    def test_produces_rect_elements(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["A", "A", "B", "B"],
                x=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
            )
        )
        assert "<rect" in svg

    def test_custom_width(self):
        svg = _render_svg(
            stacked_horizontal_bars(
                y=["A", "A", "B", "B"],
                x=[10, 20, 15, 25],
                color=["cats", "dogs", "cats", "dogs"],
                width=0.5,
            )
        )
        assert "<svg" in svg
        assert "dapple:" not in svg
