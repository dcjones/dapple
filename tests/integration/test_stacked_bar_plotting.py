from __future__ import annotations

import io

import dapple as dpl
from dapple.geometry import stacked_horizontal_bars, stacked_vertical_bars


def _render(element) -> str:
    p = dpl.plot()
    p.append(element)
    out = io.StringIO()
    p.svg(dpl.inch(4), dpl.inch(4), output=out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# stacked_vertical_bars
# ---------------------------------------------------------------------------


def test_stacked_vertical_bars_categorical_groups():
    x = ["A", "A", "B", "B", "C", "C"]
    y = [10, 20, 15, 25, 8, 12]
    color = ["cats", "dogs", "cats", "dogs", "cats", "dogs"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_three_segments():
    x = ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb"]
    y = [10, 20, 30, 5, 15, 25]
    color = ["apples", "oranges", "bananas", "apples", "oranges", "bananas"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_percent_normalized():
    x = ["A", "A", "B", "B"]
    y = [10, 40, 25, 25]
    color = ["group1", "group2", "group1", "group2"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color, normalize=100))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_proportion_normalized():
    x = ["X", "X", "X"]
    y = [1, 2, 3]
    color = ["a", "b", "c"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color, normalize=1))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_numeric_x_positions():
    x = [1, 1, 2, 2, 3, 3]
    y = [4, 6, 7, 3, 5, 5]
    color = ["alpha", "beta", "alpha", "beta", "alpha", "beta"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_duplicate_observations_summed():
    # Two rows share (x="A", color="cats"); their y values should be summed.
    x = ["A", "A", "A", "B", "B"]
    y = [5, 5, 20, 15, 25]
    color = ["cats", "cats", "dogs", "cats", "dogs"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_custom_width():
    x = ["P", "P", "Q", "Q"]
    y = [3, 7, 4, 6]
    color = ["m", "n", "m", "n"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color, width=0.5))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_vertical_bars_single_stack():
    x = ["only", "only", "only"]
    y = [10, 30, 20]
    color = ["red_group", "green_group", "blue_group"]

    svg = _render(stacked_vertical_bars(x=x, y=y, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg


# ---------------------------------------------------------------------------
# stacked_horizontal_bars
# ---------------------------------------------------------------------------


def test_stacked_horizontal_bars_categorical_groups():
    y = ["A", "A", "B", "B", "C", "C"]
    x = [10, 20, 15, 25, 8, 12]
    color = ["cats", "dogs", "cats", "dogs", "cats", "dogs"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_three_segments():
    y = ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb"]
    x = [10, 20, 30, 5, 15, 25]
    color = ["apples", "oranges", "bananas", "apples", "oranges", "bananas"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_percent_normalized():
    y = ["A", "A", "B", "B"]
    x = [10, 40, 25, 25]
    color = ["group1", "group2", "group1", "group2"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color, normalize=100))

    assert "<svg" in svg
    assert "<rect" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_proportion_normalized():
    y = ["X", "X", "X"]
    x = [1, 2, 3]
    color = ["a", "b", "c"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color, normalize=1))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_numeric_y_positions():
    y = [1, 1, 2, 2, 3, 3]
    x = [4, 6, 7, 3, 5, 5]
    color = ["alpha", "beta", "alpha", "beta", "alpha", "beta"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_duplicate_observations_summed():
    y = ["A", "A", "A", "B", "B"]
    x = [5, 5, 20, 15, 25]
    color = ["cats", "cats", "dogs", "cats", "dogs"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_custom_width():
    y = ["P", "P", "Q", "Q"]
    x = [3, 7, 4, 6]
    color = ["m", "n", "m", "n"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color, width=0.5))

    assert "<svg" in svg
    assert "dapple:" not in svg


def test_stacked_horizontal_bars_single_stack():
    y = ["only", "only", "only"]
    x = [10, 30, 20]
    color = ["red_group", "green_group", "blue_group"]

    svg = _render(stacked_horizontal_bars(y=y, x=x, color=color))

    assert "<svg" in svg
    assert "dapple:" not in svg
