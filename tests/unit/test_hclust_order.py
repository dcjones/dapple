import numpy as np
import pytest

from dapple.hclust_order import HClustOrder
from dapple.scales import xdiscrete, ydiscrete


def _assert_pair_adjacent(seq, a, b):
    ia, ib = seq.index(a), seq.index(b)
    assert abs(ia - ib) == 1, f"Expected {a} and {b} to be adjacent in {seq}"


class TestHClustOrderBasics:
    def test_orders_indices_by_clusters_rows_and_cols(self):
        # Build a matrix with two clear clusters along both axes
        # Rows: {0,1} similar, {2,3} similar
        # Cols: {0,1} similar, {2,3} similar
        X = np.zeros((4, 4), dtype=float)
        X[:2, :2] += 2.0  # top-left block
        X[2:, 2:] += 2.0  # bottom-right block

        hcl = HClustOrder(
            X, metric="euclidean", method="average", optimal_ordering=True
        )

        rows_order = list(hcl.rows([0, 1, 2, 3]))
        cols_order = list(hcl.cols([0, 1, 2, 3]))

        # We don't assert exact permutations (ties are possible), but cluster adjacency should hold
        _assert_pair_adjacent(rows_order, 0, 1)
        _assert_pair_adjacent(rows_order, 2, 3)
        _assert_pair_adjacent(cols_order, 0, 1)
        _assert_pair_adjacent(cols_order, 2, 3)

    def test_orders_labels_with_optional_label_maps(self):
        # Create columns with two clusters:
        # c0 ~ c1 near 0, c2 ~ c3 near 10 (slight perturbations)
        X = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.1, 10.0, 10.1],
                [0.0, 0.05, 10.0, 10.05],
            ],
            dtype=float,
        )
        row_labels = [f"r{i}" for i in range(X.shape[0])]
        col_labels = [f"c{j}" for j in range(X.shape[1])]

        hcl = HClustOrder(
            X,
            metric="euclidean",
            method="average",
            optimal_ordering=True,
            row_labels=row_labels,
            col_labels=col_labels,
        )

        # Start from a shuffled order; clustering should put pairs adjacent
        shuffled_cols = ["c2", "c0", "c3", "c1"]
        ordered_cols = list(hcl.cols(shuffled_cols))

        _assert_pair_adjacent(ordered_cols, "c0", "c1")
        _assert_pair_adjacent(ordered_cols, "c2", "c3")

    def test_unknown_values_are_preserved_at_end(self):
        X = np.array(
            [
                [0.0, 0.1, 10.0],
                [0.0, 0.2, 10.0],
            ],
            dtype=float,
        )
        col_labels = ["c0", "c1", "c2"]
        hcl = HClustOrder(
            X,
            metric="euclidean",
            method="average",
            optimal_ordering=True,
            col_labels=col_labels,
        )

        # "z" is unknown; it should be preserved at the end in original order
        known_sorted = list(hcl.cols(["c0", "c2"]))
        with_unknown = list(hcl.cols(["c0", "z", "c2"]))
        assert with_unknown == known_sorted + ["z"]


class TestHClustOrderWithDiscreteScales:
    def test_xdiscrete_order_by_hclust_cols(self):
        # Matrix with clustered columns: {c0, c1} and {c2, c3}
        X = np.array(
            [
                [0.0, 0.05, 10.0, 10.05],
                [0.0, 0.0, 10.0, 10.1],
                [0.0, 0.1, 10.0, 10.0],
            ],
            dtype=float,
        )
        col_labels = ["c0", "c1", "c2", "c3"]
        hcl = HClustOrder(
            X,
            metric="euclidean",
            method="average",
            optimal_ordering=True,
            col_labels=col_labels,
        )

        # Provide values as a Mapping to control labels explicitly
        # mapping: value -> label (str case means target None, label is the str)
        values_map = {"c2": "c2", "c0": "c0", "c3": "c3", "c1": "c1"}

        xscale = xdiscrete(values=values_map, order_by=hcl.cols)
        # The plot pipeline would call finalize(), but we can call it directly
        xscale.finalize()

        expected = list(hcl.cols(list(values_map.keys())))
        assert list(xscale.labels) == expected

    def test_ydiscrete_order_by_hclust_rows(self):
        # Matrix with clustered rows: {r0, r1} and {r2, r3}
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.05, 0.0],
                [10.0, 10.0, 10.0],
                [10.1, 10.0, 10.1],
            ],
            dtype=float,
        )
        row_labels = ["r0", "r1", "r2", "r3"]
        hcl = HClustOrder(
            X,
            metric="euclidean",
            method="average",
            optimal_ordering=True,
            row_labels=row_labels,
        )

        values_map = {"r2": "r2", "r0": "r0", "r3": "r3", "r1": "r1"}

        yscale = ydiscrete(values=values_map, order_by=hcl.rows)
        yscale.finalize()

        expected = list(hcl.rows(list(values_map.keys())))
        assert list(yscale.labels) == expected
