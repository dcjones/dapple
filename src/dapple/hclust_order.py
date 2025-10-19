from __future__ import annotations

"""
Hierarchical clustering order utilities for discrete scales.

This module provides `HClustOrder`, which computes hierarchical clustering
for the rows and columns of a 2D array and exposes `rows` and `cols`
order_by callables compatible with Dapple's discrete scales.
These callables accept either (values) or (values, targets); when targets are
provided, they are interpreted as matrix indices for clustering.

Example
-------
    import numpy as np
    from dapple.scales import xdiscrete, ydiscrete
    from dapple.hclust_order import HClustOrder

    X = np.random.randn(10, 12)
    hcl = HClustOrder(X, metric="euclidean", method="average")

    xscale = xdiscrete(order_by=hcl.cols)  # reorder columns scale by clustering
    yscale = ydiscrete(order_by=hcl.rows)  # reorder rows scale by clustering

You can also supply labels for rows/columns if your discrete scale values
are strings (or otherwise not numeric indices):

    row_labels = [f"r{i}" for i in range(X.shape[0])]
    col_labels = [f"c{j}" for j in range(X.shape[1])]
    hcl = HClustOrder(X, row_labels=row_labels, col_labels=col_labels)

    # Now order_by will interpret labels in terms of the clustering order
    yscale = ydiscrete(values=row_labels, order_by=hcl.rows)
    xscale = xdiscrete(values=col_labels, order_by=hcl.cols)
"""

from typing import Callable, Sequence, Iterable, Optional, Union, Tuple, Dict, List
import numpy as np

# Types for order_by callables accepted by discrete scales
OrderBy = Callable[[Sequence[object]], Sequence[object]]
Metric = Union[str, Callable[[np.ndarray, np.ndarray], float]]
LinkageMethod = str


class HClustOrder:
    """
    Compute hierarchical clustering orders for rows and columns and expose
    order_by functions for discrete scale reordering.

    This class computes and caches pairwise distances and hierarchical
    clustering linkages for both rows (X) and columns (X.T), so that applying
    both `.rows` and `.cols` does not re-compute distances.

    Parameters
    ----------
    X:
        2D array-like input data of shape (n_rows, n_cols).
    metric:
        Distance metric for both rows and columns. Defaults to 'euclidean'.
        Can be any metric accepted by scipy.spatial.distance.pdist.
    method:
        Linkage method for both rows and columns. Defaults to 'average'.
        See scipy.cluster.hierarchy.linkage for allowed methods.
    optimal_ordering:
        Whether to use optimal leaf ordering. Defaults to True.
    row_labels:
        Optional labels for rows. If provided, the `rows` order_by callable
        will interpret input values as these labels.
    col_labels:
        Optional labels for columns. If provided, the `cols` order_by callable
        will interpret input values as these labels.

    Notes
    -----
    - If labels are not provided, the order_by callables will attempt to
      interpret input values as numeric indices (ints or float-castable ints),
      which aligns with the default heatmap geometry that uses 0..n-1 positions.
    - Unknown/unmappable values are preserved at the end in their original
      relative order.
    """

    def __init__(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        metric: Metric = "euclidean",
        method: LinkageMethod = "average",
        optimal_ordering: bool = True,
        row_labels: Optional[Sequence[object]] = None,
        col_labels: Optional[Sequence[object]] = None,
    ) -> None:
        try:
            from scipy.spatial.distance import pdist
            from scipy.cluster.hierarchy import linkage, leaves_list
        except Exception as exc:
            raise ImportError(
                "scipy is required for hierarchical clustering. "
                "Install with: pip install scipy"
            ) from exc

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_rows, n_cols = X_arr.shape

        # Optional label -> position mappings
        self._row_label_to_pos: Optional[Dict[object, int]] = (
            {lab: i for i, lab in enumerate(row_labels)}
            if row_labels is not None
            else None
        )
        self._col_label_to_pos: Optional[Dict[object, int]] = (
            {lab: j for j, lab in enumerate(col_labels)}
            if col_labels is not None
            else None
        )

        # Compute and cache distances and hierarchical orders for rows
        row_pdist = pdist(X_arr, metric=metric)
        row_linkage = linkage(
            row_pdist, method=method, optimal_ordering=optimal_ordering
        )
        row_order = leaves_list(row_linkage).astype(int)
        row_rank = np.empty(n_rows, dtype=int)
        row_rank[row_order] = np.arange(n_rows)

        # Compute and cache distances and hierarchical orders for columns
        col_pdist = pdist(X_arr.T, metric=metric)
        col_linkage = linkage(
            col_pdist, method=method, optimal_ordering=optimal_ordering
        )
        col_order = leaves_list(col_linkage).astype(int)
        col_rank = np.empty(n_cols, dtype=int)
        col_rank[col_order] = np.arange(n_cols)

        # Store required artifacts
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._row_rank = row_rank
        self._col_rank = col_rank

        # Prepare order_by callables
        self._rows_order_by: OrderBy = self._make_order_by(
            n_items=n_rows, rank=row_rank, label_to_pos=self._row_label_to_pos
        )
        self._cols_order_by: OrderBy = self._make_order_by(
            n_items=n_cols, rank=col_rank, label_to_pos=self._col_label_to_pos
        )

    @property
    def rows(self) -> OrderBy:
        """
        An order_by callable suitable for ydiscrete(order_by=...).

        The callable accepts either:
          - values: a sequence of distinct values (row labels or numeric indices), or
          - values and targets: where targets are the scale targets corresponding
            to those values; when provided, targets are interpreted as row indices.
        It returns a new sequence ordered according to the hierarchical clustering
        of the rows of X.
        """
        return self._rows_order_by

    @property
    def cols(self) -> OrderBy:
        """
        An order_by callable suitable for xdiscrete(order_by=...).

        The callable accepts either:
          - values: a sequence of distinct values (column labels or numeric indices), or
          - values and targets: where targets are the scale targets corresponding
            to those values; when provided, targets are interpreted as column indices.
        It returns a new sequence ordered according to the hierarchical clustering
        of the columns of X.
        """
        return self._cols_order_by

    @staticmethod
    def _maybe_pos_from_value(
        value: object,
        n_items: int,
        label_to_pos: Optional[Dict[object, int]],
    ) -> Optional[int]:
        """
        Try to infer a position index from an arbitrary value.

        Strategy:
          1) If a label_to_pos mapping is provided, use it.
          2) Else, if value is an int or a float like 3.0, interpret as index.
        """
        if label_to_pos is not None:
            if value in label_to_pos:
                return label_to_pos[value]
            return None

        # Fall back to interpreting values as numeric indices
        if isinstance(value, (int, np.integer)):
            pos = int(value)
            return pos if 0 <= pos < n_items else None
        if isinstance(value, (float, np.floating)):
            # Accept 3.0 as index 3
            if float(value).is_integer():
                pos = int(value)
                return pos if 0 <= pos < n_items else None
        return None

    def _make_order_by(
        self,
        n_items: int,
        rank: np.ndarray,
        label_to_pos: Optional[Dict[object, int]],
    ) -> OrderBy:
        """
        Create an order_by callable that sorts input values by the given rank,
        where rank[pos] gives the leaf order rank for position pos.
        """

        def order_by(
            values: Sequence[object],
            targets: Optional[Sequence[object]] = None,
        ) -> List[object]:
            # Build sorting keys: (missing_flag, cluster_rank, original_index)
            keys: List[Tuple[int, int, int]] = []
            for i, v in enumerate(values):
                pos: Optional[int] = None

                # Prefer interpreting provided targets as matrix indices
                if targets is not None and i < len(targets):
                    t = targets[i]
                    if t is not None:
                        if isinstance(t, (int, np.integer)):
                            pos = int(t)
                        elif (
                            isinstance(t, (float, np.floating))
                            and float(t).is_integer()
                        ):
                            pos = int(t)

                        # Enforce bounds if we parsed a numeric index
                        if pos is not None and not (0 <= pos < n_items):
                            pos = None

                # Fallback: infer position from value (labels or direct indices)
                if pos is None:
                    pos = self._maybe_pos_from_value(v, n_items, label_to_pos)

                if pos is None:
                    # Unknown values go to the end in their original order
                    keys.append((1, i, i))
                else:
                    keys.append((0, int(rank[pos]), i))

            sorted_indices = sorted(range(len(values)), key=lambda j: keys[j])
            return [values[j] for j in sorted_indices]

        return order_by


__all__ = ["HClustOrder"]
