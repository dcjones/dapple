from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .bars import vertical_bars


def histogram(
    a: Any,
    bins: Any = 10,
    range: Optional[tuple[float, float]] = None,
    density: bool = False,
    weights: Optional[Sequence[float]] = None,
    *,
    color: Optional[Any] = None,
):
    """
    Construct bar geometry from the numpy histogram parameters.

    Args mirror numpy.histogram so that all bin-selection heuristics and weighting
    behaviours are available. The resulting counts (or densities) are rendered as
    vertical bars spanning the returned bin edges.
    """

    counts, edges = np.histogram(
        a, bins=bins, range=range, density=density, weights=weights
    )

    return vertical_bars(
        y=counts,
        xmin=edges[:-1],
        xmax=edges[1:],
        color=color,
    )
