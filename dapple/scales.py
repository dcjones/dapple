
from __future__ import annotations

from .colors import Colors
from .coordinates import Lengths, CtxLengths, CtxLenType
from abc import ABC, abstractmethod
from cmap import Colormap, ColormapLike
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from typing import Any, Mapping, Sequence, Tuple, Callable, Optional, Union
import numpy as np
import operator

class UnscaledExpr(ABC):
    """
    Unscaled expression generalized over unscaled values to support simple arthmetic
    expressions, e.g. ux(["a", "b", "c"]) + mm(1.0).

    Implementing this interface requires propogating visits from the scale in
    the fit and scale passes.
    """

    @abstractmethod
    def accept_fit(self, scale: Scale):
        pass

    @abstractmethod
    def accept_scale(self, scale: Scale) -> Lengths | Colors:
        pass

    def __add__(self, other):
        return UnscaledBinaryOp(self, other, operator.add)

    def __sub__(self, other):
        return UnscaledBinaryOp(self, other, operator.sub)

    def __neg__(self):
        return UnscaledUnaryOp(self, operator.neg)


@dataclass
class UnscaledValues(UnscaledExpr):
    """
    Wraps data to be plotted along with necessary metadata. Specifically:
      * unit: A name grouping values with a shared scale
      * typ: Determines how the values with be scaled.
    """

    unit: str
    values: Iterable
    typ: CtxLenType=CtxLenType.Vec

    def accept_fit(self, scale: Scale):
        scale.fit_values(self)

    def accept_scale(self, scale: Scale) -> Lengths | Colors:
        return scale.scale_values(self)

@dataclass
class UnscaledUnaryOp(UnscaledExpr):
    """
    General purpose unary operations on unscaled value expressions.
    """

    a: Union[UnscaledExpr, Lengths]
    op: Callable

    def accept_fit(self, scale: Scale):
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_fit(scale)

    def accept_scale(self, scale: Scale) -> Lengths | Colors:
        return self.op(
            self.a.accept_scale(scale) if isinstance(self.a, UnscaledExpr) else self.a
        )

@dataclass
class UnscaledBinaryOp(UnscaledExpr):
    """
    General purpose binary operations on unscaled value expressions.
    """

    a: Union[UnscaledExpr, Lengths]
    b: Union[UnscaledExpr, Lengths]
    op: Callable

    def accept_fit(self, scale: Scale):
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_fit(scale)
        if isinstance(self.b, UnscaledExpr):
            self.b.accept_fit(scale)

    def accept_scale(self, scale: Scale) -> Lengths | Colors:
        return self.op(
            self.a.accept_scale(scale) if isinstance(self.a, UnscaledExpr) else self.a,
            self.b.accept_scale(scale) if isinstance(self.b, UnscaledExpr) else self.b
        )

class Scale(ABC):
    """
    Scales take unscaled values and transform them into values in a plottable
    unit, such as colors or lengths.

    They also generate ticks to aide with drawing guides, and support a fitting
    stage where they can first visit all the unscaled values before deciding how
    to scale them.
    """

    def fit(self, expr: UnscaledExpr):
        expr.accept_fit(self)

    @abstractmethod
    def fit_values(self, values: UnscaledValues):
        pass

    @abstractmethod
    def finalize(self):
        pass

    def scale(self, expr: UnscaledExpr) -> Lengths | Colors:
        return expr.accept_scale(self)

    @abstractmethod
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        pass

    @abstractmethod
    def ticks(self) -> Tuple[NDArray[np.str_], Lengths | Colors]:
        pass


class ScaleDiscrete(Scale):
    """
    Disecrete scale which can map any collection of (hashable) values onto lengths or colors.
    """
    unit: str
    fixed: bool
    labeler: Callable[[Any], str]
    sort_by: None | Callable[[Any], Any]
    map: dict[Any, int]
    targets: np.ndarray
    labels: NDArray

    # partial target list used while fitting the scale, maps
    # values to label target pairs.
    _targets: dict[Any, Tuple[str, Any]]

    def __init__(
            self, unit: str, values: Mapping | Sequence | None = None,
            fixed: bool=False, labeler: Callable[[Any], str] = str, sort_by: None | Callable[[Any], Any] = lambda x: x):
        self.unit = unit
        self.fixed = fixed
        self.labeler = labeler
        self.sort_by = sort_by
        self._targets = dict()

        if values == None:
            pass
        elif isinstance(values, Mapping):
            for (value, target) in values.items():
                if value in self.map:
                    raise ValueError(f"Duplicate value {value} in {values}")

                self.map[value] = len(self.map)
                match target:
                    case (label, target):
                        self._targets[value] = (target, label)
                    case str():
                        self._targets[value] = (target, None)
                    case _:
                        self._targets[value] = (self.labeler(value), target)
        elif isinstance(values, Sequence):
            for value in values:
                if value in self._targets:
                    raise ValueError(f"Duplicate value {value} in {values}")
                self._targets[value] = (self.labeler(value), None)
        else:
            raise TypeError("values must be a Mapping or Sequence")

    def fit_values(self, values: UnscaledValues):
        for value in values.values:
            if value not in self._targets:
                if self.fixed:
                    raise ValueError(f"Fixed scale cannot be updated with new value {value}")

                self._targets[value] = (self.labeler(value), None)


class ScaleDiscreteLength(ScaleDiscrete):
    """
    Discrete length scale, which maps any collection of (hashable) values onto lengths.
    """

    def __init__(
            self, unit: str, values: Mapping | Sequence | None = None,
            fixed: bool=False, labeler: Callable[[Any], str] = str, sort_by: None | Callable[[Any], Any] = lambda x: x):
        self.unit = unit
        super().__init__(unit, values, fixed, labeler, sort_by)

    def finalize(self):
        if self.sort_by is not None:
            values = sorted(self._targets.keys(), key=self.sort_by)
        else:
            values = self._targets.keys()

        self.targets = np.zeros(len(self._targets), dtype=np.float32)
        self.map = dict()
        labels = []

        next_target = max(filter(
            lambda target: target is not None,
            map(lambda v: v[1], self._targets.values())),
            default=0)

        for (i, value) in enumerate(values):
            (label, target) = self._targets[value]
            self.map[value] = i
            labels.append(label)
            if target is None:
                self.targets[i] = next_target
                next_target += 1
            else:
                self.targets[i] = target

        self.labels = np.array(labels)

    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit
        indices = np.array(self.map[value] for value in values.values)
        return CtxLengths(self.targets[indices], values.unit, values.typ)

    def ticks(self) -> Tuple[NDArray[np.str_], CtxLengths]:
        return self.labels, CtxLengths(self.targets, self.unit, CtxLenType.Pos)


def xdiscrete(*args, **kwargs) -> ScaleDiscreteLength:
    return ScaleDiscreteLength("x", *args, **kwargs)

def ydiscrete(*args, **kwargs) -> ScaleDiscreteLength:
    return ScaleDiscreteLength("y", *args, **kwargs)


class ScaleDiscreteColor(ScaleDiscrete):
    def __init__(
            self, unit: str, colormap: ColormapLike, values: Mapping | Sequence | None = None,
            fixed: bool=False, labeler: Callable[[Any], str] = str, sort_by: None | Callable[[Any], Any] = lambda x: x):

        self.colormap = Colormap(colormap)
        super().__init__(unit, values, fixed, labeler, sort_by)

    def finalize(self):
        if self.sort_by is not None:
            values = sorted(self._targets.keys(), key=self.sort_by)
        else:
            values = self._targets.keys()

        self.targets = np.zeros(len(self._targets), dtype=np.float32)
        self.map = dict()
        labels = []

        next_target = max(filter(
            lambda target: target is not None,
            map(lambda v: v[1], self._targets.values())),
            default=0)

        for (i, value) in enumerate(values):
            (label, target) = (self.labeler(value), value)
            self.map[value] = i
            labels.append(label)
            if target is None:
                self.targets[i] = next_target
                next_target += 1
            else:
                assert target >= 0
                self.targets[i] = target

        # TODO: do we want this to always contain 0 and 1? Do the colormaps wrap around?
        self.targets -= self.targets.min()
        self.targets /= self.targets.max()
        self.targets = self.colormap(self.targets)
        self.labels = np.array(labels)

    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        indices = np.array(self.map[value] for value in values.values)
        return Colors(self.targets[indices,:])

    def ticks(self) -> Tuple[NDArray[np.str_], Colors]:
        return self.labels, Colors(self.targets)

def colordiscrete(*args, **kwargs) -> ScaleDiscreteColor:
    return ScaleDiscreteColor("color", *args, **kwargs)

TickStep = namedtuple("TickStep", ["tick_step", "subtick_step", "niceness"])

TICK_STEP_OPTIONS = [
    TickStep(1.0, 0.5, 1.0), TickStep(5.0, 1.0, 0.9), TickStep(2.0, 1.0, 0.7), TickStep(2.5, 0.5, 0.5), TickStep(3.0, 1.0, 0.2)
]

@dataclass
class ChooseTicksParams:
    k_min: int
    k_max: int
    k_ideal: int
    granularity_weight: float
    simplicity_weight: float
    coverage_weight: float
    niceness_weight: float

DEFAULT_CHOOSE_TICKS_PARAMS = ChooseTicksParams(
    k_min=2,
    k_max=10,
    k_ideal=5,
    granularity_weight=1/4,
    simplicity_weight=1/6,
    coverage_weight=1/2,
    niceness_weight=1/4,
)

class TickCoverage(Enum):
    Flexible=1
    StrictSub=2
    StrictSuper=3


def _label_numbers(xs: np.ndarray) -> NDArray[np.str_]:
    MAX_PRECISION = 5
    fmt_str = f"{{:.{MAX_PRECISION}f}}"

    xstrs = [fmt_str.format(x) for x in xs]
    trim = min([len(xstr) - len(xstr.rstrip("0")) for xstr in xstrs])
    if trim == MAX_PRECISION:
        trim += 1

    # TODO: fall back on scientific numbers?
    # if trim === 0:
    #     pass

    return np.array([xstr[:-trim] for xstr in xstrs], dtype=str)

# TODO:
#  - Bijections (i.e. log scales)
#  - Fixed tick spans

class ScaleContinuous(Scale):
    unit: str
    min: Optional[np.float64]
    max: Optional[np.float64]
    tick_coverage: TickCoverage
    choose_ticks_params: ChooseTicksParams
    _ticks: Optional[np.ndarray]
    _subticks: Optional[np.ndarray]
    _tick_labels: Optional[np.ndarray]
    _subtick_labels: Optional[np.ndarray]

    def __init__(self, unit: str, tick_coverage: TickCoverage, choose_tick_params: ChooseTicksParams = DEFAULT_CHOOSE_TICKS_PARAMS):
        # TODO: We should be able to pass in min and max
        # and also have a `fixed` argument like discrete scales.

        self.unit = unit
        self.min = None
        self.max = None
        self.tick_coverage = tick_coverage
        self.choose_ticks_params = choose_tick_params
        self._ticks = None
        self._subticks = None
        self._tick_labels = None
        self._subtick_labels = None

    def _cast_value(self, value: Any) -> np.float64:
        try:
            return np.float64(value)
        except ValueError:
            raise ValueError(f"Cannot use continuous scale for unit '{self.unit}' with non-numerical value: {value}")

    def fit_values(self, values: UnscaledValues):
        for value in values.values:
            value = self._cast_value(value)
            if self.min is None:
                self.min = value
            elif value < self.min:
                self.min = value
            if self.max is None:
                self.max = value
            elif value > self.max:
                self.max = value

    def finalize(self):
        # TODO: Do I actually need to do anything here?
        pass

    def ticks(self) -> Tuple[NDArray[np.str_], Lengths | Colors]:
        if self._ticks is None or self._tick_labels is None:
            self._ticks, self._subticks = self._choose_ticks()
            self._tick_labels = _label_numbers(self._ticks)
            self._subtick_labels = _label_numbers(self._subticks)

        return (self._tick_labels, CtxLengths(self._ticks, self.unit, CtxLenType.Pos))

    def _choose_ticks(self):
        """
        Continuous scale tick optimization via a version of Wilkinson's ad-hoc scoring method.
        """

        if self.min is None or self.max is None:
            raise ValueError(f"Cannot choose ticks for unit {self.unit} with no min or max")

        scale_span = self.max - self.min

        if scale_span == 0.0:
            t0 = round(self.min - 1.0)
            t1 = round(self.min + 1.0)
            return np.array([t0, t1]), np.array([], dtype=float)

        params = self.choose_ticks_params
        CONSTRAINT_PENALTY = 10000.0
        high_score = -np.inf

        oom_best = 0.0
        k_best = 0
        t0_best = 0.0
        step_best = 0.0
        substep_best = 0.0

        # Consider all orders of magnitude where we can span the range with k_max ticks
        oom = np.ceil(np.log10(scale_span))
        while params.k_max * 10.0**(oom+1) > scale_span:
            # Consider numbers of ticks
            for k in range(params.k_min, params.k_max+1):
                # Consider steps
                for step in TICK_STEP_OPTIONS:
                    step_size = step.tick_step * 10.0**oom
                    if step_size == 0.0:
                        continue

                    t0 = step_size * np.floor(self.min / step_size)

                    # Consider tick starting places
                    while t0 <= self.max:
                        score = step.niceness * params.niceness_weight

                        tk = t0 + (k-1)*step_size

                        has_zero = t0 <= 0 and np.abs(t0/step_size) < k
                        if not has_zero:
                            score += params.simplicity_weight

                        if 0 < k and k < 2*params.k_ideal:
                            score += (1 - abs(k - params.k_ideal) / params.k_ideal) * params.granularity_weight

                        coverage_jaccard = (min(self.max, tk) - max(self.min, t0)) / (max(self.max, tk) - min(self.min, t0))
                        score += coverage_jaccard * params.coverage_weight

                        # strict-ish limits on coverage
                        if self.tick_coverage == TickCoverage.StrictSub and (t0 < self.min or tk > self.max):
                            score -= CONSTRAINT_PENALTY
                        elif self.tick_coverage == TickCoverage.StrictSuper and (t0 > self.min or tk < self.max):
                            score -= CONSTRAINT_PENALTY

                        if score > high_score:
                            high_score = score
                            oom_best = oom
                            k_best = k
                            t0_best = t0
                            step_best = step.tick_step
                            substep_best = step.subtick_step

                        t0 += step_size

            oom -= 1

        if not np.isfinite(high_score):
            t0 = round(self.min - 1.0)
            t1 = round(self.min + 1.0)
            return np.array([t0, t1]), np.array([], dtype=float)

        # ticks
        step_size = step_best * 10.0**oom_best
        ticks = np.zeros(k_best, dtype=float)
        for i in range(k_best):
            ticks[i] = t0_best + i*step_size

        # subticks
        k_sub = int(round((k_best - 1) * step_best/substep_best))
        step_size = substep_best * 10.0**oom_best
        subticks = np.zeros(k_sub, dtype=float)
        for i in range(k_sub):
            subticks[i] = t0_best + i*step_size

        return ticks, subticks

class ScaleContinuousColor(ScaleContinuous):
    def __init__(self, unit: str, colormap: ColormapLike, tick_coverage: TickCoverage, choose_tick_params: ChooseTicksParams = DEFAULT_CHOOSE_TICKS_PARAMS):
        self.colormap = Colormap(colormap)
        super().__init__(unit, tick_coverage, choose_tick_params)

    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        if self.min is None or self.max is None:
            raise ValueError("ScaleContinuousColor requires min and max values")

        span = self.max - self.min
        scaled_values = self.colormap(np.fromiter(
            ((self._cast_value(value) - self.min) / span for value in values.values),
            dtype=np.float64
        ))

        return Colors(scaled_values)

def colorcontinuous(*args, **kwargs) -> ScaleContinuousColor:
    return ScaleContinuousColor(*args, **kwargs)


class ScaleContinuousLength(ScaleContinuous):
    def __init__(self, unit: str, tick_coverage: TickCoverage, choose_tick_params: ChooseTicksParams = DEFAULT_CHOOSE_TICKS_PARAMS):
        super().__init__(unit, tick_coverage, choose_tick_params)

    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit

        scaled_values = np.fromiter(
            (self._cast_value(value) for value in values.values),
            dtype=np.float64
        )

        return CtxLengths(scaled_values, values.unit, values.typ)


def xcontinuous(*args, **kwargs) -> ScaleContinuousLength:
    return ScaleContinuousLength("x", *args, **kwargs)

def ycontinuous(*args, **kwargs) -> ScaleContinuousLength:
    return ScaleContinuousLength("y", *args, **kwargs)

def sizecontinuous(*args, **kwargs) -> ScaleContinuousLength:
    return ScaleContinuousLength("size", *args, **kwargs)

_default_scales = {
    "x": (xdiscrete, xcontinuous),
    "y": (ydiscrete, ycontinuous),
    "color": (colordiscrete, colorcontinuous),
    "size": (None, sizecontinuous),
}
