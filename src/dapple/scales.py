from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Number
from typing import Any, Callable, NamedTuple, TypeAlias, cast, override

import numpy as np
from cmap import Colormap, ColormapLike
from numpy.typing import NDArray

from .colors import Colors
from .config import ChooseTicksParams, ConfigKey
from .coordinates import AbsLengths, CtxLengths, CtxLenType, Lengths


class UnscaledExpr(ABC):
    """
    Unscaled expression generalized over unscaled values to support simple arthmetic
    expressions, e.g. ux(["a", "b", "c"]) + mm(1.0).

    Implementing this interface requires propogating visits from the scale in
    the fit and scale passes.
    """

    @abstractmethod
    def accept_fit(self, scaleset: ScaleSet):
        pass

    @abstractmethod
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        pass

    @abstractmethod
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]):
        pass

    def __add__(self, other: UnscaledExpr | Lengths) -> UnscaledBinaryOp:
        return UnscaledBinaryOp(self, other, operator.add)

    def __sub__(self, other: UnscaledExpr | Lengths) -> UnscaledBinaryOp:
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
    values: Iterable[Any]
    typ: CtxLenType = CtxLenType.Vec

    def __init__(self, unit: str, values: Any, typ: CtxLenType = CtxLenType.Vec):
        if not isinstance(values, Iterable) or isinstance(values, str):
            values = [values]
        self.unit = unit
        self.values = values
        self.typ = typ

    def __len__(self) -> int:
        return len(self.values)

    @override
    def accept_fit(self, scaleset: ScaleSet) -> None:
        scaleset[self.unit].fit_values(self)

    @override
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        return scaleset[self.unit].scale_values(self)

    @override
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]) -> None:
        visitor(self)

    def all_numeric(self) -> bool:
        if isinstance(self.values, np.ndarray):
            return issubclass(self.values.dtype.type, Number)
        else:
            for v in self.values:
                if not isinstance(v, Number):
                    return False
            return True


def length_params(
    unit: str, values: Any, typ: CtxLenType
) -> ConfigKey | Lengths | UnscaledExpr:
    if isinstance(values, (ConfigKey, Lengths, UnscaledExpr)):
        return values
    else:
        # If everything is a AbsLengths or CtxLengths of the same type, pass the
        # values through rather than try to scale them.
        if isinstance(values, Iterable) and not isinstance(values, str):
            nvalues = 0
            all_len = True
            all_abslen = True
            all_ctxlen = True
            last_ctxlen_unit = None
            last_ctxlen_typ = None
            for value in values:
                nvalues += 1

                if not isinstance(value, AbsLengths):
                    all_abslen = False
                    all_len = False

                if not isinstance(value, CtxLengths):
                    all_ctxlen = False
                    all_len = False
                else:
                    if last_ctxlen_unit is None:
                        last_ctxlen_unit = value.unit
                        last_ctxlen_typ = value.typ
                    elif value.unit != last_ctxlen_unit or value.typ != last_ctxlen_typ:
                        all_ctxlen = False

            if all_abslen and nvalues > 1:
                return AbsLengths(
                    np.concat([cast(AbsLengths, v).values for v in values])
                )

            if all_ctxlen and nvalues > 1:
                return CtxLengths(
                    np.concat(
                        [cast(CtxLengths, v).values for v in values],
                    ),
                    cast(str, last_ctxlen_unit),
                    cast(CtxLenType, last_ctxlen_typ),
                )

            if all_len and nvalues > 1:
                raise ValueError(
                    "Geometry length param values are all lengths, but mismatching types."
                )

        return UnscaledValues(unit, values, typ)


@dataclass
class ColorTransformExpr(UnscaledExpr):
    value: UnscaledExpr | Colors
    transform: Callable[[Colors], Colors]

    @override
    def accept_fit(self, scaleset: ScaleSet) -> None:
        if isinstance(self.value, UnscaledExpr):
            self.value.accept_fit(scaleset)

    @override
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        base = (
            self.value.accept_scale(scaleset)
            if isinstance(self.value, UnscaledExpr)
            else self.value
        )
        assert isinstance(base, Colors)
        return self.transform(base)

    @override
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]) -> None:
        if isinstance(self.value, UnscaledExpr):
            self.value.accept_visitor(visitor)


def _compose_color_transforms(
    existing: Callable[[Any], Any] | None,
    new: Callable[[Colors], Colors] | None,
) -> Callable[[Any], Any] | None:
    if existing is None:
        return new
    if new is None:
        return existing

    def composed(value: Any) -> Any:
        return new(existing(value))

    return composed


def color_params(
    unit: str,
    values: Any,
    transform: Callable[[Colors], Colors] | None = None,
) -> ConfigKey | Colors | UnscaledExpr:
    if isinstance(values, ConfigKey):
        if transform is None:
            return values
        composed = _compose_color_transforms(values.transform, transform)
        return replace(values, transform=composed)

    if isinstance(values, Colors):
        return transform(values) if transform is not None else values

    base: UnscaledExpr
    if isinstance(values, UnscaledExpr):
        base = values
    else:
        base = UnscaledValues(unit, values)

    if transform is None:
        return base

    return ColorTransformExpr(base, transform)


@dataclass
class UnscaledUnaryOp(UnscaledExpr):
    """
    General purpose unary operations on unscaled value expressions.
    """

    a: UnscaledExpr | Lengths
    op: Callable[..., Any]

    @override
    def accept_fit(self, scaleset: ScaleSet) -> None:
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_fit(scaleset)

    @override
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        return self.op(
            self.a.accept_scale(scaleset)
            if isinstance(self.a, UnscaledExpr)
            else self.a
        )

    @override
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]) -> None:
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_visitor(visitor)


@dataclass
class UnscaledBinaryOp(UnscaledExpr):
    """
    General purpose binary operations on unscaled value expressions.
    """

    a: UnscaledExpr | Lengths
    b: UnscaledExpr | Lengths
    op: Callable[..., Any]

    def __len__(self) -> int:
        return max(len(self.a), len(self.b))

    @override
    def accept_fit(self, scaleset: ScaleSet) -> None:
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_fit(scaleset)
        if isinstance(self.b, UnscaledExpr):
            self.b.accept_fit(scaleset)

    @override
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        return self.op(
            self.a.accept_scale(scaleset)
            if isinstance(self.a, UnscaledExpr)
            else self.a,
            self.b.accept_scale(scaleset)
            if isinstance(self.b, UnscaledExpr)
            else self.b,
        )

    @override
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]) -> None:
        if isinstance(self.a, UnscaledExpr):
            self.a.accept_visitor(visitor)
        if isinstance(self.b, UnscaledExpr):
            self.b.accept_visitor(visitor)


class Scale(ABC):
    """
    Scales take unscaled values and transform them into values in a plottable
    unit, such as colors or lengths.

    They also generate ticks to aide with drawing guides, and support a fitting
    stage where they can first visit all the unscaled values before deciding how
    to scale them.
    """

    @property
    @abstractmethod
    def unit(self) -> str:
        pass

    @abstractmethod
    def fit_values(self, values: UnscaledValues) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass

    @abstractmethod
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        pass

    @abstractmethod
    def ticks(self) -> tuple[NDArray[np.str_], Lengths | Colors]:
        pass


def _label_numbers(xs: np.ndarray) -> NDArray[np.str_]:
    """
    Format an array of numbers with consistent precision.
    Determines appropriate precision based on the differences between values.
    """
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


def default_labeler(values: Sequence[Any]) -> list[str]:
    """
    Default labeler function that converts a collection of values to strings.

    For numeric values, formats all numbers with matching precision.
    For other types, uses str() conversion.
    """
    if not values:
        return []

    # Check if all values are numbers
    all_numbers = all(isinstance(v, Number) for v in values)

    if all_numbers:
        # Convert to numpy array and use number labeling
        arr = np.array([float(v) for v in values], dtype=np.float64)
        return list(_label_numbers(arr))
    else:
        # Fall back to string conversion
        return [str(v) for v in values]


class ScaleDiscrete(Scale, ABC):
    """
    Disecrete scale which can map any collection of (hashable) values onto lengths or colors.
    """

    _unit: str
    fixed: bool
    labeler: Callable[[Sequence[Any]], list[str]]
    order_by: (
        Callable[[Sequence[Any]], Sequence[Any]]
        | Callable[[Sequence[Any], Sequence[Any | None]], Sequence[Any]]
        | None
    )
    map: dict[Any, int]
    targets: NDArray[np.float64]
    labels: NDArray[np.str_]

    # partial target list used while fitting the scale, maps
    # values to label target pairs.
    _targets: dict[Any, tuple[str, Any]]

    def __init__(
        self,
        unit: str,
        values: Mapping[Any, Any] | Sequence[Any] | None = None,
        fixed: bool = False,
        labeler: Callable[[Sequence[Any]], list[str]] = default_labeler,
        order_by: Callable[[Sequence[Any]], Sequence[Any]]
        | Callable[[Sequence[Any], Sequence[Any | None]], Sequence[Any]]
        | None = sorted,
    ):
        self._unit = unit
        self.fixed = fixed
        self.labeler = labeler
        self.order_by = order_by
        self._targets = dict()
        self.map = dict()

        if values is None:
            pass
        elif isinstance(values, Mapping):
            # Collect values that need labeling
            values_to_label = []
            value_target_pairs = []
            for value, target in values.items():
                if value in self.map:
                    raise ValueError(f"Duplicate value {value} in {values}")

                self.map[value] = len(self.map)
                match target:
                    case (label, target):
                        self._targets[value] = (label, target)
                    case str():
                        self._targets[value] = (target, None)
                    case _:
                        values_to_label.append(value)
                        value_target_pairs.append((value, target))

            # Label all values at once
            if values_to_label:
                labels = self.labeler(values_to_label)
                for (value, target), label in zip(value_target_pairs, labels):
                    self._targets[value] = (label, target)
        elif isinstance(values, Sequence):
            # Label all values at once
            if values:
                labels = self.labeler(list(values))
                for value, label in zip(values, labels):
                    if value in self._targets:
                        raise ValueError(f"Duplicate value {value} in {values}")
                    self._targets[value] = (label, None)
        else:
            raise TypeError("values must be a Mapping or Sequence")

    @property
    @override
    def unit(self) -> str:
        return self._unit

    @override
    def fit_values(self, values: UnscaledValues) -> None:
        # Collect new values that need labeling
        new_values = []
        for value in values.values:
            if value not in self._targets:
                if self.fixed:
                    raise ValueError(
                        f"Fixed scale cannot be updated with new value {value}"
                    )
                new_values.append(value)

        # Label all new values at once
        if new_values:
            labels = self.labeler(new_values)
            for value, label in zip(new_values, labels):
                self._targets[value] = (label, None)


class ScaleDiscreteLength(ScaleDiscrete):
    """
    Discrete length scale, which maps any collection of (hashable) values onto lengths.
    """

    def __init__(
        self,
        unit: str,
        values: Mapping[Any, Any] | Sequence[Any] | None = None,
        fixed: bool = False,
        labeler: Callable[[Sequence[Any]], list[str]] = default_labeler,
        order_by: Callable[[Sequence[Any]], Sequence[Any]]
        | Callable[[Sequence[Any], Sequence[Any | None]], Sequence[Any]]
        | None = sorted,
    ):
        super().__init__(unit, values, fixed, labeler, order_by)

    @override
    def finalize(self) -> None:
        if self.order_by is not None:
            vals_list = list(self._targets.keys())
            prelim_targets = [self._targets[v][1] for v in vals_list]
            try:
                values = self.order_by(vals_list, prelim_targets)  # type: ignore[misc]
            except TypeError:
                values = self.order_by(vals_list)
        else:
            values = self._targets.keys()

        self.targets = np.zeros(len(self._targets), dtype=np.float64)
        self.map: dict[Any, int] = dict()
        labels: list[str] = []

        next_target = max(
            filter(
                lambda target: target is not None,
                map(lambda v: v[1], self._targets.values()),
            ),
            default=0,
        )

        for i, value in enumerate(values):
            (label, target) = self._targets[value]
            self.map[value] = i
            labels.append(label)
            if target is None:
                self.targets[i] = next_target
                next_target += 1
            else:
                self.targets[i] = target

        self.labels = np.array(labels, dtype=np.str_)

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit
        indices = np.fromiter((self.map[value] for value in values.values), dtype=int)
        return CtxLengths(self.targets[indices], values.unit, values.typ)

    @override
    def ticks(self) -> tuple[NDArray[np.str_], CtxLengths]:
        return self.labels, CtxLengths(self.targets, self.unit, CtxLenType.Pos)


def xdiscrete(*args: Any, **kwargs: Any) -> ScaleDiscreteLength:
    return ScaleDiscreteLength("x", *args, **kwargs)


def ydiscrete(*args: Any, **kwargs: Any) -> ScaleDiscreteLength:
    return ScaleDiscreteLength("y", *args, **kwargs)


class ScaleDiscreteColor(ScaleDiscrete):
    colormap: Colormap | ConfigKey | Callable[[int], Colormap]

    def __init__(
        self,
        unit: str,
        colormap: ColormapLike | ConfigKey = ConfigKey("discrete_cmap"),
        values: Mapping[Any, Any] | Sequence[Any] | None = None,
        fixed: bool = False,
        labeler: Callable[[Sequence[Any]], list[str]] = default_labeler,
        order_by: Callable[[Sequence[Any]], Sequence[Any]]
        | Callable[[Sequence[Any], Sequence[Any | None]], Sequence[Any]]
        | None = sorted,
    ):
        if isinstance(colormap, ConfigKey):
            self.colormap = colormap
        else:
            self.colormap = Colormap(colormap)
        super().__init__(unit, values, fixed, labeler, order_by)

    @override
    def finalize(self) -> None:
        if self.order_by is not None:
            vals_list = list(self._targets.keys())
            prelim_targets = [self._targets[v][1] for v in vals_list]
            try:
                values = self.order_by(vals_list, prelim_targets)  # type: ignore[misc]
            except TypeError:
                values = self.order_by(vals_list)
        else:
            values = self._targets.keys()

        self.targets = np.zeros(len(self._targets), dtype=np.float64)
        self.map: dict[Any, int] = dict()
        labels: list[str] = []

        next_target = max(
            filter(
                lambda target: target is not None,
                map(lambda v: v[1], self._targets.values()),
            ),
            default=0,
        )

        for i, value in enumerate(values):
            (label, target) = self._targets[value]
            self.map[value] = i
            labels.append(label)
            if target is None:
                self.targets[i] = next_target
                next_target += 1
            else:
                assert target >= 0
                self.targets[i] = target

        if isinstance(self.colormap, Colormap):
            colormap = self.colormap
        elif isinstance(self.colormap, ConfigKey):
            # This shouldn't happen after finalize, but handle it gracefully
            raise ValueError(
                f"ConfigKey {self.colormap} was not resolved before finalize"
            )
        elif callable(self.colormap):
            ntargets = int(np.ceil(self.targets.max() + 1))
            colormap = self.colormap(ntargets)  # type: ignore[misc]
        else:
            colormap = Colormap(self.colormap)  # type: ignore[arg-type]

        assert isinstance(colormap, Colormap)

        # spacing depends on whether the colormap is cyclic or not
        c0 = np.asarray(colormap(0.0).rgba)
        c1 = np.asarray(colormap(1.0).rgba)
        iscyclic = np.sqrt(((c0 - c1) ** 2).sum()) < 1e-1
        if iscyclic:
            self.targets /= self.targets.max() + 1
        else:
            self.targets /= self.targets.max()

        self.targets = colormap(self.targets)
        self.labels = np.array(labels, dtype=np.str_)

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        indices = np.fromiter((self.map[value] for value in values.values), dtype=int)
        return Colors(self.targets[indices, :])

    @override
    def ticks(self) -> tuple[NDArray[np.str_], Colors]:
        return self.labels, Colors(self.targets)


def colordiscrete(*args: Any, **kwargs: Any) -> ScaleDiscreteColor:
    return ScaleDiscreteColor("color", *args, **kwargs)


class TickStep(NamedTuple):
    tick_step: float
    subtick_step: float
    niceness: float


TICK_STEP_OPTIONS = [
    TickStep(1.0, 0.5, 1.0),
    TickStep(5.0, 1.0, 0.9),
    TickStep(2.0, 1.0, 0.7),
    TickStep(2.5, 0.5, 0.5),
    TickStep(3.0, 1.0, 0.2),
]


class TickCoverage(Enum):
    Flexible = 1
    StrictSub = 2
    StrictSuper = 3

    @classmethod
    def from_str(cls, value: str) -> "TickCoverage":
        value = value.lower()
        if value == "flexible":
            return cls.Flexible
        elif value == "strictsub" or value == "sub":
            return cls.StrictSub
        elif value == "strictsuper" or value == "super":
            return cls.StrictSuper
        else:
            raise ValueError(f"Invalid tick coverage: {value}")


# TODO:
#  - Bijections (i.e. log scales)
#  - Fixed tick spans


class ScaleContinuous(Scale, ABC):
    _unit: str
    min: np.float64 | None
    max: np.float64 | None
    tick_coverage: TickCoverage | ConfigKey
    choose_ticks_params: ChooseTicksParams | ConfigKey
    _ticks: NDArray[np.float64] | None
    _subticks: NDArray[np.float64] | None
    _tick_labels: NDArray[np.str_] | None
    _subtick_labels: NDArray[np.str_] | None

    def __init__(
        self,
        unit: str,
        tick_coverage: TickCoverage | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
    ):
        # TODO: We should be able to pass in min and max
        # and also have a `fixed` argument like discrete scales.

        self._unit = unit
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
            raise ValueError(
                f"Cannot use continuous scale for unit '{self.unit}' with non-numerical value: {value}"
            )

    @property
    @override
    def unit(self) -> str:
        return self._unit

    @override
    def fit_values(self, values: UnscaledValues) -> None:
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

    @override
    def finalize(self) -> None:
        # TODO: Do I actually need to do anything here?
        pass

    @override
    def ticks(self) -> tuple[NDArray[np.str_], Lengths | Colors]:
        if self._ticks is None or self._tick_labels is None:
            self._ticks, self._subticks = self._choose_ticks()
            self._tick_labels = _label_numbers(self._ticks)
            self._subtick_labels = _label_numbers(self._subticks)

        return (self._tick_labels, CtxLengths(self._ticks, self.unit, CtxLenType.Pos))

    def _choose_ticks(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Continuous scale tick optimization via a version of Wilkinson's ad-hoc scoring method.
        """

        if self.min is None or self.max is None:
            raise ValueError(
                f"Cannot choose ticks for unit {self.unit} with no min or max"
            )

        scale_span = self.max - self.min

        if scale_span == 0.0:
            t0 = round(self.min - 1.0)
            t1 = round(self.min + 1.0)
            return np.array([t0, t1]), np.array([], dtype=float)

        assert isinstance(self.choose_ticks_params, ChooseTicksParams)
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
        while params.k_max * 10.0 ** (oom + 1) > scale_span:
            # Consider numbers of ticks
            for k in range(params.k_min, params.k_max + 1):
                # Consider steps
                for step in TICK_STEP_OPTIONS:
                    step_size = step.tick_step * 10.0**oom
                    if step_size == 0.0:
                        continue

                    t0 = step_size * np.floor(self.min / step_size)

                    # Consider tick starting places
                    while t0 <= self.max:
                        score = step.niceness * params.niceness_weight

                        tk = t0 + (k - 1) * step_size

                        has_zero = t0 <= 0 and np.abs(t0 / step_size) < k
                        if has_zero:
                            score += params.simplicity_weight

                        if 0 < k and k < 2 * params.k_ideal:
                            score += (
                                1 - abs(k - params.k_ideal) / params.k_ideal
                            ) * params.granularity_weight

                        coverage_jaccard = (min(self.max, tk) - max(self.min, t0)) / (
                            max(self.max, tk) - min(self.min, t0)
                        )
                        score += coverage_jaccard * params.coverage_weight

                        # strict-ish limits on coverage
                        if self.tick_coverage == TickCoverage.StrictSub and (
                            t0 < self.min or tk > self.max
                        ):
                            score -= CONSTRAINT_PENALTY
                        elif self.tick_coverage == TickCoverage.StrictSuper and (
                            t0 > self.min or tk < self.max
                        ):
                            score -= CONSTRAINT_PENALTY

                        if score > high_score:
                            high_score = score
                            oom_best = oom
                            k_best = k
                            t0_best = t0
                            step_best = step.tick_step
                            substep_best = step.subtick_step

                        t0 += step_size / 2

            oom -= 1

        if not np.isfinite(high_score):
            t0 = round(self.min - 1.0)
            t1 = round(self.min + 1.0)
            return np.array([t0, t1]), np.array([], dtype=float)

        # ticks
        step_size = step_best * 10.0**oom_best
        ticks = np.zeros(k_best, dtype=float)
        for i in range(k_best):
            ticks[i] = t0_best + i * step_size

        # subticks
        k_sub = int(round((k_best - 1) * step_best / substep_best))
        step_size = substep_best * 10.0**oom_best
        subticks = np.zeros(k_sub, dtype=float)
        for i in range(k_sub):
            subticks[i] = t0_best + i * step_size

        return ticks, subticks


class ScaleContinuousColor(ScaleContinuous):
    colormap: Colormap | ConfigKey

    def __init__(
        self,
        unit: str,
        colormap: ColormapLike | ConfigKey = ConfigKey("continuous_cmap"),  # type: ignore[assignment]
        tick_coverage: TickCoverage | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
    ):
        if not isinstance(colormap, ConfigKey):
            colormap = Colormap(colormap)
        self.colormap = colormap
        super().__init__(unit, tick_coverage, choose_tick_params)

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        if self.min is None or self.max is None:
            raise ValueError("ScaleContinuousColor requires min and max values")

        assert isinstance(self.colormap, Colormap), (
            f"Expected Colormap but got {type(self.colormap)}"
        )

        span = self.max - self.min
        scaled_values = self.colormap(
            np.fromiter(
                (
                    (self._cast_value(value) - self.min) / span
                    for value in values.values
                ),
                dtype=np.float64,
            )
        )

        return Colors(scaled_values)


def colorcontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousColor:
    return ScaleContinuousColor(*args, **kwargs)


class ScaleContinuousLength(ScaleContinuous):
    def __init__(
        self,
        unit: str,
        tick_coverage: TickCoverage | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
    ):
        super().__init__(unit, tick_coverage, choose_tick_params)

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit

        scaled_values = np.fromiter(
            (self._cast_value(value) for value in values.values), dtype=np.float64
        )

        return CtxLengths(scaled_values, values.unit, values.typ)


def xcontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousLength:
    return ScaleContinuousLength("x", *args, **kwargs)


def ycontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousLength:
    return ScaleContinuousLength("y", *args, **kwargs)


def sizecontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousLength:
    return ScaleContinuousLength("size", *args, **kwargs)


_default_scales: dict[str, tuple[Callable[..., Scale] | None, Callable[..., Scale]]] = {
    "x": (xdiscrete, xcontinuous),
    "y": (ydiscrete, ycontinuous),
    "color": (colordiscrete, colorcontinuous),
    "size": (None, sizecontinuous),
}

ScaleSet: TypeAlias = dict[str, Scale]
