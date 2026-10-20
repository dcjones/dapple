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
from numpy.typing import ArrayLike, NDArray

from .colors import Colors
from .shapes import NUM_SHAPES, Shapes
from .config import ChooseTicksParams, ConfigKey
from .coordinates import (
    AbsLengths,
    CtxLengths,
    CtxLenType,
    Lengths,
    LengthsSequence,
    mm,
)


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


def shape_params(unit: str, values: Any) -> ConfigKey | Shapes | UnscaledExpr:
    """
    Wrap raw shape data so it can be discovered and mapped by a shape scale.

    Mirrors :func:`color_params`, but shapes have no transforms.
    """
    if isinstance(values, (ConfigKey, Shapes, UnscaledExpr)):
        return values
    return UnscaledValues(unit, values)


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
    """
    Construct a discrete x scale, mapping categorical values onto evenly spaced
    x positions.

    Args:
        values: Optionally fixes the values the scale maps, and their order. A
            sequence gives the values in the order they should appear; a mapping
            gives, for each value, either a label, a target, or a
            (label, target) pair.
        fixed: When true, raise an error if the data contains a value not listed
            in `values`, rather than extending the scale.
        labeler: Function mapping a sequence of values to their label strings.
        order_by: Function used to order the values. Called with the values, or
            with the values and their preliminary targets, if it accepts two
            arguments. Pass None to keep insertion order.

    Returns:
        A `ScaleDiscreteLength` to be passed to `plot`.
    """
    return ScaleDiscreteLength("x", *args, **kwargs)


def ydiscrete(*args: Any, **kwargs: Any) -> ScaleDiscreteLength:
    """
    Construct a discrete y scale, mapping categorical values onto evenly spaced
    y positions.

    Args:
        values: Optionally fixes the values the scale maps, and their order. A
            sequence gives the values in the order they should appear; a mapping
            gives, for each value, either a label, a target, or a
            (label, target) pair.
        fixed: When true, raise an error if the data contains a value not listed
            in `values`, rather than extending the scale.
        labeler: Function mapping a sequence of values to their label strings.
        order_by: Function used to order the values. Called with the values, or
            with the values and their preliminary targets, if it accepts two
            arguments. Pass None to keep insertion order.

    Returns:
        A `ScaleDiscreteLength` to be passed to `plot`.
    """
    return ScaleDiscreteLength("y", *args, **kwargs)


class ScaleDiscreteColor(ScaleDiscrete):
    colormap: Colormap | ConfigKey | Callable[[int], Colormap]

    def __init__(
        self,
        unit: str,
        colormap: Callable | ColormapLike | ConfigKey = ConfigKey("discrete_cmap"),
        values: Mapping[Any, Any] | Sequence[Any] | None = None,
        fixed: bool = False,
        labeler: Callable[[Sequence[Any]], list[str]] = default_labeler,
        order_by: Callable[[Sequence[Any]], Sequence[Any]]
        | Callable[[Sequence[Any], Sequence[Any | None]], Sequence[Any]]
        | None = sorted,
    ):
        if isinstance(colormap, ConfigKey) or callable(colormap):
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
    """
    Construct a discrete color scale, mapping categorical values onto colors
    drawn from a colormap.

    Args:
        colormap: A colormap, a sequence of colors, or a callable taking the
            number of distinct values and returning a colormap (e.g.
            `distinguishable_colors`). Defaults to the config's
            "discrete_cmap".
        values: Optionally fixes the values the scale maps, and their order. A
            sequence gives the values in the order they should appear; a mapping
            gives, for each value, either a label, a target, or a
            (label, target) pair.
        fixed: When true, raise an error if the data contains a value not listed
            in `values`, rather than extending the scale.
        labeler: Function mapping a sequence of values to their label strings.
        order_by: Function used to order the values. Called with the values, or
            with the values and their preliminary targets, if it accepts two
            arguments. Pass None to keep insertion order.

    Returns:
        A `ScaleDiscreteColor` to be passed to `plot`.
    """
    return ScaleDiscreteColor("color", *args, **kwargs)


class ScaleDiscreteShape(ScaleDiscrete):
    """
    Discrete shape scale, mapping any collection of (hashable) values onto marker
    shapes. There are a fixed number of shapes (:data:`~dapple.shapes.NUM_SHAPES`);
    if there are more distinct values than shapes, indices wrap around.
    """

    def __init__(
        self,
        unit: str = "shape",
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
        self.map = dict()
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
        shape_indices = self.targets[indices].astype(int) % NUM_SHAPES
        return cast(Any, Shapes(shape_indices))

    @override
    def ticks(self) -> tuple[NDArray[np.str_], Any]:
        return self.labels, Shapes(self.targets.astype(int) % NUM_SHAPES)


def shapediscrete(*args: Any, **kwargs: Any) -> ScaleDiscreteShape:
    """
    Construct a discrete shape scale, mapping categorical values onto marker
    shapes.

    Args:
        values: Optionally fixes the values the scale maps, and their order. A
            sequence gives the values in the order they should appear; a mapping
            gives, for each value, either a label, a target, or a
            (label, target) pair.
        fixed: When true, raise an error if the data contains a value not listed
            in `values`, rather than extending the scale.
        labeler: Function mapping a sequence of values to their label strings.
        order_by: Function used to order the values. Called with the values, or
            with the values and their preliminary targets, if it accepts two
            arguments. Pass None to keep insertion order.

    Returns:
        A `ScaleDiscreteShape` to be passed to `plot`.
    """
    return ScaleDiscreteShape("shape", *args, **kwargs)


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
    """
    How the span of ticks on a continuous scale may relate to the data range.

    Attributes:
        Flexible: Ticks may fall slightly inside or outside the data range,
            whichever gives nicer tick values.
        StrictSub: Ticks must lie within the data range.
        StrictSuper: Ticks must cover the data range.
    """

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
            raise ValueError(
                f"Invalid tick coverage: {value}. Expected one of "
                '"flexible", "sub", "strictsub", "super", or "strictsuper".'
            )

    @classmethod
    def parse(cls, value: "TickCoverage | str") -> "TickCoverage":
        """Accept either a `TickCoverage` or its string shorthand."""
        if isinstance(value, TickCoverage):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        raise TypeError(
            f"Tick coverage must be a TickCoverage or string, not {type(value).__name__}"
        )


_TICK_CONSTRAINT_PENALTY = 10000.0


def _tick_score(
    t0: float,
    tk: float,
    k: int,
    niceness: float,
    includes_origin: bool,
    dmin: float,
    dmax: float,
    coverage: TickCoverage,
    params: ChooseTicksParams,
) -> float:
    """
    Score a candidate run of ``k`` ticks spanning ``[t0, tk]`` against the data
    range ``[dmin, dmax]``, following Wilkinson's ad-hoc scoring method.

    All positions must be in a space where ticks are compared linearly (e.g.
    log space for log ticks). ``includes_origin`` rewards ticks containing a
    natural reference point, such as zero on a linear scale.
    """
    score = niceness * params.niceness_weight

    if includes_origin:
        score += params.simplicity_weight

    if 0 < k and k < 2 * params.k_ideal:
        score += (
            1 - abs(k - params.k_ideal) / params.k_ideal
        ) * params.granularity_weight

    coverage_jaccard = (min(dmax, tk) - max(dmin, t0)) / (max(dmax, tk) - min(dmin, t0))
    score += coverage_jaccard * params.coverage_weight

    # strict-ish limits on coverage
    if coverage == TickCoverage.StrictSub and (t0 < dmin or tk > dmax):
        score -= _TICK_CONSTRAINT_PENALTY
    elif coverage == TickCoverage.StrictSuper and (t0 > dmin or tk < dmax):
        score -= _TICK_CONSTRAINT_PENALTY

    return score


def linear_ticks(
    dmin: float,
    dmax: float,
    coverage: TickCoverage,
    params: ChooseTicksParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Choose evenly spaced ticks for the range ``[dmin, dmax]`` via a version of
    Wilkinson's ad-hoc scoring method.

    Returns:
        A tuple ``(ticks, subticks)``.
    """
    scale_span = dmax - dmin

    if scale_span == 0.0:
        t0 = round(dmin - 1.0)
        t1 = round(dmin + 1.0)
        return np.array([t0, t1], dtype=float), np.array([], dtype=float)

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

                t0 = step_size * np.floor(dmin / step_size)

                # Consider tick starting places
                while t0 <= dmax:
                    tk = t0 + (k - 1) * step_size
                    has_zero = t0 <= 0 and np.abs(t0 / step_size) < k
                    score = _tick_score(
                        t0, tk, k, step.niceness, has_zero, dmin, dmax, coverage, params
                    )

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
        t0 = round(dmin - 1.0)
        t1 = round(dmin + 1.0)
        return np.array([t0, t1], dtype=float), np.array([], dtype=float)

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


def _linear_ticks_in_domain(
    dmin: float,
    dmax: float,
    coverage: TickCoverage,
    params: ChooseTicksParams,
    in_domain: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Linear ticks, discarding any that fall outside a transform's domain.

    Falls back on strictly interior ticks, then on the range endpoints, if too
    few ticks survive.
    """
    for cov in (coverage, TickCoverage.StrictSub):
        ticks, subticks = linear_ticks(dmin, dmax, cov, params)
        ticks = ticks[in_domain(ticks)]
        if len(ticks) >= 2:
            return ticks, subticks[in_domain(subticks)]
    return np.array([dmin, dmax], dtype=float), np.array([], dtype=float)


class LogTickSequence(NamedTuple):
    """
    A family of candidate log-scale ticks: ``mantissa * tick_base**exponent``
    for every exponent that is a multiple of ``exponent_step``.
    """

    mantissas: tuple[float, ...]
    exponent_step: int
    niceness: float


LOG10_TICK_OPTIONS = [
    LogTickSequence((1.0,), 1, 1.0),
    LogTickSequence((1.0, 2.0, 5.0), 1, 0.7),
    LogTickSequence((1.0, 3.0), 1, 0.5),
    LogTickSequence((1.0,), 2, 0.8),
    LogTickSequence((1.0,), 5, 0.7),
    LogTickSequence((1.0,), 3, 0.5),
    LogTickSequence((1.0,), 10, 0.6),
    LogTickSequence((1.0,), 20, 0.5),
    LogTickSequence((1.0,), 50, 0.4),
    LogTickSequence((1.0,), 100, 0.3),
]

LOG2_TICK_OPTIONS = [
    LogTickSequence((1.0,), 1, 1.0),
    LogTickSequence((1.0,), 2, 0.8),
    LogTickSequence((1.0,), 4, 0.6),
    LogTickSequence((1.0,), 5, 0.5),
    LogTickSequence((1.0,), 10, 0.5),
    LogTickSequence((1.0,), 20, 0.4),
    LogTickSequence((1.0,), 50, 0.3),
    LogTickSequence((1.0,), 100, 0.3),
]


def _log_base(x: Any, base: int) -> Any:
    """Logarithm that is exact for powers of 10 and 2."""
    if base == 10:
        return np.log10(x)
    if base == 2:
        return np.log2(x)
    return np.log(x) / np.log(base)


def _log_tick_values(
    mantissas: tuple[float, ...], tick_base: int, exponents: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(values, log_positions)`` for each mantissa/exponent pair."""
    e = np.repeat(exponents, len(mantissas)).astype(np.float64)
    m = np.tile(np.asarray(mantissas, dtype=np.float64), len(exponents))
    # Dividing by a positive power is more accurate than multiplying by a
    # negative one (e.g. 3 / 1000 == 0.003 but 3 * 10**-3 != 0.003).
    values = np.where(e >= 0, m * float(tick_base) ** e, m / float(tick_base) ** -e)
    positions = e + _log_base(m, tick_base)
    return values, positions


def log_ticks(
    dmin: float,
    dmax: float,
    coverage: TickCoverage,
    params: ChooseTicksParams,
    tick_base: int = 10,
    options: Sequence[LogTickSequence] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Choose ticks for a logarithmic scale over the positive range ``[dmin, dmax]``.

    Candidates are runs of powers of ``tick_base`` (possibly skipping some, or
    with intermediate mantissas like 2 and 5), scored in log space with the same
    criteria as `linear_ticks`. Ranges spanning less than one power of
    ``tick_base`` fall back on linear ticks.

    Returns:
        A tuple ``(ticks, subticks)`` in data space.
    """
    if options is None:
        options = LOG10_TICK_OPTIONS if tick_base == 10 else LOG2_TICK_OPTIONS

    def positive(x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return x > 0

    lmin = float(_log_base(np.float64(dmin), tick_base))
    lmax = float(_log_base(np.float64(dmax), tick_base))

    if lmax <= lmin:
        e = np.floor(lmin)
        return (
            np.array([float(tick_base) ** e, float(tick_base) ** (e + 1)]),
            np.array([], dtype=float),
        )

    if lmax - lmin < 1.0:
        return _linear_ticks_in_domain(dmin, dmax, coverage, params, positive)

    # Tolerate rounding error from round-tripping the range through the
    # transform, so ticks landing exactly on the data range aren't rejected.
    eps = 1e-9 * (lmax - lmin)
    lmin -= eps
    lmax += eps

    high_score = -np.inf
    best: tuple[NDArray[np.float64], LogTickSequence] | None = None

    for option in options:
        step = option.exponent_step
        e_lo = int(np.floor(lmin / step)) - 1
        e_hi = int(np.ceil(lmax / step)) + 1
        exponents = np.arange(e_lo, e_hi + 1, dtype=np.int64) * step
        values, positions = _log_tick_values(option.mantissas, tick_base, exponents)

        n = len(values)
        for i in range(n):
            t0 = float(positions[i])
            if t0 > lmax:
                break
            for k in range(params.k_min, params.k_max + 1):
                if i + k > n:
                    break
                tk = float(positions[i + k - 1])
                includes_one = t0 <= 0.0 <= tk
                score = _tick_score(
                    t0,
                    tk,
                    k,
                    option.niceness,
                    includes_one,
                    lmin,
                    lmax,
                    coverage,
                    params,
                )
                if score > high_score:
                    high_score = score
                    best = (values[i : i + k], option)

    if best is None:
        return _linear_ticks_in_domain(dmin, dmax, coverage, params, positive)

    ticks, option = best

    # Subticks: every power when powers are skipped, otherwise the full set of
    # integer multiples within each power.
    if option.exponent_step > 1 or tick_base != 10:
        sub_mantissas: tuple[float, ...] = (1.0,)
    else:
        sub_mantissas = tuple(float(m) for m in range(1, 10))
    e0 = int(np.floor(_log_base(ticks[0], tick_base)))
    e1 = int(np.ceil(_log_base(ticks[-1], tick_base)))
    subticks, _ = _log_tick_values(
        sub_mantissas, tick_base, np.arange(e0, e1 + 1, dtype=np.int64)
    )
    eps = 1e-9 * (ticks[-1] - ticks[0])
    subticks = subticks[(subticks >= ticks[0] - eps) & (subticks <= ticks[-1] + eps)]

    return ticks.copy(), subticks


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _label_log_numbers(xs: NDArray[np.float64], base: int = 10) -> NDArray[np.str_]:
    """
    Label ticks on a log scale. Values of moderate magnitude are written out in
    full, otherwise every label is written in exponential form, e.g. ``10⁻⁴``
    or ``2×10⁻⁴``.
    """
    nonzero = np.abs(xs[xs != 0])
    if len(nonzero) == 0 or (np.all(nonzero >= 1e-3) and np.all(nonzero <= 1e4)):
        return np.array(
            [np.format_float_positional(float(x), trim="-") for x in xs], dtype=str
        )

    labels: list[str] = []
    for x in xs:
        x = float(x)
        if x == 0:
            labels.append("0")
            continue
        e = int(np.floor(_log_base(abs(x), base) + 1e-9))
        mantissa = x / float(base) ** e
        power = f"{base}{str(e).translate(_SUPERSCRIPTS)}"
        if np.isclose(mantissa, 1.0):
            labels.append(power)
        elif np.isclose(mantissa, -1.0):
            labels.append(f"-{power}")
        else:
            labels.append(f"{mantissa:.3g}×{power}")
    return np.array(labels, dtype=str)


class Transform(ABC):
    """
    A monotonically increasing transformation applied by a continuous scale
    before it maps values linearly onto lengths, colors, or sizes.

    Besides the forward mapping, a transform decides where ticks are placed
    and how they are labeled, since nice ticks depend on the transformation
    (e.g. powers of ten on a log scale).
    """

    name: str
    domain: str

    @abstractmethod
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map data values into transformed space."""

    @abstractmethod
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map transformed values back into data space."""

    @abstractmethod
    def in_domain(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test which data values the transform is defined for."""

    @abstractmethod
    def ticks(
        self,
        dmin: float,
        dmax: float,
        coverage: TickCoverage,
        params: ChooseTicksParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Choose ``(ticks, subticks)`` in data space for the data-space range
        ``[dmin, dmax]``.
        """

    def labels(self, ticks: NDArray[np.float64]) -> NDArray[np.str_]:
        """Label ticks given in data space."""
        return _label_numbers(ticks)


class IdentityTransform(Transform):
    """The default transform, leaving values unchanged."""

    name = "identity"
    domain = "all real numbers"

    @override
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    @override
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return y

    @override
    def in_domain(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.ones(np.shape(x), dtype=bool)

    @override
    def ticks(
        self,
        dmin: float,
        dmax: float,
        coverage: TickCoverage,
        params: ChooseTicksParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return linear_ticks(dmin, dmax, coverage, params)


class LogTransform(Transform):
    """
    Logarithmic transform. Ticks are placed at nice values in powers of
    ``tick_base``, independent of the logarithm's base.
    """

    domain = "positive numbers"

    def __init__(self, base: float = 10.0, tick_base: int = 10, name: str = "log10"):
        self.base = base
        self.tick_base = tick_base
        self.name = name

    @override
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.base == 10.0:
            return np.log10(x)
        if self.base == 2.0:
            return np.log2(x)
        return np.log(x) / np.log(self.base)

    @override
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.power(self.base, y)

    @override
    def in_domain(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return x > 0

    @override
    def ticks(
        self,
        dmin: float,
        dmax: float,
        coverage: TickCoverage,
        params: ChooseTicksParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return log_ticks(dmin, dmax, coverage, params, tick_base=self.tick_base)

    @override
    def labels(self, ticks: NDArray[np.float64]) -> NDArray[np.str_]:
        return _label_log_numbers(ticks, self.tick_base)


class Log1pTransform(Transform):
    """
    The transform ``log(1 + x)``, useful for counts that include zero.
    """

    name = "log1p"
    domain = "numbers greater than -1"

    @override
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.log1p(x)

    @override
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.expm1(y)

    @override
    def in_domain(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return x > -1

    @override
    def ticks(
        self,
        dmin: float,
        dmax: float,
        coverage: TickCoverage,
        params: ChooseTicksParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if dmax < 10.0:
            return _linear_ticks_in_domain(dmin, dmax, coverage, params, self.in_domain)

        ticks, subticks = log_ticks(max(dmin, 1.0), dmax, coverage, params)
        # Zero is the natural lower tick when the data starts at or near it.
        near_zero = dmin <= 0.0 or (coverage != TickCoverage.StrictSub and dmin < 0.5)
        if near_zero and ticks[0] > 0.0:
            ticks = np.concatenate([[0.0], ticks])
        return ticks, subticks

    @override
    def labels(self, ticks: NDArray[np.float64]) -> NDArray[np.str_]:
        return _label_log_numbers(ticks)


class SqrtTransform(Transform):
    """Square root transform, with ticks at evenly spaced data values."""

    name = "sqrt"
    domain = "non-negative numbers"

    @override
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(x)

    @override
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.square(y)

    @override
    def in_domain(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return x >= 0

    @override
    def ticks(
        self,
        dmin: float,
        dmax: float,
        coverage: TickCoverage,
        params: ChooseTicksParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _linear_ticks_in_domain(dmin, dmax, coverage, params, self.in_domain)


_TRANSFORMS: dict[str, Transform] = {
    "identity": IdentityTransform(),
    "log10": LogTransform(10.0, 10, "log10"),
    "log2": LogTransform(2.0, 2, "log2"),
    "log": LogTransform(float(np.e), 10, "log"),
    "ln": LogTransform(float(np.e), 10, "ln"),
    "log1p": Log1pTransform(),
    "sqrt": SqrtTransform(),
}


def resolve_transform(transform: Transform | str) -> Transform:
    """
    Look up a transform by name (one of "identity", "log10", "log2", "log",
    "ln", "log1p", "sqrt"), or pass a `Transform` instance through.
    """
    if isinstance(transform, Transform):
        return transform
    try:
        return _TRANSFORMS[transform]
    except KeyError:
        raise ValueError(
            f"Unknown transform '{transform}'. Expected one of: "
            + ", ".join(_TRANSFORMS)
        ) from None


TickSpec: TypeAlias = ArrayLike | Callable[[float, float], ArrayLike]
TickLabeler: TypeAlias = Callable[[NDArray[np.float64]], Sequence[str]]


class ScaleContinuous(Scale, ABC):
    """
    Base class for scales mapping numbers onto lengths, colors, or sizes.

    Values first pass through the scale's `Transform` (the identity by
    default). The scale's ``min``, ``max``, and tick positions are all kept in
    transformed space, while ticks are chosen and labeled in data space.
    """

    _unit: str
    min: np.float64 | None
    max: np.float64 | None
    transform: Transform
    tick_spec: TickSpec | None
    labeler: TickLabeler | None
    tick_coverage: TickCoverage | str | ConfigKey
    choose_ticks_params: ChooseTicksParams | ConfigKey
    _ticks: NDArray[np.float64] | None
    _subticks: NDArray[np.float64] | None
    _tick_labels: NDArray[np.str_] | None
    _subtick_labels: NDArray[np.str_] | None

    def __init__(
        self,
        unit: str,
        tick_coverage: TickCoverage | str | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
        min: float | None = None,
        max: float | None = None,
        *,
        transform: Transform | str = "identity",
        ticks: TickSpec | None = None,
        labeler: TickLabeler | None = None,
    ):
        # TODO: and also have a `fixed` argument like discrete scales.

        self._unit = unit
        self.transform = resolve_transform(transform)
        self.min = self._transform_scalar(min) if min is not None else None
        self.max = self._transform_scalar(max) if max is not None else None
        self.tick_spec = ticks
        self.labeler = labeler
        # Strings given directly are parsed now so mistakes surface early;
        # those from the config are parsed once the ConfigKey is replaced.
        self.tick_coverage = (
            TickCoverage.parse(tick_coverage)
            if isinstance(tick_coverage, str)
            else tick_coverage
        )
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

    def _as_array(self, values: UnscaledValues) -> NDArray[np.float64]:
        vals = values.values
        if isinstance(vals, np.ndarray):
            try:
                return vals.astype(np.float64)
            except ValueError:
                raise ValueError(
                    f"Cannot use continuous scale for unit '{self.unit}' with non-numerical values"
                )
        return np.fromiter(
            (self._cast_value(value) for value in vals), dtype=np.float64
        )

    def _check_domain(self, vals: NDArray[np.float64]) -> None:
        bad = ~(self.transform.in_domain(vals) | np.isnan(vals))
        if np.any(bad):
            raise ValueError(
                f"Scale for unit '{self.unit}' with transform '{self.transform.name}' "
                f"is only defined for {self.transform.domain}, but got {vals[bad][0]}"
            )

    def _transform_values(self, vals: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map data values into transformed space, checking they're in its domain."""
        self._check_domain(vals)
        return self.transform.forward(vals)

    def _transform_scalar(self, value: float) -> np.float64:
        return np.float64(
            self._transform_values(np.array([value], dtype=np.float64))[0]
        )

    @property
    @override
    def unit(self) -> str:
        return self._unit

    @override
    def fit_values(self, values: UnscaledValues) -> None:
        vals = self._as_array(values)
        if vals.size == 0:
            return
        tvals = self._transform_values(vals)
        vmin = tvals.min()
        vmax = tvals.max()
        if self.min is None or vmin < self.min:
            self.min = vmin
        if self.max is None or vmax > self.max:
            self.max = vmax

    @override
    def finalize(self) -> None:
        # TODO: Do I actually need to do anything here?
        pass

    @override
    def ticks(self) -> tuple[NDArray[np.str_], Lengths | Colors]:
        if self._ticks is None or self._tick_labels is None:
            ticks, subticks = self._choose_ticks()
            self._tick_labels = self._label_ticks(ticks)
            self._subtick_labels = self._label_ticks(subticks)
            self._ticks = self.transform.forward(ticks)
            self._subticks = self.transform.forward(subticks)

        return (self._tick_labels, CtxLengths(self._ticks, self.unit, CtxLenType.Pos))

    def _label_ticks(self, ticks: NDArray[np.float64]) -> NDArray[np.str_]:
        if len(ticks) == 0:
            return np.array([], dtype=str)
        if self.labeler is not None:
            return np.asarray(self.labeler(ticks), dtype=str)
        return self.transform.labels(ticks)

    def _choose_ticks(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Choose ``(ticks, subticks)`` in data space, either from the scale's
        explicit tick specification or using the transform's tick finder.
        """
        if self.min is None or self.max is None:
            raise ValueError(
                f"Cannot choose ticks for unit {self.unit} with no min or max"
            )

        dmin, dmax = sorted(
            float(v) for v in self.transform.inverse(np.array([self.min, self.max]))
        )

        if self.tick_spec is not None:
            spec = self.tick_spec
            raw = spec(dmin, dmax) if callable(spec) else spec
            ticks = np.sort(np.asarray(raw, dtype=np.float64).ravel())
            self._check_domain(ticks)
            return ticks, np.array([], dtype=np.float64)

        assert isinstance(self.choose_ticks_params, ChooseTicksParams)
        assert not isinstance(self.tick_coverage, ConfigKey)
        self.tick_coverage = TickCoverage.parse(self.tick_coverage)
        return self.transform.ticks(
            dmin,
            dmax,
            self.tick_coverage,
            self.choose_ticks_params,
        )


class ScaleContinuousColor(ScaleContinuous):
    colormap: Colormap | ConfigKey
    mid: np.float64 | None

    def __init__(
        self,
        unit: str,
        colormap: ColormapLike | ConfigKey = ConfigKey("continuous_cmap"),  # type: ignore[assignment]
        tick_coverage: TickCoverage | str | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
        min: float | None = None,
        max: float | None = None,
        mid: float | None = None,
        *,
        transform: Transform | str = "identity",
        ticks: TickSpec | None = None,
        labeler: TickLabeler | None = None,
    ):
        if not isinstance(colormap, ConfigKey):
            colormap = Colormap(colormap)
        self.colormap = colormap
        super().__init__(
            unit,
            tick_coverage,
            choose_tick_params,
            min=min,
            max=max,
            transform=transform,
            ticks=ticks,
            labeler=labeler,
        )
        self.mid = self._transform_scalar(mid) if mid is not None else None
        self._apply_mid()

    @override
    def fit_values(self, values: UnscaledValues) -> None:
        super().fit_values(values)
        self._apply_mid()

    def _apply_mid(self) -> None:
        if self.mid is not None and self.min is not None and self.max is not None:
            dist = max(abs(self.max - self.mid), abs(self.min - self.mid))
            self.min = self.mid - dist
            self.max = self.mid + dist

    def normalize(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Map values in the scale's (transformed) space onto colormap positions,
        where ``min`` maps to 0 and ``max`` maps to 1.
        """
        if self.min is None or self.max is None:
            raise ValueError("ScaleContinuousColor requires min and max values")
        span = self.max - self.min
        if span == 0:
            return np.zeros_like(values, dtype=np.float64)
        return (values - self.min) / span

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        if self.min is None or self.max is None:
            raise ValueError("ScaleContinuousColor requires min and max values")

        assert isinstance(self.colormap, Colormap), (
            f"Expected Colormap but got {type(self.colormap)}"
        )

        vals_arr = self._transform_values(self._as_array(values))
        return Colors(self.colormap(self.normalize(vals_arr)))


def colorcontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousColor:
    """
    Construct a continuous color scale, mapping numbers onto colors by
    interpolating a colormap.

    Args:
        colormap: A colormap or sequence of colors to interpolate
            (default: taken from the config's "continuous_cmap").
        tick_coverage: A `TickCoverage`, or one of the strings "flexible",
            "sub", or "super", controlling whether ticks may extend beyond
            the data range (default: taken from the config).
        choose_tick_params: `ChooseTicksParams` tuning tick selection
            (default: taken from the config).
        min: Lower end of the scale. Defaults to the minimum of the data.
        max: Upper end of the scale. Defaults to the maximum of the data.
        mid: If given, the value placed at the middle of the colormap, with
            `min` and `max` expanded symmetrically about it. Useful for
            diverging colormaps. Given in data space, so with a log
            transform it is the geometric midpoint.
        transform: A `Transform`, or the name of one ("identity", "log10",
            "log2", "log", "ln", "log1p", "sqrt"), applied to values before
            they are mapped. Ticks are placed and labeled to suit it.
        ticks: Explicit tick values in data space, or a function taking the
            data-space ``(min, max)`` and returning them. Defaults to ticks
            chosen by the transform.
        labeler: Function mapping an array of data-space tick values to
            label strings. Defaults to the transform's labeler.

    Returns:
        A `ScaleContinuousColor` to be passed to `plot`.
    """
    return ScaleContinuousColor("color", *args, **kwargs)


class ScaleContinuousLength(ScaleContinuous):
    def __init__(
        self,
        unit: str,
        tick_coverage: TickCoverage | str | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
        min: float | None = None,
        max: float | None = None,
        *,
        transform: Transform | str = "identity",
        ticks: TickSpec | None = None,
        labeler: TickLabeler | None = None,
    ):
        super().__init__(
            unit,
            tick_coverage,
            choose_tick_params,
            min=min,
            max=max,
            transform=transform,
            ticks=ticks,
            labeler=labeler,
        )

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit
        scaled_values = self._transform_values(self._as_array(values))
        return CtxLengths(scaled_values, values.unit, values.typ)


class ScaleContinuousSize(ScaleContinuous):
    range_min: Lengths
    range_max: Lengths

    def __init__(
        self,
        range_min: Lengths = mm(0.2),
        range_max: Lengths = mm(1.0),
        unit: str = "size",
        tick_coverage: TickCoverage | str | ConfigKey = ConfigKey("tick_coverage"),  # type: ignore[assignment]
        choose_tick_params: ChooseTicksParams | ConfigKey = ConfigKey(  # type: ignore[assignment]
            "tick_params"
        ),
        min: float | None = None,
        max: float | None = None,
        *,
        transform: Transform | str = "identity",
        ticks: TickSpec | None = None,
        labeler: TickLabeler | None = None,
    ):
        self.range_min = range_min
        self.range_max = range_max
        super().__init__(
            unit,
            tick_coverage,
            choose_tick_params,
            min=min,
            max=max,
            transform=transform,
            ticks=ticks,
            labeler=labeler,
        )

    @override
    def scale_values(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.unit == self.unit
        if self.min is None or self.max is None:
            # Fallback for no values
            return self.range_min

        vals_arr = self._transform_values(self._as_array(values))

        span = self.max - self.min
        if span != 0:
            norm_vals = (vals_arr - self.min) / span
        else:
            norm_vals = np.zeros_like(vals_arr)

        # Linear interpolation: range_min + norm_vals * (range_max - range_min)
        # We need to handle the broadcasting if norm_vals is an array.
        # LengthsMulOp(norm_vals, (self.range_max - self.range_min)) doesn't work directly if norm_vals is an array.
        # However, we can use the resolve logic or implement it here.

        results: list[Lengths] = []
        range_diff = self.range_max - self.range_min
        for v in norm_vals:
            results.append(self.range_min + float(v) * range_diff)

        if not results:
            return self.range_min

        return self._combine_lengths(results)

    def _combine_lengths(self, lengths: list[Lengths]) -> Lengths:
        if not lengths:
            return mm(0)
        return LengthsSequence(lengths)

    @override
    def ticks(self) -> tuple[NDArray[np.str_], Lengths | Colors]:
        labels, ticks = super().ticks()
        assert isinstance(ticks, CtxLengths)

        # Scale the ticks to the range
        span = self.max - self.min
        if span != 0:
            norm_ticks = (ticks.values - self.min) / span
        else:
            norm_ticks = np.zeros_like(ticks.values)

        range_diff = self.range_max - self.range_min
        scaled_ticks = [self.range_min + float(v) * range_diff for v in norm_ticks]

        return labels, self._combine_lengths(scaled_ticks)


def xcontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousLength:
    """
    Construct a continuous x scale, mapping numbers onto x positions.

    A ``transform`` applies to each coordinate individually after geometries
    are constructed, so statistics computed by a geometry (e.g. hexbin or
    histogram bins) are computed on untransformed data.

    Args:
        tick_coverage: A `TickCoverage`, or one of the strings "flexible",
            "sub", or "super", controlling whether ticks may extend beyond
            the data range (default: taken from the config).
        choose_tick_params: `ChooseTicksParams` tuning tick selection
            (default: taken from the config).
        min: Lower end of the scale. Defaults to the minimum of the data.
        max: Upper end of the scale. Defaults to the maximum of the data.
        transform: A `Transform`, or the name of one ("identity", "log10",
            "log2", "log", "ln", "log1p", "sqrt"), applied to values before
            they are mapped. Ticks are placed and labeled to suit it.
        ticks: Explicit tick values in data space, or a function taking the
            data-space ``(min, max)`` and returning them. Defaults to ticks
            chosen by the transform.
        labeler: Function mapping an array of data-space tick values to
            label strings. Defaults to the transform's labeler.

    Returns:
        A `ScaleContinuousLength` to be passed to `plot`.
    """
    return ScaleContinuousLength("x", *args, **kwargs)


def ycontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousLength:
    """
    Construct a continuous y scale, mapping numbers onto y positions.

    A ``transform`` applies to each coordinate individually after geometries
    are constructed, so statistics computed by a geometry (e.g. hexbin or
    histogram bins) are computed on untransformed data.

    Args:
        tick_coverage: A `TickCoverage`, or one of the strings "flexible",
            "sub", or "super", controlling whether ticks may extend beyond
            the data range (default: taken from the config).
        choose_tick_params: `ChooseTicksParams` tuning tick selection
            (default: taken from the config).
        min: Lower end of the scale. Defaults to the minimum of the data.
        max: Upper end of the scale. Defaults to the maximum of the data.
        transform: A `Transform`, or the name of one ("identity", "log10",
            "log2", "log", "ln", "log1p", "sqrt"), applied to values before
            they are mapped. Ticks are placed and labeled to suit it.
        ticks: Explicit tick values in data space, or a function taking the
            data-space ``(min, max)`` and returning them. Defaults to ticks
            chosen by the transform.
        labeler: Function mapping an array of data-space tick values to
            label strings. Defaults to the transform's labeler.

    Returns:
        A `ScaleContinuousLength` to be passed to `plot`.
    """
    return ScaleContinuousLength("y", *args, **kwargs)


def sizecontinuous(*args: Any, **kwargs: Any) -> ScaleContinuousSize:
    """
    Construct a continuous size scale, mapping numbers onto lengths, used for
    example to vary point radius.

    Args:
        range_min: Length the smallest value maps to (default: `mm(0.2)`).
        range_max: Length the largest value maps to (default: `mm(1.0)`).
        tick_coverage: A `TickCoverage`, or one of the strings "flexible",
            "sub", or "super", controlling whether ticks may extend beyond
            the data range (default: taken from the config).
        choose_tick_params: `ChooseTicksParams` tuning tick selection
            (default: taken from the config).
        min: Lower end of the scale. Defaults to the minimum of the data.
        max: Upper end of the scale. Defaults to the maximum of the data.
        transform: A `Transform`, or the name of one ("identity", "log10",
            "log2", "log", "ln", "log1p", "sqrt"), applied to values before
            they are mapped. Ticks are placed and labeled to suit it.
        ticks: Explicit tick values in data space, or a function taking the
            data-space ``(min, max)`` and returning them. Defaults to ticks
            chosen by the transform.
        labeler: Function mapping an array of data-space tick values to
            label strings. Defaults to the transform's labeler.

    Returns:
        A `ScaleContinuousSize` to be passed to `plot`.
    """
    return ScaleContinuousSize(*args, **kwargs)


_default_scales: dict[str, tuple[Callable[..., Scale] | None, Callable[..., Scale]]] = {
    "x": (xdiscrete, xcontinuous),
    "y": (ydiscrete, ycontinuous),
    "color": (colordiscrete, colorcontinuous),
    "shape": (shapediscrete, shapediscrete),
    "size": (None, sizecontinuous),
}

ScaleSet: TypeAlias = dict[str, Scale]
