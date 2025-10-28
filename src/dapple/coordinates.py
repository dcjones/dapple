from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Overflow
from enum import Enum
from functools import singledispatch
from numpy.typing import NDArray
from numbers import Real
from typing import (
    Any,
    TypeAlias,
    Tuple,
    Optional,
    NamedTuple,
    TYPE_CHECKING,
    override,
    cast,
)
from scipy.optimize import linprog
import numpy as np
import sys

if TYPE_CHECKING:
    from .occupancy import Occupancy
    from .scales import ScaleSet


class Serializable(ABC):
    """
    Simple serialization interface for types that need to be converted to
    strings during SVG serialization.
    """

    @abstractmethod
    def serialize(self) -> None | str | list[str]:
        pass

    # Ideally we should be able to delete a attriute by returning None.
    # That's kind of annoying to support.


@dataclass
class ResolveContext:
    coords: "AbsCoordSet"
    scales: "ScaleSet"
    occupancy: "Occupancy"


class Resolvable(ABC):
    @abstractmethod
    def resolve(self, ctx: ResolveContext) -> object:
        pass


@singledispatch
def resolve(value, ctx):
    return value


@resolve.register(dict)
def _(arg, ctx) -> dict[str, object]:
    return {k: resolve(v, ctx) for k, v in arg.items()}


@resolve.register(Resolvable)
def _(arg, ctx) -> object:
    return arg.resolve(ctx)


class Lengths(Resolvable, ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Lengths:
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def assert_scalar(self):
        if not self.isscalar():
            raise ValueError(f"Scalar length expected but found {len(self)} lengths.")

    def isscalar(self) -> bool:
        return len(self) == 1

    def __add__(self, other: Lengths) -> Lengths:
        return LengthsAddOp(self, other)

    def __sub__(self, other: Lengths) -> Lengths:
        return self + -other

    def __rmul__(self, other: float) -> Lengths:
        return LengthsMulOp(other, self)

    def __neg__(self) -> Lengths:
        return LengthsNegOp(self)

    def min_parts(self) -> list[Lengths]:
        """
        Generate a list of length expressions `exprs` where self == min(exprs[0], exprs[1], ...).
        """
        return [self]

    def max_parts(self) -> list[Lengths]:
        """
        Generate a list of length expressions `exprs` where self == max(exprs[0], exprs[1], ...).
        """
        return [self]

    @abstractmethod
    def unmin(self) -> Lengths:
        pass

    @abstractmethod
    def unmax(self) -> Lengths:
        pass

    def min(self, other: Optional[Lengths] = None) -> Lengths:
        if other is None:
            return self.unmin()

        # simplification rules:
        # min(au, bu) = min(a, b)u (u is absolute)
        if isinstance(self, AbsLengths) and isinstance(other, AbsLengths):
            return abslengths(min(self.values.min(), other.values.min()))
        # min(au, bu) = min(a, b)u (u is contextual)
        if (
            isinstance(self, CtxLengths)
            and isinstance(other, CtxLengths)
            and self.unit == other.unit
            and self.typ == other.typ
        ):
            return ctxlengths(
                min(self.values.min(), other.values.min()), self.unit, self.typ
            )
        # min(α + γ, β + γ) = min(α, β) + γ
        if isinstance(self, LengthsAddOp) and isinstance(other, LengthsAddOp):
            if self.a == other.a:
                return self.a + self.b.min(other.b)
            elif self.b == other.b:
                return self.a.min(other.a) + other.b
        # min(cα, cβ) = c min(α, β)
        if isinstance(self, LengthsMulOp) and isinstance(other, LengthsMulOp):
            if self.a == other.a:
                if self.a >= 0.0:
                    return self.a * self.b.min(other.b)
                else:
                    return self.a * self.b.max(other.b)
        # min(-α, -β) = -max(α, β)
        if isinstance(self, LengthsNegOp) and isinstance(other, LengthsNegOp):
            return -self.a.max(other.a)

        return LengthsMinOp(self, other)

    def max(self, other: Optional[Lengths] = None) -> Lengths:
        if other is None:
            return self.unmax()

        # simplification rules:
        # max(au, bu) = max(a, b)u (u is absolute)
        if isinstance(self, AbsLengths) and isinstance(other, AbsLengths):
            return abslengths(max(self.values.max(), other.values.max()))
        # max(au, bu) = max(a, b)u (u is contextual)
        if (
            isinstance(self, CtxLengths)
            and isinstance(other, CtxLengths)
            and self.unit == other.unit
            and self.typ == other.typ
        ):
            return ctxlengths(
                max(self.values.max(), other.values.max()), self.unit, self.typ
            )
        # max(α + γ, β + γ) = max(α, β) + γ
        if isinstance(self, LengthsAddOp) and isinstance(other, LengthsAddOp):
            if self.a == other.a:
                return self.a + self.b.max(other.b)
            elif self.b == other.b:
                return self.a.max(other.a) + other.b
        # max(cα, cβ) = c max(α, β)
        if isinstance(self, LengthsMulOp) and isinstance(other, LengthsMulOp):
            if self.a == other.a:
                if self.a >= 0.0:
                    return self.a * self.b.max(other.b)
                else:
                    return self.a * self.b.min(other.b)
        # max(-α, -β) = -min(α, β)
        if isinstance(self, LengthsNegOp) and isinstance(other, LengthsNegOp):
            return -self.a.min(other.a)

        return LengthsMaxOp(self, other)

    @abstractmethod
    def units(self) -> set[str]:
        pass


@dataclass
class AbsLengths(Lengths, Serializable):
    """
    Representation of lengths in millimeters.
    """

    values: NDArray[np.float64]

    def __init__(self, values: np.ndarray):
        if len(values) == 0:
            self.values = np.array([], dtype=np.float64)
            return

        alleq = True
        for value in values:
            alleq &= value == values[0]

        if alleq:
            self.values = np.array([values[0]], dtype=np.float64)
        else:
            self.values = values.astype(np.float64)

        # Lengths should be considered immutable.
        self.values.setflags(write=False)

    def __call__(self, values: Real | list[Real] | np.ndarray) -> AbsLengths:
        if not self.isscalar():
            raise Exception("Call-construct syntax only usable with scalar lengths")

        return abslengths(values, scale=self.scalar_value())

    @override
    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        for value in self.values:
            yield AbsLengths(np.array([value]))

    @override
    def __getitem__(self, index) -> "AbsLengths":
        """
        Get items from values and construct a new AbsLengths.
        Maintains array structure even for scalar indices.
        """
        if isinstance(index, int):
            # For scalar indices, keep as 1D array with single element
            return AbsLengths(np.array([self.values[index]]))
        else:
            # For slices or other indices, use normal indexing
            return AbsLengths(self.values[index])

    @override
    def unmin(self) -> AbsLengths:
        """
        Unary minimum.
        """
        return AbsLengths(self.values.min(keepdims=True))

    @override
    def unmax(self) -> AbsLengths:
        """
        Unary maximum.
        """
        return AbsLengths(self.values.max(keepdims=True))

    def repeat_scalar(self, n: int) -> "AbsLengths":
        self.assert_scalar()
        return AbsLengths(np.repeat(self.values, n))

    @override
    def serialize(self) -> None | str | list[str]:
        if self.isscalar():
            v = self.scalar_value()
            return f"{v:.2f}"
        else:
            return [f"{v:.2f}" for v in self.values]

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        return self

    def __repr__(self) -> str:
        return f"AbsLengths({self.values})"

    def __str__(self) -> str:
        if self.isscalar():
            return f"{self.values[0]}mm"
        else:
            return f"{self.values}mm"

    def __rmul__(self, other: float) -> AbsLengths:
        return AbsLengths(other * self.values)

    def units(self) -> set[str]:
        return set()


# Constructing absolute lengths


@singledispatch
def abslengths(value, scale: float = 1.0) -> AbsLengths:
    raise NotImplementedError(
        f"Type {type(value)} can't be converted to an absolute length."
    )


@abslengths.register(float)
def _(value, scale: float = 1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float64))


@abslengths.register(int)
def _(value, scale: float = 1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float64))


@abslengths.register(list)
def _(value, scale: float = 1.0) -> AbsLengths:
    return AbsLengths(scale * np.asarray(value, dtype=np.float64))


@abslengths.register(np.ndarray)
def _(value, scale: float = 1.0) -> AbsLengths:
    return AbsLengths((scale * value).astype(np.float64))


# The problem with doing it this way is that
mm = abslengths(1.0)
cm = abslengths(10.0)
pt = abslengths(0.352778)
inch = abslengths(25.4)


class CtxLenType(Enum):
    Vec = 1
    Pos = 2


def _ctx_len_type_str(typ: CtxLenType) -> str:
    match typ:
        case CtxLenType.Vec:
            return "v"
        case CtxLenType.Pos:
            return ""


@dataclass
class CtxLengths(Lengths):
    """
    Representation of lengths in unresolved contrived eoordinate system.
    """

    values: NDArray[np.float64]
    unit: str
    typ: CtxLenType

    def __init__(self, values: NDArray[np.float64], unit: str, typ: CtxLenType):
        self.unit = unit
        self.typ = typ

        if len(values) == 0:
            self.values = np.array([], dtype=np.float64)
            return

        assert len(values) > 0
        alleq = True
        for value in values:
            alleq &= value == values[0]

        if alleq:
            self.values = np.array([values[0]], dtype=np.float64)
        else:
            self.values = values.astype(np.float64)

        # Lengths should be considered immutable.
        self.values.setflags(write=False)

    def __call__(self, values: Real | list[Real] | np.ndarray) -> CtxLengths:
        if not self.isscalar():
            raise Exception("Call-construct syntax only usable with scalar lengths")

        return ctxlengths(values, self.unit, self.typ, scale=self.scalar_value())

    @override
    def __len__(self) -> int:
        return len(self.values)

    @override
    def __getitem__(self, index) -> "CtxLengths":
        """
        Get items from values and construct a new AbsLengths.
        Maintains array structure even for scalar indices.
        """
        if isinstance(index, int):
            # For scalar indices, keep as 1D array with single element
            return CtxLengths(np.array([self.values[index]]), self.unit, self.typ)
        else:
            # For slices or other indices, use normal indexing
            return CtxLengths(self.values[index], self.unit, self.typ)

    @override
    def unmin(self) -> CtxLengths:
        """
        Unary minimum.
        """
        return CtxLengths(self.values.min(keepdims=True), self.unit, self.typ)

    @override
    def unmax(self) -> CtxLengths:
        """
        Unary maximum.
        """
        return CtxLengths(self.values.max(keepdims=True), self.unit, self.typ)

    @override
    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(
                f"Scalar length expected but found {len(self.values)} lengths."
            )

    def is_scalar(self):
        return len(self.values) == 1

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        coord = ctx.coords.get(self.unit)
        if coord is None:
            raise ValueError(f"No coordinate for {self.unit}.")

        match self.typ:
            case CtxLenType.Vec:
                return AbsLengths(coord.scale * self.values)
            case CtxLenType.Pos:
                return AbsLengths(coord.translate + coord.scale * self.values)

    @override
    def __repr__(self) -> str:
        return f"CtxLengths({self.values}, {self.unit}, {self.typ})"

    @override
    def __str__(self) -> str:
        unit_str = self.unit + _ctx_len_type_str(self.typ)

        if self.is_scalar():
            return f"{self.values[0]}{unit_str}"
        else:
            return f"{self.values}{unit_str}"

    @override
    def __rmul__(self, other: float) -> CtxLengths:
        return CtxLengths(other * self.values, self.unit, self.typ)

    @override
    def units(self) -> set[str]:
        return set([self.unit])

    def __iter__(self):
        for value in self.values:
            yield ctxlengths(value, self.unit, self.typ)


@singledispatch
def ctxlengths(value, unit: str, typ: CtxLenType, scale: float = 1.0) -> CtxLengths:
    raise NotImplementedError(
        f"Type {type(value)} can't be converted to a contextual length."
    )


@ctxlengths.register(float)
def _(value, unit: str, typ: CtxLenType, scale: float = 1.0) -> CtxLengths:
    return CtxLengths(np.array([value * scale], dtype=np.float64), unit, typ)


@ctxlengths.register(int)
def _(value, unit: str, typ: CtxLenType, scale: float = 1.0) -> CtxLengths:
    return CtxLengths(np.array([value * scale], dtype=np.float64), unit, typ)


@ctxlengths.register(list)
def _(value, unit: str, typ: CtxLenType, scale: float = 1.0) -> CtxLengths:
    return CtxLengths(scale * np.asarray(value, dtype=np.float64), unit, typ)


@ctxlengths.register(np.ndarray)
def _(value, unit: str, typ: CtxLenType, scale: float = 1.0) -> CtxLengths:
    return CtxLengths((scale * value).astype(np.float64), unit, typ)


cx = ctxlengths(1.0, "x", CtxLenType.Pos)
cxv = ctxlengths(1.0, "x", CtxLenType.Vec)
cy = ctxlengths(1.0, "y", CtxLenType.Pos)
cyv = ctxlengths(1.0, "y", CtxLenType.Vec)
vw = ctxlengths(1.0, "vw", CtxLenType.Pos)
vwv = ctxlengths(1.0, "vw", CtxLenType.Vec)
vh = ctxlengths(1.0, "vh", CtxLenType.Pos)
vhv = ctxlengths(1.0, "vh", CtxLenType.Vec)
fw = ctxlengths(1.0, "fw", CtxLenType.Pos)
fwv = ctxlengths(1.0, "fwv", CtxLenType.Vec)
fh = ctxlengths(1.0, "fh", CtxLenType.Pos)
fhv = ctxlengths(1.0, "fhv", CtxLenType.Vec)


@dataclass
class LengthsAddOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        if len(a) != len(b) and len(a) != 1 and len(b) != 1:
            raise ValueError(f"Length mismatch: {len(a)} != {len(b)}")
        self.a = a
        self.b = b

    @override
    def __getitem__(self, index) -> Lengths:
        return LengthsAddOp(self.a[index], self.b[index])

    @override
    def __len__(self) -> int:
        return len(self.a)

    @override
    def unmin(self) -> Lengths:
        if self.a.isscalar() or self.b.isscalar():
            return LengthsAddOp(self.a.unmin(), self.b.unmin())
        else:
            return LengthsUnaryMinOp(self)

    @override
    def unmax(self) -> Lengths:
        if self.a.isscalar() or self.b.isscalar():
            return LengthsAddOp(self.a.unmax(), self.b.unmax())
        else:
            return LengthsUnaryMaxOp(self.a)

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a = self.a.resolve(ctx)
        b = self.b.resolve(ctx)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)
        return AbsLengths(a.values + b.values)

    def __repr__(self) -> str:
        return f"LengthsAddOp({self.a!r}, {self.b!r})"

    def __str__(self) -> str:
        return f"({self.a} + {self.b})"

    def units(self) -> set[str]:
        return self.a.units().union(self.b.units())


@dataclass
class LengthsMulOp(Lengths):
    a: float
    b: Lengths

    @override
    def __len__(self) -> int:
        return len(self.b)

    @override
    def __getitem__(self, index) -> Lengths:
        return LengthsMulOp(self.a, self.b[index])

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        b = self.b.resolve(ctx)
        assert isinstance(b, AbsLengths)
        return AbsLengths(self.a * b.values)

    @override
    def unmin(self) -> LengthsMulOp:
        if self.a < 0.0:
            return LengthsMulOp(self.a, self.b.unmax())
        else:
            return LengthsMulOp(self.a, self.b.unmin())

    @override
    def unmax(self) -> LengthsMulOp:
        if self.a < 0.0:
            return LengthsMulOp(self.a, self.b.unmin())
        else:
            return LengthsMulOp(self.a, self.b.unmax())

    @override
    def __repr__(self) -> str:
        return f"LengthsMulOp({self.a!r}, {self.b!r})"

    @override
    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

    @override
    def units(self) -> set[str]:
        return self.b.units()


@dataclass
class LengthsNegOp(Lengths):
    a: Lengths

    @override
    def __len__(self) -> int:
        return len(self.a)

    @override
    def __getitem__(self, index) -> Lengths:
        return LengthsNegOp(self.a[index])

    @override
    def unmin(self) -> LengthsNegOp:
        return LengthsNegOp(self.a.unmax())

    @override
    def unmax(self) -> LengthsNegOp:
        return LengthsNegOp(self.a.unmin())

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a = self.a.resolve(ctx)
        assert isinstance(a, AbsLengths)
        return AbsLengths(-a.values)

    @override
    def __repr__(self) -> str:
        return f"LengthsNegOp({self.a!r})"

    @override
    def __str__(self) -> str:
        return f"-{self.a}"

    @override
    def units(self) -> set[str]:
        return self.a.units()


@dataclass
class LengthsMinOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        a.assert_scalar()
        b.assert_scalar()
        self.a = a
        self.b = b

    def __len__(self) -> int:
        return 1

    @override
    def __getitem__(self, index) -> Lengths:
        return LengthsMinOp(self.a[index], self.b[index])

    @override
    def unmin(self) -> LengthsMinOp:
        return self

    @override
    def unmax(self) -> LengthsMinOp:
        return self

    @override
    def min_parts(self) -> list[Lengths]:
        return self.a.min_parts() + self.b.min_parts()

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a = self.a.resolve(ctx)
        b = self.b.resolve(ctx)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.minimum(a.values, b.values))

    @override
    def __repr__(self) -> str:
        return f"LengthsMinOp({self.a!r})"

    @override
    def __str__(self) -> str:
        return f"({self.a}).min({self.b})"

    @override
    def units(self) -> set[str]:
        return self.a.units().union(self.b.units())


@dataclass
class LengthsMaxOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        a.assert_scalar()
        b.assert_scalar()
        self.a = a
        self.b = b

    @override
    def __len__(self) -> int:
        return 1

    @override
    def __getitem__(self, index) -> Lengths:
        return LengthsMaxOp(self.a[index], self.b[index])

    @override
    def unmin(self) -> LengthsMaxOp:
        return self

    @override
    def unmax(self) -> LengthsMaxOp:
        return self

    @override
    def max_parts(self) -> list[Lengths]:
        return self.a.max_parts() + self.b.max_parts()

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a = self.a.resolve(ctx)
        b = self.b.resolve(ctx)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.maximum(a.values, b.values))

    @override
    def __repr__(self) -> str:
        return f"LengthsMaxOp({self.a!r})"

    @override
    def __str__(self) -> str:
        return f"({self.a}).max({self.b})"

    def units(self) -> set[str]:
        return self.a.units().union(self.b.units())


@dataclass
class LengthsUnaryMinOp(Lengths):
    a: Lengths

    @override
    def __len__(self) -> int:
        return 1

    @override
    def __getitem__(self, idx: int) -> Lengths:
        return self.a[idx]

    @override
    def unmin(self) -> Lengths:
        return self

    @override
    def unmax(self) -> Lengths:
        return self

    @override
    def units(self) -> set[str]:
        return self.a.units()

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a_abs = self.a.resolve(ctx)
        assert isinstance(a_abs, AbsLengths)
        return a_abs.unmin()


@dataclass
class LengthsUnaryMaxOp(Lengths):
    a: Lengths

    @override
    def __len__(self) -> int:
        return 1

    @override
    def __getitem__(self, idx: int) -> Lengths:
        return self.a[idx]

    @override
    def unmin(self) -> Lengths:
        return self

    @override
    def unmax(self) -> Lengths:
        return self

    @override
    def units(self) -> set[str]:
        return self.a.units()

    @override
    def resolve(self, ctx: ResolveContext) -> AbsLengths:
        a_abs = self.a.resolve(ctx)
        assert isinstance(a_abs, AbsLengths)
        return a_abs.unmax()


@dataclass
class AbsCoordTransform(Resolvable):
    scale: float
    translate: float

    def resolve(self, ctx: ResolveContext) -> AbsCoordTransform:
        return self


AbsCoordSet: TypeAlias = dict[str, AbsCoordTransform]


@dataclass
class CoordTransform(Resolvable):
    scale: Lengths
    translate: Lengths

    def resolve(self, ctx: ResolveContext) -> AbsCoordTransform:
        abs_scale = self.scale.resolve(ctx)
        abs_translate = self.translate.resolve(ctx)
        assert isinstance(abs_scale, AbsLengths) and isinstance(
            abs_translate, AbsLengths
        )

        return AbsCoordTransform(abs_scale.scalar_value(), abs_translate.scalar_value())


CoordSet: TypeAlias = dict[str, AbsCoordTransform | CoordTransform]


@dataclass
class AbsTransform(Serializable):
    """
    Transformation matrix applied to absolute lengths.

    Coordinates (x, y) are transformed like

      [ a c xt ]   [ x ]
      [ b d yt ] * [ y ]
      [ 0 0 1  ]   [ 1 ]

    """

    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float

    def isidentity(self) -> bool:
        return (
            self.a == 1
            and self.b == 0
            and self.c == 0
            and self.d == 1
            and self.tx == 0
            and self.ty == 0
        )

    @override
    def serialize(self) -> None | str:
        # only translation
        if self.a == 1.0 and self.b == 0.0 and self.c == 0.0 and self.d == 1.0:
            if self.tx == 0.0 and self.ty == 0.0:
                return None
            else:
                return f"translate({self.tx:.3f}, {self.ty:.3f})"
        # only scaling
        elif self.c == 0.0 and self.d == 1.0:
            return f"scale({self.a:.3f}, {self.b:.3f})"
        # general transformation
        else:
            return f"matrix({self.a:.3f}, {self.b:.3f}, {self.c:.3f}, {self.d:.3f}, {self.tx:.3f}, {self.ty:.3f})"


@dataclass
class Transform(Resolvable):
    a: float
    b: float
    c: float
    d: float
    tx: Lengths
    ty: Lengths

    def __init__(
        self, a: float, b: float, c: float, d: float, tx: Lengths, ty: Lengths
    ):
        tx.assert_scalar()
        ty.assert_scalar()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tx = tx
        self.ty = ty

    @override
    def resolve(self, ctx: ResolveContext) -> AbsTransform:
        return AbsTransform(
            self.a,
            self.b,
            self.c,
            self.d,
            self.tx.resolve(ctx).scalar_value(),
            self.ty.resolve(ctx).scalar_value(),
        )


def translate(x: Lengths, y: Lengths) -> Transform:
    return Transform(1.0, 0.0, 0.0, 1.0, x, y)


class CoordConstraint(NamedTuple):
    """
    Store terms across units for one coordinate constraint
    """

    # Because x and y are not lengths but rather positions, we need
    # be able to represent the absence of these units, hence allowing None.

    x: float | None = None
    xv: float = 0
    y: float | None = None
    yv: float = 0
    mm: float = 0
    # TODO: why may also consider supporting fw and fh terms.


ZERO_COORD_CONSTRAINT = CoordConstraint()


class CoordBounds:
    """
    Keeps track potential coordinate constraints and solve the coordinate transform.
    """

    constraints: set[CoordConstraint]

    def __init__(self):
        self.constraints = set()

    def update(
        self,
        l: Lengths,
        partial_factor: float = 1.0,
        partial_term: CoordConstraint = ZERO_COORD_CONSTRAINT,
    ):
        x = partial_term.x
        xv = partial_term.xv
        y = partial_term.y
        yv = partial_term.yv
        mm = partial_term.mm
        additional_units = False

        stack: list[tuple[float, Lengths]] = [(partial_factor, l)]
        while stack:
            factor, term = stack.pop()

            match term:
                case AbsLengths():
                    if term.isscalar():
                        mm += factor * term.scalar_value()
                    else:
                        partial_subterm = CoordConstraint(x=x, xv=xv, y=y, yv=yv, mm=mm)
                        self.update(term.unmin(), factor, partial_subterm)
                        self.update(term.unmax(), factor, partial_subterm)

                case CtxLengths():
                    if term.isscalar():
                        if term.unit == "x":
                            if term.typ == CtxLenType.Pos:
                                x = (
                                    0.0 if x is None else x
                                ) + factor * term.scalar_value()
                            elif term.typ == CtxLenType.Vec:
                                xv += factor * term.scalar_value()
                        elif term.unit == "y":
                            if term.typ == CtxLenType.Pos:
                                y = (
                                    0.0 if y is None else y
                                ) + factor * term.scalar_value()
                            elif term.typ == CtxLenType.Vec:
                                yv += factor * term.scalar_value()
                        else:
                            additional_units = True
                    else:
                        partial_subterm = CoordConstraint(x=x, xv=xv, y=y, yv=yv, mm=mm)
                        self.update(term.unmin(), factor, partial_subterm)
                        self.update(term.unmax(), factor, partial_subterm)

                case LengthsAddOp():
                    stack.append((factor, term.a))
                    stack.append((factor, term.b))

                case LengthsMulOp():
                    stack.append((factor * term.a, term.b))

                case LengthsNegOp():
                    stack.append((-factor, term.a))

                case LengthsUnaryMinOp():
                    partial_subterm = CoordConstraint(x=x, xv=xv, y=y, yv=yv, mm=mm)
                    for subterm in term.a:
                        self.update(subterm, factor, partial_subterm)

                case LengthsUnaryMaxOp():
                    partial_subterm = CoordConstraint(x=x, xv=xv, y=y, yv=yv, mm=mm)
                    for subterm in term.a:
                        self.update(subterm, factor, partial_subterm)

                case _:
                    raise ValueError(
                        f"Constraint with unsupported length type: {type(term)}"
                    )

        if x is not None or y is not None:
            if additional_units:
                raise ValueError(f"Constraint with unsupported unit mixture.")

            self.constraints.add(CoordConstraint(x=x, xv=xv, y=y, yv=yv, mm=mm))

    def update_from_ticks(self, scales: ScaleSet):
        for _unit, scale in scales.items():
            _labels, ticks = scale.ticks()

            if not isinstance(ticks, Lengths):
                continue

            ticks_min, ticks_max = ticks[0], ticks[-1]
            self.update(ticks_min)
            self.update(ticks_max)

    def solve(
        self,
        flip_x: bool,
        flip_y: bool,
        fw_transform: AbsCoordTransform,
        fh_transform: AbsCoordTransform,
        aspect_ratio: float | None = None,
    ) -> CoordSet:
        """
        Figure out the translation and scale for the x and y coordinates.

        This is just a pretty simple linear programming problem where we try to maximize the coordinate scale
        while keeping every recorded bound within the main plot viewport (the "focus").
        """

        if len(self.constraints) == 0:
            return {
                "x": AbsCoordTransform(1.0, 0.0),
                "y": AbsCoordTransform(1.0, 0.0),
            }

        # variables are: scale_x, scale_y, translate_x, translate_y
        idx_scale_x = 0
        idx_scale_y = 1
        idx_translate_x = 2
        idx_translate_y = 3

        bounds = [
            (None, 0) if flip_x else (0, None),
            (None, 0) if flip_y else (0, None),
            (None, None),
            (None, None),
        ]

        # objective is set to maximize the scale subject to the constraints
        c = np.array(
            [1.0 if flip_x else -1.0, 1.0 if flip_y else -1.0, 0.0, 0.0],
            dtype=np.float32,
        )

        A_ub = np.zeros((2 * len(self.constraints), 4), dtype=np.float32)
        b_ub = np.zeros(2 * len(self.constraints), dtype=np.float32)
        for i, constraint in enumerate(self.constraints):
            j = 2 * i

            # Constraints can either be wrt to width or height, not both.
            if constraint.x is not None and constraint.y is not None:
                raise ValueError("Constraint cannot be wrt to both width and height")

            if constraint.x is not None:
                ub_size = fw_transform.scale
            elif constraint.y is not None:
                ub_size = fh_transform.scale
            else:
                raise ValueError("Constraint must be wrt to either width or height")

            b_ub[j] = 0.0
            A_ub[j, idx_scale_x] = (
                0.0 if constraint.x is None else constraint.x
            ) + constraint.xv
            A_ub[j, idx_scale_y] = (
                0.0 if constraint.y is None else constraint.y
            ) + constraint.yv
            A_ub[j, idx_translate_x] = 0.0 if constraint.x is None else 1.0
            A_ub[j, idx_translate_y] = 0.0 if constraint.y is None else 1.0
            A_ub[j + 1, :] = A_ub[j, :]

            # non-negativity constraint: 0 <= constraint -> -constraint <= 0
            A_ub[j, :] = -A_ub[j, :]
            b_ub[j] = constraint.mm

            # upper bound size constraint
            b_ub[j + 1] = ub_size - constraint.mm

        if aspect_ratio is not None:
            if flip_x != flip_y:
                sign = -1.0
            else:
                sign = 1.0

            A_eq = np.array(
                [
                    [
                        sign * aspect_ratio,
                        -1.0,
                        0.0,
                        0.0,
                    ]
                ],
                dtype=np.float32,
            )
            b_eq = np.array([0], dtype=np.float32)
        else:
            A_eq = None
            b_eq = None

        solution = linprog(
            c=c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub
        )

        if not solution.success:
            raise ValueError("Failed to solve linear programming coordinate problem")

        coordset: CoordSet = {
            "x": AbsCoordTransform(
                solution.x[idx_scale_x],
                solution.x[idx_translate_x],
            ),
            "y": AbsCoordTransform(
                solution.x[idx_scale_y],
                solution.x[idx_translate_y],
            ),
        }

        return coordset
