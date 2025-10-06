from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Overflow
from enum import Enum
from functools import singledispatch
from numpy.typing import NDArray
from typing import Any, TypeAlias, Tuple, Optional, TYPE_CHECKING, override, cast
import numpy as np
import sympy

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
    def to_sympy(self) -> sympy.Expr:
        """
        Convert the lengths expression to a sympy expression.
        """
        pass

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

    @override
    def to_sympy(self) -> sympy.Expr:
        self.assert_scalar()
        return self.values[0] * sympy.Symbol("mm", positive=True)

    def units(self) -> set[str]:
        return set()


# Constructing absolute lengths


@singledispatch
def abslengths(value, scale=1.0) -> AbsLengths:
    raise NotImplementedError(
        f"Type {type(value)} can't be converted to an absolute length."
    )


@abslengths.register(float)
def _(value, scale=1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float64))


@abslengths.register(int)
def _(value, scale=1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float64))


@abslengths.register(list)
def _(value, scale=1.0) -> AbsLengths:
    return AbsLengths(scale * np.asarray(value, dtype=np.float64))


@abslengths.register(np.ndarray)
def _(value, scale=1.0) -> AbsLengths:
    return AbsLengths((scale * value).astype(np.float64))


def mm(value):
    return abslengths(value)


def cm(value):
    return abslengths(value, scale=10)


def pt(value):
    return abslengths(value, scale=0.352778)


def inch(value):
    return abslengths(value, scale=25.4)


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
    def to_sympy(self) -> sympy.Expr:
        self.assert_scalar()
        if self.typ == CtxLenType.Vec:
            sym = sympy.Function(self.unit + "_v")
        else:
            sym = sympy.Function(self.unit)

        return sym(self.values[0])

    @override
    def units(self) -> set[str]:
        return set([self.unit])


@singledispatch
def ctxlengths(value, unit: str, typ: CtxLenType) -> CtxLengths:
    raise NotImplementedError(
        f"Type {type(value)} can't be converted to a contextual length."
    )


@ctxlengths.register(float)
def _(value, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.array([value], dtype=np.float64), unit, typ)


@ctxlengths.register(int)
def _(value, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.array([value], dtype=np.float64), unit, typ)


@ctxlengths.register(list)
def _(value, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.asarray(value, dtype=np.float64), unit, typ)


@ctxlengths.register(np.ndarray)
def _(value, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(value.astype(np.float64), unit, typ)


def cx(value) -> CtxLengths:
    return ctxlengths(value, "x", CtxLenType.Pos)


def cxv(value) -> CtxLengths:
    return ctxlengths(value, "x", CtxLenType.Vec)


def cy(value) -> CtxLengths:
    return ctxlengths(value, "y", CtxLenType.Pos)


def cyv(value) -> CtxLengths:
    return ctxlengths(value, "y", CtxLenType.Vec)


def vw(value) -> CtxLengths:
    return ctxlengths(value, "vw", CtxLenType.Pos)


def vwv(value) -> CtxLengths:
    return ctxlengths(value, "vw", CtxLenType.Vec)


def vh(value) -> CtxLengths:
    return ctxlengths(value, "vh", CtxLenType.Pos)


def vhv(value) -> CtxLengths:
    return ctxlengths(value, "vh", CtxLenType.Vec)


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

    def to_sympy(self) -> sympy.Expr:
        return self.a.to_sympy() + self.b.to_sympy()

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
    def to_sympy(self) -> sympy.Expr:
        return self.a * self.b.to_sympy()

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
    def to_sympy(self) -> sympy.Expr:
        return -self.a.to_sympy()

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
    def to_sympy(self) -> sympy.Expr:
        return sympy.Min(self.a.to_sympy(), self.b.to_sympy())

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

    @override
    def to_sympy(self) -> sympy.Expr:
        return sympy.Max(self.a.to_sympy(), self.b.to_sympy())

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

    @override
    def to_sympy(self) -> sympy.Expr:
        return sympy.Min(*[self.a[i].to_sympy() for i in range(len(self))])


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

    @override
    def to_sympy(self) -> sympy.Expr:
        return sympy.Max(*[self.a[i].to_sympy() for i in range(len(self))])


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


class CoordBounds:
    """
    Used to keep track of upper and lower bounds corresponding to each contextual unit.
    """

    bounds: dict[str, Tuple[Lengths, Lengths]]

    def __init__(self):
        self.bounds = dict()

    def update(self, l: Lengths):
        for unit in l.units():
            if unit in self.bounds:
                lower, upper = self.bounds[unit]
                self.bounds[unit] = (lower.min(l.min()), upper.max(l.max()))
            else:
                self.bounds[unit] = (l.min(), l.max())

    def update_from_ticks(self, scales: ScaleSet):
        for unit, scale in scales.items():
            _labels, ticks = scale.ticks()

            if not isinstance(ticks, Lengths):
                continue

            ticks_min, ticks_max = ticks[0], ticks[-1]
            if unit in self.bounds:
                lower, upper = self.bounds[unit]
                self.bounds[unit] = (lower.min(ticks_min), upper.max(ticks_max))
            else:
                self.bounds[unit] = (ticks_min, ticks_max)

    def solve(self, flipped: set[str]) -> CoordSet:
        vw_sym, vh_sym = cast(
            tuple[sympy.Symbol, sympy.Symbol], sympy.symbols("vw vh", positive=True)
        )
        translate_sym, scale_sym = cast(
            tuple[sympy.Symbol, sympy.Symbol], sympy.symbols("translate scale")
        )

        coordset: CoordSet = dict()
        for unit, (lower, upper) in self.bounds.items():
            if unit == "x":
                ref_unit = vw_sym
            elif unit == "y":
                ref_unit = vh_sym
            else:
                continue

            # Ideally we'd solve for `translate` and `scale` in the pairs of equations
            #   lower == 0vw
            #   upper = 1vw
            #
            # Frequently though we aren't able to solve these due to min and max in the equations.
            # Here we take a conservative approach and instead solve for every combination of possible lower
            # and upper bounds then take the minimum translsation and scale.
            lower_parts = lower.min_parts()
            upper_parts = upper.max_parts()

            # TODO: We might try converting the whole thing to sympy,
            # simplifying, then splitting into parts.

            lower_part_exprs = [
                self._rewrite_sympy_expression(
                    scale_sym, translate_sym, part.to_sympy(), unit
                )
                for part in lower_parts
            ]

            upper_part_exprs = [
                self._rewrite_sympy_expression(
                    scale_sym, translate_sym, part.to_sympy(), unit
                )
                for part in upper_parts
            ]

            # We try to find the pair of bounds that leads to the smalest
            # scale and choose that as the active bounds. We can't always
            # unambiguously evaluate the minimum, so we assume a specific
            # vw/vh size here to avoid inequalities that can't be evaluated.
            # This will lead to incorrect results in some cases for very small
            # plots.
            ASSUMED_REF_UNIT_SIZE = 75

            scale_expr: None | sympy.Basic = None
            translate_expr: None | sympy.Basic = None
            for lower_part_expr in lower_part_exprs:
                for upper_part_expr in upper_part_exprs:
                    solution = cast(
                        dict[sympy.Symbol, sympy.Basic],
                        sympy.solve(
                            [lower_part_expr - ref_unit, upper_part_expr]
                            if unit in flipped
                            else [lower_part_expr, upper_part_expr - ref_unit],
                            [scale_sym, translate_sym],
                            rational=False,
                        ),
                    )

                    if scale_sym not in solution or translate_sym not in solution:
                        continue

                    if scale_expr is None or sympy.ask(
                        abs(solution[scale_sym]) < abs(scale_expr),
                        sympy.Q.ge(ref_unit, ASSUMED_REF_UNIT_SIZE),
                    ):
                        scale_expr = solution[scale_sym]
                        translate_expr = solution[translate_sym]

            assert isinstance(scale_expr, sympy.Basic)
            assert isinstance(translate_expr, sympy.Basic)

            coordset[unit] = CoordTransform(
                sympy_to_length(scale_expr),
                sympy_to_length(translate_expr),
            )

            # This is the simpler approach that I wished worked more consistently.
            # lower_expr = self._rewrite_sympy_expression(
            #     scale_sym, translate_sym, lower.to_sympy(), unit
            # )
            # upper_expr = self._rewrite_sympy_expression(
            #     scale_sym, translate_sym, upper.to_sympy(), unit
            # )

            # solution = sympy.solve(
            #     [lower_expr - ref_unit, upper_expr]
            #     if unit in flipped
            #     else [lower_expr, upper_expr - ref_unit],
            #     [scale_sym, translate_sym],
            # )

            # coordset[unit] = CoordTransform(
            #     sympy_to_length(solution[scale_sym]),
            #     sympy_to_length(solution[translate_sym]),
            # )

        return coordset

    def _rewrite_sympy_expression(
        self,
        scale_sym: sympy.Symbol,
        translate_sym: sympy.Symbol,
        expr: sympy.Expr,
        unit: str,
    ):
        """
        Substitute `a*unit` with `a*scale + translate` in preparation for
        solving the coordinate transform.
        """

        unit_sym = sympy.Function(unit)
        c = sympy.Wild(
            "c",
            properties=[lambda k: k.is_number],
        )

        expr_rewrite = expr.replace(
            unit_sym(c), lambda c: c * scale_sym + translate_sym
        )

        unit_vec_sym = sympy.Function(unit + "_v")
        expr_rewrite = expr_rewrite.replace(unit_vec_sym(c), lambda c: c * scale_sym)

        return expr_rewrite


def sympy_to_length(expr: sympy.Basic) -> Lengths:
    """
    Convert a small subset of sympy expressions to a Length expression.
    """

    if isinstance(expr, (sympy.Float, sympy.Integer, sympy.Rational)):
        return mm(float(expr))
    elif isinstance(expr, sympy.Symbol):
        if expr.name == "mm":
            return mm(1.0)
        else:
            if expr.name.endswith("_v"):
                return ctxlengths(1.0, expr.name, CtxLenType.Vec)
            else:
                return ctxlengths(1.0, expr.name.rstrip("_v"), CtxLenType.Pos)
    elif issubclass(type(expr), sympy.Function):
        assert len(expr.args) == 1 and isinstance(expr.args[0], sympy.Number)
        arg = float(expr.args[0])
        if expr.name.endswith("_v"):
            return ctxlengths(arg, expr.name, CtxLenType.Vec)
        else:
            return ctxlengths(arg, expr.name.rstrip("_v"), CtxLenType.Pos)
    elif isinstance(expr, sympy.Add):
        return sympy_to_length(expr.args[0]) + sympy_to_length(expr.args[1])
    elif isinstance(expr, sympy.Mul):
        a, b = expr.args
        if a.is_number and not b.is_number:
            assert isinstance(a, sympy.Number)
            return float(a) * sympy_to_length(b)
        elif not a.is_number and b.is_number:
            assert isinstance(b, sympy.Number)
            return float(b) * sympy_to_length(a)
        else:
            raise Exception("Length expression only support scalar multiply")
    elif isinstance(expr, sympy.Min):
        return sympy_to_length(expr.args[0]).min(sympy_to_length(expr.args[1]))
    elif isinstance(expr, sympy.Max):
        return sympy_to_length(expr.args[0]).max(sympy_to_length(expr.args[1]))
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")
