
from __future__ import annotations

from .occupancy import Occupancy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from numpy.typing import NDArray
from typing import Any, TypeAlias, Tuple
import numpy as np
import sympy

class Resolvable(ABC):
    @abstractmethod
    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Any:
        pass

@singledispatch
def resolve(value, coords: AbsCoordSet, occupancy: Occupancy) -> Any:
    return value

@resolve.register
def _(arg: Resolvable, coords: AbsCoordSet, occupancy: Occupancy) -> Any:
    return arg.resolve(coords, occupancy)


class Lengths(Resolvable):
    @abstractmethod
    def __len__(self) -> int:
        pass

    def assert_scalar(self):
        if not self.isscalar():
            raise ValueError(f"Scalar length expected but found {len(self)} lengths.")

    def isscalar(self) -> bool:
        return len(self) == 1

    def __add__(self, other: Lengths) -> Lengths:
        return LengthsAddOp(self, other)

    def __rmul__(self, other: float) -> Lengths:
        return LengthsMulOp(other, self)

    def __neg__(self) -> Lengths:
        return LengthsNegOp(self)

    def __abs__(self) -> Lengths:
        return LengthsAbsOp(self)

    def min(self, other: Lengths) -> Lengths:
        # simplification rules:
        # min(au, bu) = min(a, b)u (u is absolute)
        if isinstance(self, AbsLengths) and isinstance(other, AbsLengths):
            return abslengths(min(self.values.min(), other.values.min()))
        # min(au, bu) = min(a, b)u (u is contextual)
        if isinstance(self, CtxLengths) and isinstance(other, CtxLengths) and self.unit == other.unit and self.typ == other.typ:
            # TODO: Ok, what do I do with position versus vector?
            return ctxlengths(min(self.values.min(), other.values.min()), self.unit, self.typ)
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

    def max(self, other: Lengths) -> Lengths:
        # simplification rules:
        # max(au, bu) = max(a, b)u (u is absolute)
        if isinstance(self, AbsLengths) and isinstance(other, AbsLengths):
            return abslengths(max(self.values.max(), other.values.max()))
        # max(au, bu) = max(a, b)u (u is contextual)
        if isinstance(self, CtxLengths) and isinstance(other, CtxLengths) and self.unit == other.unit and self.typ == other.typ:
            # TODO: Ok, what do I do with position versus vector?
            return ctxlengths(max(self.values.max(), other.values.max()), self.unit, self.typ)
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
class AbsLengths(Lengths):
    """
    Representation of lengths in millimeters.
    """
    values: NDArray[np.float64]

    def __init__(self, values: np.ndarray):
        self.values = values.astype(np.float64)

    def __len__(self) -> int:
        return len(self.values)

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        return self

    def __repr__(self) -> str:
        return f"AbsLengths({self.values})"

    def __str__(self) -> str:
        if self.isscalar():
            return f"{self.values[0]}mm"
        else:
            return f"{self.values}mm"

    def to_sympy(self) -> sympy.Expr:
        self.assert_scalar()
        return self.values[0] * sympy.Symbol("mm", positive=True)

    def units(self) -> set[str]:
        return set()

# Constructing absolute lengths

@singledispatch
def abslengths(value, scale=1.0) -> AbsLengths:
    raise NotImplementedError(f"Type {type(value)} can't be converted to an absolute length.")

@abslengths.register
def _(value: float, scale=1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float64))

@abslengths.register
def _(value: list, scale=1.0) -> AbsLengths:
    return AbsLengths(scale * np.asarray(value, dtype=np.float64))

@abslengths.register
def _(value: np.ndarray, scale=1.0) -> AbsLengths:
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
    Vec=1
    Pos=2

def _ctx_len_type_str(typ: CtxLenType) -> str:
    match typ:
        case CtxLenType.Vec:
            return "v"
        case CtxLenType.Pos:
            return ""

@dataclass
class CtxLengths(Lengths):
    """
    Representation of lengths in unresolved contrived coordinate system.
    """
    values: NDArray[np.float64]
    unit: str
    typ: CtxLenType

    def __len__(self) -> int:
        return len(self.values)

    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(f"Scalar length expected but found {len(self.values)} lengths.")

    def is_scalar(self):
        return len(self.values) == 1

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        coord = coords.get(self.unit)
        if coord is None:
            raise ValueError(f"No coordinate for {self.unit}.")

        match self.typ:
            case CtxLenType.Vec:
                return AbsLengths(coord.scale * self.values)
            case CtxLenType.Pos:
                return AbsLengths(coord.translate + coord.scale * self.values)

    def __repr__(self) -> str:
        return f"CtxLengths({self.values}, {self.unit}, {self.typ})"

    def __str__(self) -> str:
        unit_str = self.unit + _ctx_len_type_str(self.typ)

        if self.is_scalar():
            return f"{self.values[0]}{unit_str}"
        else:
            return f"{self.values}{unit_str}"

    def to_sympy(self) -> sympy.Expr:
        self.assert_scalar()
        return self.values[0] * sympy.Symbol(self.unit, positive=True)

    def units(self) -> set[str]:
        return set([self.unit])

@singledispatch
def ctxlengths(value, unit: str, typ: CtxLenType) -> CtxLengths:
    raise NotImplementedError(f"Type {type(value)} can't be converted to a contextual length.")

@ctxlengths.register
def _(value: float, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.array([value], dtype=np.float64), unit, typ)

@ctxlengths.register
def _(value: list, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.asarray(value, dtype=np.float64), unit, typ)

@ctxlengths.register
def _(value: np.ndarray, unit: str, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(value.astype(np.float64), unit, typ)

def cx(value) -> CtxLengths:
    return ctxlengths(value, "x", CtxLenType.Pos)

def cxv(value) -> CtxLengths:
    return ctxlengths(value, "x", CtxLenType.Vec)

def cy(value) -> CtxLengths:
    return ctxlengths(value, "y", CtxLenType.Pos)

def cyv(value) -> CtxLengths:
    return ctxlengths(value, "y", CtxLenType.Vec)

def cw(value) -> CtxLengths:
    return ctxlengths(value, "w", CtxLenType.Pos)

def cwv(value) -> CtxLengths:
    return ctxlengths(value, "w", CtxLenType.Vec)

def ch(value) -> CtxLengths:
    return ctxlengths(value, "h", CtxLenType.Pos)

def chv(value) -> CtxLengths:
    return ctxlengths(value, "h", CtxLenType.Vec)

@dataclass
class LengthsAddOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        if len(self.a) != len(self.b):
            raise ValueError(f"Length mismatch: {len(self.a)} != {len(self.b)}")
        self.a = a
        self.b = b

    def __len__(self) -> int:
        return len(self.a)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
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

    def __len__(self) -> int:
        return len(self.b)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        b = self.b.resolve(coords, occupancy)
        assert isinstance(b, AbsLengths)
        return AbsLengths(self.a * b.values)

    def __repr__(self) -> str:
        return f"LengthsMulOp({self.a!r}, {self.b!r})"

    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

    def to_sympy(self) -> sympy.Expr:
        return self.a * self.b.to_sympy()

    def units(self) -> set[str]:
        return self.b.units()

@dataclass
class LengthsNegOp(Lengths):
    a: Lengths

    def __len__(self) -> int:
        return len(self.a)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        a = self.a.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths)
        return AbsLengths(-a.values)

    def __repr__(self) -> str:
        return f"LengthsNegOp({self.a!r})"

    def __str__(self) -> str:
        return f"-{self.a}"

    def to_sympy(self) -> sympy.Expr:
        return -self.a.to_sympy()

    def units(self) -> set[str]:
        return self.a.units()

@dataclass
class LengthsAbsOp(Lengths):
    a: Lengths

    def __len__(self) -> int:
        return len(self.a)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        a = self.a.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths)
        return AbsLengths(np.abs(a.values))

    def __repr__(self) -> str:
        return f"LengthsAbsOp({self.a!r})"

    def __str__(self) -> str:
        return f"abs({self.a})"

    def to_sympy(self) -> sympy.Expr:
        return abs(self.a.to_sympy())

    def units(self) -> set[str]:
        return self.a.units()

@dataclass
class LengthsMinOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        if len(self.a) != len(self.b):
            raise ValueError(f"Length mismatch: {len(self.a)} != {len(self.b)}")
        self.a = a
        self.b = b

    def __len__(self) -> int:
        return len(self.a)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.minimum(a.values, b.values))

    def __repr__(self) -> str:
        return f"LengthsMinOp({self.a!r})"

    def __str__(self) -> str:
        return f"({self.a}).min({self.b})"

    def to_sympy(self) -> sympy.Expr:
        return sympy.Min(self.a.to_sympy(), self.b.to_sympy())

    def units(self) -> set[str]:
        return self.a.units().union(self.b.units())

@dataclass
class LengthsMaxOp(Lengths):
    a: Lengths
    b: Lengths

    def __init__(self, a: Lengths, b: Lengths):
        if len(self.a) != len(self.b):
            raise ValueError(f"Length mismatch: {len(self.a)} != {len(self.b)}")
        self.a = a
        self.b = b

    def __len__(self) -> int:
        return len(self.a)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsLengths:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.maximum(a.values, b.values))

    def __repr__(self) -> str:
        return f"LengthsMaxOp({self.a!r})"

    def __str__(self) -> str:
        return f"({self.a}).max({self.b})"

    def to_sympy(self) -> sympy.Expr:
        return sympy.Max(self.a.to_sympy(), self.b.to_sympy())

    def units(self) -> set[str]:
        return self.a.units().union(self.b.units())

@dataclass
class AbsCoordTransform(Resolvable):
    scale: float
    translate: float

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsCoordTransform:
        return self

AbsCoordSet: TypeAlias = dict[str, AbsCoordTransform]

@dataclass
class CoordTransform(Resolvable):
    scale: Lengths
    translate: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsCoordTransform:
        abs_scale = self.scale.resolve(coords, occupancy)
        abs_translate = self.translate.resolve(coords, occupancy)
        assert isinstance(abs_scale, AbsLengths) and isinstance(abs_translate, AbsLengths)

        return AbsCoordTransform(abs_scale.scalar_value(), abs_translate.scalar_value())

CoordSet: TypeAlias = dict[str, AbsCoordTransform | CoordTransform]


@dataclass
class AbsTransform:
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
        return self.a == 1 and self.b == 0 and self.c == 0 and self.d == 1 and self.tx == 0 and self.ty == 0

    def __str__(self) -> str:
        # only translation
        if self.a == 1.0 and self.b == 0.0 and self.c == 0.0 and self.d == 1.0:
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

    def __init__(self, a: float, b: float, c: float, d: float, tx: Lengths, ty: Lengths):
        tx.assert_scalar()
        ty.assert_scalar()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tx = tx
        self.ty = ty

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> AbsTransform:
        return AbsTransform(
            self.a, self.b, self.c, self.d, self.tx.resolve(coords, occupancy), self.ty.resolve(coords, occupancy)
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
                self.bounds[unit] = (lower.min(l), upper.max(l))
            else:
                self.bounds[unit] = (l, l)
