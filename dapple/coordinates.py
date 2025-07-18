
from __future__ import annotations

from .occupancy import Occupancy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from numpy.typing import NDArray
from typing import TypeAlias
import numpy as np

class Resolvable(ABC):
    @abstractmethod
    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        pass


class Lengths(Resolvable):
    def __add__(self, other: Lengths) -> Lengths:
        return LengthsAddOp(self, other)

    def __rmul__(self, other: float) -> Lengths:
        return LengthsMulOp(other, self)

    def __abs__(self) -> Lengths:
        return LengthsAbsOp(self)

    def min(self, other: Lengths) -> Lengths:
        return LengthsMinOp(self, other)

    def max(self, other: Lengths) -> Lengths:
        return LengthsMaxOp(self, other)


@dataclass
class AbsLengths(Lengths):
    """
    Representation of lengths in millimeters.
    """
    values: NDArray[np.float64]

    def __init__(self, values: np.ndarray):
        self.values = values.astype(np.float64)

    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(f"Scalar length expected but found {len(self.values)} lengths.")

    def is_scalar(self) -> bool:
        return len(self.values) == 1

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        return self

    def __repr__(self) -> str:
        return f"AbsLengths({self.values})"

    def __str__(self) -> str:
        if self.is_scalar():
            return f"{self.values[0]}mm"
        else:
            return f"{self.values}mm"

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

    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(f"Scalar length expected but found {len(self.values)} lengths.")

    def is_scalar(self):
        return len(self.values) == 1

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
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

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)
        return AbsLengths(a.values + b.values)

    def __repr__(self) -> str:
        return f"LengthsAddOp({self.a!r}, {self.b!r})"

    def __str__(self) -> str:
        return f"({self.a} + {self.b})"

@dataclass
class LengthsMulOp(Lengths):
    a: float
    b: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        b = self.b.resolve(coords, occupancy)
        assert isinstance(b, AbsLengths)
        return AbsLengths(self.a * b.values)

    def __repr__(self) -> str:
        return f"LengthsMulOp({self.a!r}, {self.b!r})"

    def __str__(self) -> str:
        return f"({self.a} * {self.b})"

@dataclass
class LengthsAbsOp(Lengths):
    a: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths)
        return AbsLengths(np.abs(a.values))

    def __repr__(self) -> str:
        return f"LengthsAbsOp({self.a!r})"

    def __str__(self) -> str:
        return f"abs({self.a})"

@dataclass
class LengthsMinOp(Lengths):
    a: Lengths
    b: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.minimum(a.values, b.values))

    def __repr__(self) -> str:
        return f"LengthsMinOp({self.a!r})"

    def __str__(self) -> str:
        return f"({self.a}).min({self.b})"

@dataclass
class LengthsMaxOp(Lengths):
    a: 'Lengths'
    b: 'Lengths'

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.maximum(a.values, b.values))

    def __repr__(self) -> str:
        return f"LengthsMaxOp({self.a!r})"

    def __str__(self) -> str:
        return f"({self.a}).max({self.b})"

@dataclass
class AbsCoordTransform(Lengths):
    scale: float
    translate: float

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        return self

AbsCoordSet: TypeAlias = dict[str, AbsCoordTransform]

@dataclass
class CtxCoordTransform(Lengths):
    scale: Lengths
    translate: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        abs_scale = self.scale.resolve(coords, occupancy)
        abs_translate = self.translate.resolve(coords, occupancy)
        assert isinstance(abs_scale, AbsLengths) and isinstance(abs_translate, AbsLengths)

        return AbsCoordTransform(abs_scale.scalar_value(), abs_translate.scalar_value())

CoordSet: TypeAlias = dict[str, AbsCoordTransform | CtxCoordTransform]
