
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

@dataclass
class AbsLengths(Resolvable):
    """
    Representation of lengths in millimeters.
    """
    values: NDArray[np.float32]

    def __init__(self, values: np.ndarray):
        self.values = values.astype(np.float32)

    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(f"Scalar length expected but found {len(self.values)} lengths.")

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        return self

# TODO: Support operations
#   - AbsLengths + AbsLengths
#   - AbsLengths - AbsLengths
#   - float * AbsLengths

# Constructing absolute lengths

@singledispatch
def abslengths(value, scale=1.0) -> AbsLengths:
    raise NotImplementedError(f"Type {type(value)} can't be converted to an absolute length.")

@abslengths.register
def _(value: float, scale=1.0) -> AbsLengths:
    return AbsLengths(np.array([value * scale], dtype=np.float32))

@abslengths.register
def _(value: list, scale=1.0) -> AbsLengths:
    return AbsLengths(scale * np.asarray(value, dtype=np.float32))

@abslengths.register
def _(value: np.ndarray, scale=1.0) -> AbsLengths:
    return AbsLengths((scale * value).astype(np.float32))

def mm(value):
    return abslengths(value)

def cm(value):
    return abslengths(value, scale=10)

def pt(value):
    return abslengths(value, scale=0.352778)

def inch(value):
    return abslengths(value, scale=25.4)

class CtxUnit(Enum):
    CtxUnitX=1 # contextual x units
    CtxUnitY=2 # contextual y units
    CtxUnitW=3 # contextual viewport width units (in [0, 1])
    CtxUnitH=4 # contextual viewport height units (in [0, 1])

class CtxLenType(Enum):
    CtxVec=1
    CtxPos=2

@dataclass
class CtxLengths(Resolvable):
    """
    Representation of lengths in unresolved contrived coordinate system.
    """
    values: NDArray[np.float32]
    unit: CtxUnit
    typ: CtxLenType

    def assert_scalar(self):
        if len(self.values) != 1:
            raise ValueError(f"Scalar length expected but found {len(self.values)} lengths.")

    def scalar_value(self) -> float:
        self.assert_scalar()
        return float(self.values[0])

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        coord = coords.get(self.unit)
        if coord is None:
            raise ValueError(f"No coordinate for {self.unit}.")

        match self.typ:
            case CtxLenType.CtxVec:
                return AbsLengths(coord.scale * self.values)
            case CtxLenType.CtxPos:
                return AbsLengths(coord.translate + coord.scale * self.values)

@singledispatch
def ctxlengths(value, unit: CtxUnit, typ: CtxLenType) -> CtxLengths:
    raise NotImplementedError(f"Type {type(value)} can't be converted to a contextual length.")

@ctxlengths.register
def _(value: float, unit: CtxUnit, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.array([value], dtype=np.float32), unit, typ)

@ctxlengths.register
def _(value: list, unit: CtxUnit, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(np.asarray(value, dtype=np.float32), unit, typ)

@ctxlengths.register
def _(value: np.ndarray, unit: CtxUnit, typ: CtxLenType) -> CtxLengths:
    return CtxLengths(value.astype(np.float32), unit, typ)

def cx(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitX, CtxLenType.CtxPos)

def cxv(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitX, CtxLenType.CtxVec)

def cy(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitY, CtxLenType.CtxPos)

def cyv(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitY, CtxLenType.CtxVec)

def cw(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitW, CtxLenType.CtxPos)

def cwv(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitW, CtxLenType.CtxVec)

def ch(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitH, CtxLenType.CtxPos)

def chv(value) -> CtxLengths:
    return ctxlengths(value, CtxUnit.CtxUnitH, CtxLenType.CtxVec)

@dataclass
class LengthsAddOp(Resolvable):
    a: Lengths
    b: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)
        return AbsLengths(a.values + b.values)

@dataclass
class LengthsMulOp(Resolvable):
    a: float
    b: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        b = self.b.resolve(coords, occupancy)
        assert isinstance(b, AbsLengths)
        return AbsLengths(self.a * b.values)

@dataclass
class LengthsMinOp(Resolvable):
    a: Lengths
    b: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.minimum(a.values, b.values))

@dataclass
class LengthsMaxOp(Resolvable):
    a: 'Lengths'
    b: 'Lengths'

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        a = self.a.resolve(coords, occupancy)
        b = self.b.resolve(coords, occupancy)
        assert isinstance(a, AbsLengths) and isinstance(b, AbsLengths)

        return AbsLengths(np.maximum(a.values, b.values))

# TODO:
#   - Can we support +, *, min, max?
#   - Pretty printing these expressions

LengthsExpr: TypeAlias = LengthsAddOp | LengthsMulOp | LengthsMinOp | LengthsMaxOp
Lengths: TypeAlias = AbsLengths | CtxLengths | LengthsExpr

@dataclass
class AbsCoordTransform(Resolvable):
    scale: float
    translate: float

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        return self

AbsCoordSet: TypeAlias = dict[CtxUnit, AbsCoordTransform]

@dataclass
class CtxCoordTransform(Resolvable):
    scale: Lengths
    translate: Lengths

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Resolvable:
        abs_scale = self.scale.resolve(coords, occupancy)
        abs_translate = self.translate.resolve(coords, occupancy)
        assert isinstance(abs_scale, AbsLengths) and isinstance(abs_translate, AbsLengths)

        return AbsCoordTransform(abs_scale.scalar_value(), abs_translate.scalar_value())

CoordSet: TypeAlias = dict[CtxUnit, AbsCoordTransform | CtxCoordTransform]

# TODO: Representation of unscaled values
# We should include a flag to say whether it is a position or a vector.
# This should perhaps go in a separate scale.py file.

class UnscaledValues:
    pass
