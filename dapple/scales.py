

from typing import Any, Mapping, Sequence, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .colors import Colors
from .coordinates import Lengths, CtxLengths, CtxUnit, CtxLenType


# Here I'm thinking unit can be anything. We need to handle, e.g.
# point position and size as different aesthetics. Or stroke and fill.
#
# These need different names because we want to apply different scales.
#
# Questions:
#   1. Do we need to keep track of positions vs vectors?
#   2. Can we have a set a predefined aesthetics?
#   3. How do we know what default scale to apply to aesthetics?
#
# We don't need pos vs vec as long as we have a solid system for default scales.
#
# I'd like to avoid using predefined aesthetics. I think it should be up to the geometry
# to translate parameters into coherent aesthetics.
#
# Towards that end, I think we have an aesthetic name and it's associated "type"
#
# Like ("fill", "color"), ("stroke", "color")
#
# This way we let people write their own geometries without having to "register" default
# scales.
#
# Alternatively, we could just have them register default scales somehow, right?


_default_scales = {
    # TOOD: once we've defined scales
    # This is actually slightly tricky, because we need to guess continuous vs discrete
    # based on the types we collect. So we may need a pair of scales for each of these.
    "x": None,
    "y": None,
    "color": None,
    "size": None,
}

@dataclass
class UnscaledValues:
    aesthetic: str
    values: np.ndarray

    def __init__(self, aesthetic: str, values: ArrayLike):
        self.aesthetic = aesthetic
        self.values = np.array(values)

class Scale(ABC):
    @abstractmethod
    def fit(self, values: UnscaledValues):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def scale(self, values: UnscaledValues) -> Lengths | Colors:
        pass

    # TODO: Interface for getting labels?
    # Mostly we won't need labels for every value.

# TODO:
# What is the interface for this??? In julia we can say
#
#  xdiscrete("f" => "foo", "b" => "bar"), which I really liked, but can't quite do here.
#
# I think we have two options:
#   xdiscrete({"f": "foo", "b": "bar"})
# or
#   xdiscrete(["f", "b"])
#
# Dict is not ordered though, is it? Oh it is in modern python!
#
# What about specifying targets instead of just labels? We often want to do
# that with colors, but it necessitates two maps.
#
# We need an interface to map values to labels and/or targets.
#
# colordiscrete({"f": RED, "b": "BLUE"})
# colordiscrete({"f": ("foo", RED), "b": ("bar", BLUE)})
#
# I guess something like this, but it becomes ambiguous when only
# one value is specified. (I guess we insist that labels are strs and colors are something...)
#


class ScaleDiscrete(Scale):
    aesthetic: str
    fixed: bool
    labeler: Callable[[Any], str]
    sort_by: None | Callable[[Any], Any]
    map: dict[Any, int]
    targets: np.ndarray
    labels: NDArray

    # partial target list used while fitting the scale, maps
    # values to label target pairs.
    _targets: dict[Any, Tuple[str, Any]]

    # TODO: Port over some extra features
    #   - sorted ordering (maybe this can be a function parameter, like just pass order=sorted)

    def __init__(
            self, aesthetic: str, values: Mapping | Sequence | None = None,
            fixed: bool=False, labeler: Callable[[Any], str] = str, sort_by: None | Callable[[Any], Any] = lambda x: x):
        self.aesthetic = aesthetic
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

    def fit(self, values: UnscaledValues):
        for value in values.values:
            if value not in self._targets:
                if self.fixed:
                    raise ValueError(f"Fixed scale cannot be updated with new value {value}")

                self._targets[value] = (self.labeler(value), None)


class ScaleDiscreteLength(ScaleDiscrete):
    unit: CtxUnit
    typ: CtxLenType

    def __init__(
            self, aesthetic: str, unit: CtxUnit, typ: CtxLenType, values: Mapping | Sequence | None = None,
            fixed: bool=False, labeler: Callable[[Any], str] = str, sort_by: None | Callable[[Any], Any] = lambda x: x):
        self.unit = unit
        self.typ = typ
        super().__init__(aesthetic, values, fixed, labeler, sort_by)

    def finalize(self):
        if self.sort_by is not None:
            values = sorted(self._targets.keys(), key=self.sort_by)
        else:
            values = self._targets.keys()


        self.targets = np.zeros(len(self._targets), dtype=np.float32)
        self.map = dict()
        labels = []
        next_target = max(self._targets) + 1

        for (i, value) in enumerate(values):
            (label, target) = (self.labeler(value), value)
            self.map[value] = i
            labels.append(label)
            if target is None:
                self.targets[i] = next_target
                next_target += 1
            else:
                self.targets[i] = target

        self.labels = np.array(labels)

    def scale(self, values: UnscaledValues) -> Lengths | Colors:
        assert values.aesthetic == self.aesthetic
        indices = np.array(self.map[value] for value in values.values)
        return CtxLengths(self.targets[indices], self.unit, self.typ)

    # TODO: Maybe we have a unified tick_labels interface across scales
    def tick_labels(self) -> Tuple[NDArray[np.str_], CtxLengths]:
        return self.labels, CtxLengths(self.targets, self.unit, self.typ)

class ScaleDiscreteColor(ScaleDiscrete):
    # TODO: Should look pretty similar to ScaleDiscreteLength but we need some
    # color scheme logic. Figure out if there is a good color scheme package for python first.
    pass

class ScaleContinuous(Scale):
    aesthetic: str
    # TODO: This differs from discrete scale primarily in that it only really
    # keeps track of the minimum and maximum values.
    #
    # I think we also have logic here for defining the ticks. In Dapple we are always
    # saying `tick_coverage=:sub`. We need a way to replace that with something nice.

    pass
