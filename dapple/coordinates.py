
# Here we need to define some coordinate and unit types. This was cheap in julia
# but not so much in python. I think we have a few options to consider.
#
#   - No scalar lengths. Always use wrapped ndarrays
#   - Use bare floats as absolute lengths, with the convention
#     that it's always interpreted as as mm.
#   - If we do that, we'll still need a representation for ContextualLength
#     as well as length expressions, which can't be bare. Here we'll used
#     wrapped ndarrays, which makes me think we may as well do that everywhere.

import numpy as np
import numpy.typing as npt


"""
Representation of lengths in millimeters.
"""
class AbsoluteLengths:
    values: npt.NDArray[np.float32]

    def __init__(self, values: npt.ArrayLike | float):
        if isinstance(values, (int, float)):
            self.values = np.array([values], dtype=np.float32)
        else:
            self.values = np.array(values, dtype=np.float32)

    def assert_scalar(self):
        if self.values.shape != (1,):
            raise ValueError("Expected scalar length, got array of shape", self.values.shape)


def mm(value: float):
    return AbsoluteLengths(value)

def cm(value: float):
    return AbsoluteLengths(value * 10)

def pt(value: float):
    return AbsoluteLengths(value * 0.352778)

def inch(value: float):
    return AbsoluteLengths(value * 25.4)


"""
Representation of lengths in unresolved contrived coordinate system.
"""
class ContextualLengths:
    value: npt.NDArray[np.float32]
    symbol: str

    def __init__(self, symbol: str, value: npt.ArrayLike | float):
        self.symbol = symbol
        if isinstance(value, (int, float)):
            self.value = np.array([value], dtype=np.float32)
        else:
            self.value = np.array(value, dtype=np.float32)


"""
An expression over some combination of absolute and contextual units.
"""
class LengthExpressions:
    pass


# TODO: Common operations


# TODO: Representation of unscaled values
# We should include a flag to say whether it is a position or a vector.
#
#
