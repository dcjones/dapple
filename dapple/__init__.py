
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element

from dapple.coordinates import Resolvable, AbsCoordSet
from dapple.occupancy import Occupancy


# Figuring some things out here:
#   - We should be able to build our tree using xml.etree.ElementTree, which
#     is pretty much our XML representation.
#
#

from .treemap import treemap
from .scales import Scale
from .coordinates import Resolvable


class Plot(ResolvableElement):
    def __init__(self):
        super().__init__("dapple:plot")

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Union[Lengths, Element]:
        # TODO: Do a lot of stuff here.
        #
        #   - Figure out scale, coords, layouts.
        #   - Traverse and rewrite the tree.
        #   - Return that rewritten tree.
        pass


def plot(*args, **kwargs) -> Plot:
    """

    """
    pl = Plot()

    for arg in args:
        if isinstance(arg, Element):
            pl.append(arg)
        elif isinstance(arg, Scale):
            # TODO: I guess we keep scales in an attribute?
            pass
        else:
            raise TypeError(f"Unsupported type for plot argument: {type(arg)}")

    for (k, v)in kwargs.items():
        # TODO: not sure what keyword args this actually supports...
        pass

    return pl



def resolve():
    pass
