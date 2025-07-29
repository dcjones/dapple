
from xml.etree.ElementTree import Element
from .coordinates import Resolvable, AbsCoordSet, Lengths, Occupancy, resolve
from .colors import Colors
from . import svg
from typing import Collection
from itertools import cycle


class ResolvableElement(Element, Resolvable):
    """
    An XML element that can be resolved (unually into another element).
    """

    def __init__(self, tag: str, attrib={}, **extra):
        super().__init__(tag, attrib, **extra)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Union[Lengths, Element]:
        attrib = {k: resolve(v, coords, occupancy) for (k, v) in self.attrib.items()}
        el = Element(self.tag, attrib)

        for child in self:
            el.append(resolve(child, coords, occupancy))

        return el

class VectorizedElement(ResolvableElement):
    """
    An element that is is given vector attribute values and is resolved into multiple single lengths.
    """

    def __init__(self, tag: str, attrib={}, **extra):
        super().__init__(tag, attrib, **extra)

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Element:
        g = svg.g()

        keys = []
        value_iters = []
        veclen = None
        for (key, value) in self.attrib.items():
            keys.append(key)
            if isinstance(value, (Lengths, Colors, Collection)):
                if veclen is None:
                    veclen = len(value)
                else:
                    assert veclen == len(value), f"VectorizedElement attribute {key} has length {len(value)}, but previous attribute has length {veclen}"
            else:
                value_iters.append(cycle(value))

        if veclen is None:
            raise ValueError("VectorizedElement must have vector attributes")

        resolved_children = [
            resolve(child, coords, occupancy)
            for child in self
        ]

        for values in zip(*value_iters):
            attrib = {k: v for (k, v) in zip(keys, values)}
            el = Element(self.tag, attrib)
            el.extend(resolved_children)
            g.append(el)

        return g


class ContextElement(ResolvableElement):
    """
    A special container element that sets up a coordinate transform.
    """

    def __init__():
        pass





# TODO: ctx tags
