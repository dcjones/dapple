
from xml.etree.ElementTree import Element
from .coordinates import CoordTransform, Resolvable, AbsCoordSet, Lengths, Occupancy, Transform, resolve, mm, translate
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

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Element:
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
        # TODO: Actually, I think we shouldn't expand this when we resolve. Instead
        # we should do this when we write to xml.

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
    A special container element used as a shortcut for defining the special `vw`
    and `vh` units. The primary use case is to define a particular region using
    a bounding box (though bounds are not anywhere enforced), and setting up
    geometry relative to that bounding box using (vw, vh) units.
    """

    def __init__(self, x: Lengths, y: Lengths, width: Lengths, height: Lengths):
        x.assert_scalar()
        y.assert_scalar()
        width.assert_scalar()
        height.assert_scalar()

        # The idea here is to push the translation part to an actual svg
        # transform. That way everything gets translated the same way and we
        # don't have to prefix everything with a 0vw + ..., 0vh + ...
        attribs = {
            "dapple:coords": {
                "vw": CoordTransform(width, mm(0)),
                "vh": CoordTransform(height, mm(0)),
            },
            "transform": translate(x, y)
        }

        super().__init__("g", attribs)
