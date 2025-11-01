from dapple.geometry.points import points
from dapple.elements import VectorizedElement
from dapple.config import ConfigKey
from dapple.coordinates import mm


def test_points_default_size():
    element = points([0, 1], [0, 1])

    assert isinstance(element, VectorizedElement)
    size_attr = element.attrib["r"]
    assert isinstance(size_attr, ConfigKey)
    assert size_attr.key == "pointsize"


def test_points_custom_size():
    custom_size = mm(1.2)
    element = points([0, 1], [0, 1], size=custom_size)

    assert element.attrib["r"] is custom_size
