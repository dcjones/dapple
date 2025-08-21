
from dapple.textextents import Font
from dapple.coordinates import mm

def test_test_extents():
    font = Font("DejaVu Sans", mm(12))  # 12mm font size

    width_a, height_a = font.get_extents("Hello")
    assert width_a.scalar_value() > 0
    assert height_a.scalar_value() > 0

    width_b, height_b = font.get_extents("Hello World")
    assert width_b.scalar_value() > 0
    assert height_b.scalar_value() > 0

    assert width_a.scalar_value() < width_b.scalar_value()
    assert height_a.scalar_value() <= height_b.scalar_value()
