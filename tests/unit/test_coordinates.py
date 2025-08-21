import pytest
import dapple.coordinates as dplc

def test_abs_length_init():
    with pytest.raises(Exception):
        dplc.mm("1.0")

    with pytest.raises(Exception):
        dplc.mm(None)

    dplc.mm(1)
    dplc.mm(1.0)
    dplc.mm([1, 2, 3])
    dplc.cm([1, 2, 3])
    dplc.pt([1, 2, 3])
    dplc.inch([1, 2, 3])

def test_ctx_length_init():
    with pytest.raises(Exception):
        dplc.cx("1.0")

    with pytest.raises(Exception):
        dplc.cx(None)

    dplc.cx(1)
    dplc.cx(1.0)
    dplc.cx([1, 2, 3])
    dplc.cy([1, 2, 3])
    dplc.cyv([1, 2, 3])
    dplc.cw([1, 2, 3])
    dplc.cwv([1, 2, 3])
    dplc.ch([1, 2, 3])
    dplc.chv([1, 2, 3])

def test_expression_construction():
    assert dplc.mm(1).scalar_value() == 1
    with pytest.raises(Exception):
        dplc.mm([1, 2]).scalar_value()

    dplc.cx(1) + dplc.mm(1)
    1 * dplc.mm(1)
    -dplc.mm(1)
    abs(dplc.mm(1))
    dplc.mm(1).min(dplc.cx(2))
    dplc.mm(1).max(dplc.cx(2))

def test_min_simplification():
    assert dplc.mm(1).min(dplc.mm(2)) == dplc.mm(1)
    assert dplc.cx(1).min(dplc.cx(2)) == dplc.cx(1)
    assert (dplc.cx(1) + dplc.cy(1)).min(dplc.cx(1) + dplc.mm(1)) == dplc.cx(1) + dplc.cy(1).min(dplc.mm(1))
    assert (dplc.cy(1) + dplc.cx(1)).min(dplc.mm(1) + dplc.cx(1)) == dplc.cy(1).min(dplc.mm(1)) + dplc.cx(1)
    assert (2.0 * dplc.cx(1)).min(2.0 * dplc.cy(1)) == 2.0 * dplc.cx(1).min(dplc.cy(1))
    assert (-2.0 * dplc.cx(1)).min(-2.0 * dplc.cy(1)) == -2.0 * dplc.cx(1).max(dplc.cy(1))
    assert (-dplc.cx(1)).min(-dplc.cy(1)) == -dplc.cx(1).max(dplc.cy(1))

def test_max_simplification():
    assert dplc.mm(1).max(dplc.mm(2)) == dplc.mm(2)
    assert dplc.cx(1).max(dplc.cx(2)) == dplc.cx(2)
    assert (dplc.cx(1) + dplc.cy(1)).max(dplc.cx(1) + dplc.mm(1)) == dplc.cx(1) + dplc.cy(1).max(dplc.mm(1))
    assert (dplc.cy(1) + dplc.cx(1)).max(dplc.mm(1) + dplc.cx(1)) == dplc.cy(1).max(dplc.mm(1)) + dplc.cx(1)
    assert (2.0 * dplc.cx(1)).max(2.0 * dplc.cy(1)) == 2.0 * dplc.cx(1).max(dplc.cy(1))
    assert (-2.0 * dplc.cx(1)).max(-2.0 * dplc.cy(1)) == -2.0 * dplc.cx(1).min(dplc.cy(1))
    assert (-dplc.cx(1)).max(-dplc.cy(1)) == -dplc.cx(1).min(dplc.cy(1))
