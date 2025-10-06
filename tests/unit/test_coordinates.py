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
    dplc.vw([1, 2, 3])
    dplc.vwv([1, 2, 3])
    dplc.vh([1, 2, 3])
    dplc.vhv([1, 2, 3])


def test_expression_construction():
    assert dplc.mm(1).scalar_value() == 1
    with pytest.raises(Exception):
        dplc.mm([1, 2]).scalar_value()

    dplc.cx(1) + dplc.mm(1)
    1 * dplc.mm(1)
    -dplc.mm(1)
    dplc.mm(1).min(dplc.cx(2))
    dplc.mm(1).max(dplc.cx(2))


def test_min_simplification():
    one_mm = dplc.mm(1)
    two_mm = dplc.mm(2)
    one_cx = dplc.cx(1)
    two_cx = dplc.cx(2)
    one_cy = dplc.cy(1)
    mul_op = dplc.LengthsMulOp

    assert one_mm.min(two_mm) == one_mm
    assert one_cx.min(two_cx) == one_cx
    assert (one_cx + one_cy).min(one_cx + one_mm) == one_cx + one_cy.min(one_mm)
    assert (one_cy + one_cx).min(one_mm + one_cx) == one_cy.min(one_mm) + one_cx

    assert mul_op(2.0, one_cx).min(mul_op(2.0, one_cy)) == mul_op(
        2.0, one_cx.min(one_cy)
    )
    assert mul_op(-2.0, one_cx).min(mul_op(-2.0, one_cy)) == mul_op(
        -2.0, one_cx.max(one_cy)
    )
    assert (-one_cx).min(-one_cy) == -one_cx.max(one_cy)


def test_max_simplification():
    one_mm = dplc.mm(1)
    two_mm = dplc.mm(2)
    one_cx = dplc.cx(1)
    two_cx = dplc.cx(2)
    one_cy = dplc.cy(1)
    mul_op = dplc.LengthsMulOp

    assert one_mm.max(two_mm) == two_mm
    assert one_cx.max(two_cx) == two_cx
    assert (one_cx + one_cy).max(one_cx + one_mm) == one_cx + one_cy.max(one_mm)
    assert (one_cy + one_cx).max(one_mm + one_cx) == one_cy.max(one_mm) + one_cx

    assert mul_op(2.0, one_cx).max(mul_op(2.0, one_cy)) == mul_op(
        2.0, one_cx.max(one_cy)
    )
    assert mul_op(-2.0, one_cx).max(mul_op(-2.0, one_cy)) == mul_op(
        -2.0, one_cx.min(one_cy)
    )
    assert (-one_cx).max(-one_cy) == -one_cx.min(one_cy)


def test_sympy_conversion():
    def check_round_trip(expr):
        assert dplc.sympy_to_length(expr.to_sympy()) == expr

    check_round_trip(dplc.cx(2))
    check_round_trip(dplc.cx(2) + dplc.cy(2))
    check_round_trip(3.0 * dplc.cy(2))

    # have to handle sympy potentially swapping the argument order
    min_op = dplc.LengthsMinOp
    a = min_op(dplc.cy(2), dplc.cx(1))
    b = min_op(dplc.cx(1), dplc.cy(2))
    a_rt = dplc.sympy_to_length(a.to_sympy())
    assert a_rt == a or a_rt == b

    max_op = dplc.LengthsMinOp
    a = max_op(dplc.cy(2), dplc.cx(1))
    b = max_op(dplc.cx(1), dplc.cy(2))
    a_rt = dplc.sympy_to_length(a.to_sympy())
    assert a_rt == a or a_rt == b
