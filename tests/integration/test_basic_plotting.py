import io

import dapple as dpl


def test_basic_plots():
    dpl.plot().svg(dpl.inch(3), dpl.inch(3))

    out = io.StringIO()
    dpl.plot().svg(dpl.inch(3), dpl.inch(3), output=out)


def test_basic_points():
    out = io.StringIO()
    dpl.plot(
        dpl.geometry.points(x=[0, 1, 2], y=[0, 1, 2], size=[10.0, 20.0, 30.0])
    ).svg(dpl.inch(3), dpl.inch(3), output=out)


def test_size_scale():
    out = io.StringIO()
    dpl.plot(
        dpl.geometry.points(x=[0, 1, 2], y=[0, 1, 2], size=[10.0, 20.0, 30.0]),
        dpl.sizecontinuous(0.1 * dpl.mm, 0.1 * dpl.cxv),
    ).svg(dpl.inch(3), dpl.inch(3), output=out)
