
import dapple as dpl
import io

def test_basic_plots():
    dpl.plot().svg(dpl.inch(3), dpl.inch(3))

    out = io.StringIO()
    dpl.plot().svg(dpl.inch(3), dpl.inch(3), output=out)
