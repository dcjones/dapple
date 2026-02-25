
import numpy as np
import pytest
from dapple.scales import ScaleContinuousSize, UnscaledValues
from dapple.coordinates import mm, AbsLengths, ResolveContext, AbsCoordTransform, LengthsSequence
from dapple.occupancy import Occupancy

def test_scale_continuous_size_basic():
    scale = ScaleContinuousSize(range_min=mm(2), range_max=mm(10))
    uv = UnscaledValues("size", [0, 5, 10])
    
    scale.fit_values(uv)
    scale.finalize()
    
    scaled = scale.scale_values(uv)
    assert isinstance(scaled, LengthsSequence)
    assert len(scaled) == 3
    
    ctx = ResolveContext(
        coords={
            "vw": AbsCoordTransform(100.0, 0.0),
            "vh": AbsCoordTransform(100.0, 0.0),
        },
        scales={},
        occupancy=Occupancy(mm(100), mm(100))
    )
    
    resolved = scaled.resolve(ctx)
    assert isinstance(resolved, AbsLengths)
    np.testing.assert_allclose(resolved.values, [2.0, 6.0, 10.0])

def test_scale_continuous_size_zero_span():
    scale = ScaleContinuousSize(range_min=mm(2), range_max=mm(10))
    uv = UnscaledValues("size", [5, 5, 5])
    
    scale.fit_values(uv)
    scale.finalize()
    
    scaled = scale.scale_values(uv)
    
    ctx = ResolveContext(
        coords={
            "vw": AbsCoordTransform(100.0, 0.0),
            "vh": AbsCoordTransform(100.0, 0.0),
        },
        scales={},
        occupancy=Occupancy(mm(100), mm(100))
    )
    
    resolved = scaled.resolve(ctx)
    # If span is zero, it defaults to range_min (which is 2.0)
    np.testing.assert_allclose(resolved.values, [2.0, 2.0, 2.0])

def test_scale_continuous_size_mixed_units():
    # Test with mixed units (not fully supported by everything but should resolve)
    from dapple.coordinates import cxv
    scale = ScaleContinuousSize(range_min=mm(1), range_max=cxv(10))
    uv = UnscaledValues("size", [0, 1])
    
    scale.fit_values(uv)
    scale.finalize()
    
    scaled = scale.scale_values(uv)
    
    ctx = ResolveContext(
        coords={
            "vw": AbsCoordTransform(100.0, 0.0),
            "vh": AbsCoordTransform(100.0, 0.0),
            "x": AbsCoordTransform(10.0, 0.0), # cxv(1) = 10mm
        },
        scales={},
        occupancy=Occupancy(mm(100), mm(100))
    )
    
    resolved = scaled.resolve(ctx)
    # 0 -> 1mm
    # 1 -> 10 * 10mm = 100mm
    np.testing.assert_allclose(resolved.values, [1.0, 100.0])
