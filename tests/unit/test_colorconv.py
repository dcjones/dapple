import pytest
import numpy as np

from dapple.colors import rgb_to_xyz, xyz_to_rgb, xyz_to_oklab, oklab_to_xyz


def test_xyz_to_oklab():
    # Taken from: https://bottosson.github.io/posts/oklab/#table-of-example-xyz-and-oklab-pairs
    #

    xyz_vals = np.array(
        [
            [0.950, 1.000, 1.089],
            [1.000, 0.000, 0.000],
            [0.000, 1.000, 0.000],
            [0.000, 0.000, 1.000],
        ],
        dtype=np.float32,
    )

    lab_vals = np.array(
        [
            [1.000, 0.000, 0.000],
            [0.450, 1.236, -0.019],
            [0.922, -0.672, 0.263],
            [0.153, -1.415, -0.449],
        ],
        dtype=np.float32,
    )

    assert xyz_to_oklab(xyz_vals) == pytest.approx(lab_vals, abs=0.01)
    assert oklab_to_xyz(lab_vals) == pytest.approx(xyz_vals, abs=0.01)


def test_roundtrip():
    rgb = np.astype(np.random.rand(1000, 3), np.float32)
    xyz = rgb_to_xyz(rgb)
    rgb2 = xyz_to_rgb(xyz)
    assert rgb2 == pytest.approx(rgb, abs=0.01)


def test_roundtrip_oklab():
    rgb = np.random.rand(1000, 3)
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_oklab(xyz)
    xyz2 = oklab_to_xyz(lab)
    rgb2 = xyz_to_rgb(xyz2)
    assert rgb2 == pytest.approx(rgb, abs=0.01)
