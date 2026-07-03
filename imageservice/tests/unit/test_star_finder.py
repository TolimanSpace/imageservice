"""
tests/unit/test_star_finder.py — Unit tests for startracker.star_finder
"""

import pytest
import numpy as np

from startracker.star_finder import (
    deconvolve_image,
    find_stars,
    load_deconvolution_kernel,
    roi_local_to_sensor,
    sensor_to_roi_local,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def identity_kernel():
    """Flat frequency-domain kernel (passes all frequencies equally)."""
    return np.ones((128, 128), dtype=np.complex64)


@pytest.fixture
def delta_kernel():
    """
    Kernel that approximates identity in the spatial domain.
    FFT of a centred delta function → uniform phase.
    """
    delta = np.zeros((128, 128), dtype=np.float32)
    delta[64, 64] = 1.0
    from scipy.fft import fft2
    return fft2(delta)


class TestDeconvolveImage:

    def test_shape_preserved(self, identity_kernel, star_roi):
        out = deconvolve_image(star_roi, identity_kernel)
        assert out.shape == star_roi.shape

    def test_dtype_float32(self, identity_kernel, star_roi):
        out = deconvolve_image(star_roi, identity_kernel)
        assert out.dtype == np.float32

    def test_zero_image_returns_zeros(self, identity_kernel):
        img = np.zeros((128, 128), dtype=np.uint16)
        out = deconvolve_image(img, identity_kernel)
        assert out.max() == 0.0

    def test_shape_mismatch_raises(self, star_roi):
        bad_kernel = np.ones((64, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match="shape"):
            deconvolve_image(star_roi, bad_kernel)


class TestFindStars:

    def test_finds_one_star(self, identity_kernel):
        img = np.zeros((128, 128), dtype=np.uint16)
        img[64, 64] = 4000
        result = find_stars(img, identity_kernel, n_stars=1)
        assert len(result["xs"]) == 1
        assert len(result["ys"]) == 1

    def test_empty_image_returns_empty(self, identity_kernel):
        img = np.zeros((128, 128), dtype=np.uint16)
        result = find_stars(img, identity_kernel)
        assert result["xs"] == []
        assert result["ys"] == []

    def test_finds_at_most_n_stars(self, identity_kernel, two_star_roi):
        result = find_stars(two_star_roi, identity_kernel, n_stars=2)
        assert len(result["xs"]) <= 2

    def test_high_threshold_reduces_detections(self, identity_kernel,
                                                two_star_roi):
        low  = find_stars(two_star_roi, identity_kernel,
                          n_stars=2, threshold=0.01)
        high = find_stars(two_star_roi, identity_kernel,
                          n_stars=2, threshold=0.99)
        assert len(low["xs"]) >= len(high["xs"])

    def test_returns_float_coordinates(self, identity_kernel):
        img = np.zeros((128, 128), dtype=np.uint16)
        img[64, 64] = 4000
        result = find_stars(img, identity_kernel, n_stars=1)
        if result["xs"]:
            assert isinstance(result["xs"][0], float)
            assert isinstance(result["ys"][0], float)


class TestCoordinateTransforms:

    def test_roi_to_sensor_and_back(self):
        xs, ys = [10.5, 20.3], [15.0, 25.7]
        sensor = roi_local_to_sensor(xs, ys, 500, 600)
        local  = sensor_to_roi_local(sensor["xs"], sensor["ys"], 500, 600)
        for a, b in zip(local["xs"], xs):
            assert abs(a - b) < 1e-9
        for a, b in zip(local["ys"], ys):
            assert abs(a - b) < 1e-9

    def test_zero_offset_identity(self):
        xs, ys = [100.0], [200.0]
        sensor = roi_local_to_sensor(xs, ys, 0, 0)
        assert sensor["xs"] == xs
        assert sensor["ys"] == ys

    def test_empty_lists(self):
        result = roi_local_to_sensor([], [], 100, 200)
        assert result["xs"] == []
        assert result["ys"] == []

    def test_load_kernel_wrong_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_deconvolution_kernel(str(tmp_path / "missing.npy"))

    def test_load_kernel_wrong_ndim(self, tmp_path):
        p = tmp_path / "bad_kernel.npy"
        import numpy as np
        np.save(str(p), np.ones((4, 4, 4)))
        with pytest.raises(ValueError, match="2-D"):
            load_deconvolution_kernel(str(p))


# ============================================================================
