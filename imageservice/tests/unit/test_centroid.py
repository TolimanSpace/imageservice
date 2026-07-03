"""
tests/unit/test_centroid.py - Unit tests for startracker.centroid
"""

import pytest
import numpy as np

from workers.centroid import (
    CENTRE_ROI_LABEL,
    CentroidResult,
    PointingError,
    compute_centroid,
    compute_pointing_error,
)

pytestmark = pytest.mark.unit

SENSOR_W = 4504
SENSOR_H = 4504
ROI_W    = 128
ROI_H    = 128
OFF_X    = (SENSOR_W - ROI_W) // 2
OFF_Y    = (SENSOR_H - ROI_H) // 2


# ---------------------------------------------------------------------------
# compute_centroid
# ---------------------------------------------------------------------------

class TestComputeCentroid:

    def test_single_pixel_exact_position(self):
        img = np.zeros((64, 128), dtype=np.uint16)
        img[20, 50] = 1000
        r = compute_centroid(img)
        assert r is not None
        assert abs(r.x - 50.0) < 1e-9
        assert abs(r.y - 20.0) < 1e-9

    def test_two_equal_pixels_midpoint(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[10, 20] = 500
        img[10, 30] = 500
        r = compute_centroid(img)
        assert abs(r.x - 25.0) < 1e-9
        assert abs(r.y - 10.0) < 1e-9

    def test_weighted_centroid_pulled_toward_brighter_pixel(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 20] = 100
        img[32, 40] = 300   # 3x brighter - centroid should be closer to 40
        r = compute_centroid(img)
        # Weighted mean: (20*100 + 40*300) / 400 = 35.0
        assert abs(r.x - 35.0) < 1e-9

    def test_zero_image_returns_none(self, blank_roi):
        assert compute_centroid(blank_roi) is None

    def test_min_total_intensity_threshold(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 50
        assert compute_centroid(img, min_total_intensity=100) is None
        assert compute_centroid(img, min_total_intensity=50)  is not None

    def test_non_2d_raises_value_error(self):
        with pytest.raises(ValueError, match="2D"):
            compute_centroid(np.zeros((4, 4, 4), dtype=np.uint16))

    def test_total_intensity_correct(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 2000
        img[31, 31] = 500
        r = compute_centroid(img)
        assert r.total_intensity == 2500.0

    def test_peak_value_correct(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 3000
        img[20, 20] = 1000
        r = compute_centroid(img)
        assert r.peak_value == 3000

    def test_centroid_at_roi_centre(self, star_roi):
        r = compute_centroid(star_roi)
        assert r is not None
        # Gaussian centred at (ROI_H//2, ROI_W//2) - within 1 pixel
        assert abs(r.x - ROI_W // 2) < 1.0
        assert abs(r.y - ROI_H // 2) < 1.0

    def test_uint16_no_overflow(self):
        """Large uint16 array should not overflow during summation."""
        img = np.full((256, 256), 4095, dtype=np.uint16)
        r = compute_centroid(img)
        assert r is not None
        assert r.total_intensity == 256 * 256 * 4095
        # Centroid should be at exact centre
        assert abs(r.x - 127.5) < 1e-6
        assert abs(r.y - 127.5) < 1e-6


# ---------------------------------------------------------------------------
# compute_pointing_error
# ---------------------------------------------------------------------------

class TestComputePointingError:

    def test_star_at_sensor_centre_zero_error(self):
        """Star exactly at sensor centre → (dx, dy) = (0, 0)."""
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[ROI_H // 2, ROI_W // 2] = 3000

        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=1_000_000, frame_id=1,
        )
        assert pe is not None
        assert abs(pe.dx) < 1e-6
        assert abs(pe.dy) < 1e-6

    def test_star_displaced_right(self):
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[ROI_H // 2, ROI_W // 2 + 10] = 3000
        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=1_000_000, frame_id=1,
        )
        assert abs(pe.dx - 10.0) < 1e-6
        assert abs(pe.dy) < 1e-6

    def test_star_displaced_up(self):
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[ROI_H // 2 - 5, ROI_W // 2] = 3000
        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=1_000_000, frame_id=1,
        )
        assert abs(pe.dy - (-5.0)) < 1e-6

    def test_no_signal_returns_none(self, blank_roi):
        pe = compute_pointing_error(
            blank_roi, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=0, frame_id=0,
        )
        assert pe is None

    def test_timestamp_propagated(self):
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[ROI_H // 2, ROI_W // 2] = 3000
        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=987_654_321, frame_id=42,
        )
        assert pe.timestamp_ns == 987_654_321
        assert pe.frame_id == 42

    def test_as_tuple(self):
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[ROI_H // 2, ROI_W // 2] = 3000
        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=0, frame_id=0,
        )
        t = pe.as_tuple()
        assert len(t) == 2
        assert t == (pe.dx, pe.dy)

    def test_sensor_coordinates_correct(self):
        img = np.zeros((ROI_H, ROI_W), dtype=np.uint16)
        img[0, 0] = 3000   # top-left pixel of ROI
        pe = compute_pointing_error(
            img, OFF_X, OFF_Y, SENSOR_W, SENSOR_H,
            timestamp_ns=0, frame_id=0,
        )
        assert abs(pe.x_sensor - OFF_X) < 1e-6
        assert abs(pe.y_sensor - OFF_Y) < 1e-6

    def test_centre_roi_label_constant(self):
        assert CENTRE_ROI_LABEL == "centre"
