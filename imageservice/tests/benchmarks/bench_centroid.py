"""
tests/benchmarks/bench_centroid.py
Performance benchmarks for the centroid computation hot path.

Run with:
    pytest tests/benchmarks/bench_centroid.py --benchmark-enable -v

Results are compared across runs to detect regressions.
"""

import numpy as np
import pytest

from workers.centroid import compute_centroid, compute_pointing_error
from tests.conftest import (
    ROI_HEIGHT, ROI_OFFSET_X, ROI_OFFSET_Y, ROI_WIDTH,
    SENSOR_HEIGHT, SENSOR_WIDTH,
)

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def roi_128x128():
    rng = np.random.default_rng(0)
    img = rng.integers(10, 30, (128, 128), dtype=np.uint16)
    cy, cx = 64, 64
    y, x = np.ogrid[:128, :128]
    img += (3000 * np.exp(-((x-cx)**2 + (y-cy)**2) / 18)).astype(np.uint16)
    return np.clip(img, 0, 4095).astype(np.uint16)


@pytest.fixture(scope="module")
def roi_64x4504():
    """Wide-strip ROI matching the multi-ROI full-width centre region."""
    rng = np.random.default_rng(1)
    img = rng.integers(10, 30, (64, 4504), dtype=np.uint16)
    cy, cx = 32, 2252
    y, x = np.ogrid[:64, :4504]
    img += (3000 * np.exp(-((x-cx)**2 + (y-cy)**2) / 18)).astype(np.uint16)
    return np.clip(img, 0, 4095).astype(np.uint16)


class TestCentroidBenchmarks:

    def test_bench_compute_centroid_128x128(self, benchmark, roi_128x128):
        """
        Benchmark: compute_centroid on a 128×128 ROI.
        Target: < 100 µs on edge hardware (well within 10 ms frame budget).
        """
        result = benchmark(compute_centroid, roi_128x128)
        assert result is not None

    def test_bench_compute_centroid_64x4504(self, benchmark, roi_64x4504):
        """
        Benchmark: compute_centroid on a 64×4504 wide-strip ROI.
        Target: < 500 µs on edge hardware.
        """
        result = benchmark(compute_centroid, roi_64x4504)
        assert result is not None

    def test_bench_compute_pointing_error_128x128(self, benchmark,
                                                    roi_128x128):
        """
        Benchmark: full compute_pointing_error call including coordinate
        transforms. This is the complete hot-path centroid step.
        """
        result = benchmark(
            compute_pointing_error,
            roi_128x128,
            ROI_OFFSET_X, ROI_OFFSET_Y,
            SENSOR_WIDTH, SENSOR_HEIGHT,
            timestamp_ns=123_456_789,
            frame_id=1,
        )
        assert result is not None


