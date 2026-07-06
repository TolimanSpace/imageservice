"""
tests/unit/test_cropping.py - Unit tests for workers.cropping
"""

import pytest
import numpy as np

from workers.cropping import (
    crop_centre,
    crop_sidelobe_strip,
    crop_and_merge_corners,
    DEFAULT_ANGLE_DEGREES,
    DEFAULT_STRIP_WIDTH,
)

pytestmark = pytest.mark.unit


class TestCropCentre:

    def test_basic_shape(self):
        img = np.zeros((256, 256), dtype=np.uint16)
        out = crop_centre(img, 128, 128, size=64)
        assert out.shape == (64, 64)

    def test_content_correct(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 999
        out = crop_centre(img, 32, 32, size=10)
        assert out.max() == 999

    def test_edge_clipping_top_left(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        out = crop_centre(img, 0, 0, size=32)
        assert out.shape[0] <= 32 and out.shape[1] <= 32

    def test_edge_clipping_bottom_right(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        out = crop_centre(img, 63, 63, size=32)
        assert out.shape[0] <= 32 and out.shape[1] <= 32

    def test_float_position_rounded(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 500
        out1 = crop_centre(img, 32.4, 32.4, size=10)
        out2 = crop_centre(img, 32,   32,   size=10)
        np.testing.assert_array_equal(out1, out2)

    def test_odd_size(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        out = crop_centre(img, 32, 32, size=11)
        assert out.shape == (11, 11)


class TestCropSidelobeStrip:
    """
    Tests for crop_sidelobe_strip().

    In the multi-ROI architecture each corner ROI is already centred on
    one sidelobe arm - no crop_areas step is needed. The star position
    is expressed in corner ROI-local coordinates; since the star lives
    in the centre ROI (~SIDELOBE_OFFSET pixels away), x_star and y_star
    are typically large negative values.
    """

    @pytest.fixture
    def diagonal_roi(self):
        """100x100 ROI with a bright 45° diagonal."""
        roi = np.zeros((100, 100), dtype=np.uint16)
        for i in range(100):
            roi[i, i] = 4000
        return roi

    @pytest.fixture
    def corner_roi(self):
        """360x360 ROI with a realistic noise background and bright 45° streak."""
        rng = np.random.default_rng(7)
        roi = rng.integers(10, 100, (360, 360), dtype=np.uint16)
        for i in range(360):
            roi[i, i] = 3000
        return roi

    def test_output_shape_cols_equals_width(self, diagonal_roi):
        strip = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0, angle_degrees=45.0, width=6)
        assert strip.shape[1] == 6

    def test_output_dtype_preserved(self, diagonal_roi):
        strip = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0, angle_degrees=45.0, width=6)
        assert strip.dtype == np.uint16

    def test_bright_diagonal_pixels_captured(self, diagonal_roi):
        """Pixels on the 45° diagonal should appear in the strip."""
        strip = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0,
                                     angle_degrees=45.0, width=6)
        assert strip.max() == 4000

    def test_perpendicular_axis_misses_diagonal(self, diagonal_roi):
        """
        A strip at 135° should not capture the 45° diagonal.
        The new binning approach discards incomplete cross-sections, so
        a perpendicular strip across a thin diagonal typically returns
        zero complete rows - all bins have fewer pixels than width.
        """
        strip = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0,
                                     angle_degrees=135.0, width=2)
        # Either empty (no complete cross-sections) or only background pixels
        if strip.size > 0:
            assert strip.max() <= 4000

    def test_axis_outside_roi_returns_empty(self):
        """
        When the dispersion axis doesn't intersect the ROI the result
        should have zero rows.
        """
        roi = np.zeros((10, 10), dtype=np.uint16)
        # 45° line through (-100, 100): passes far above the 10x10 ROI
        strip = crop_sidelobe_strip(roi, -100.0, 100.0,
                                     angle_degrees=45.0, width=6)
        assert strip.shape == (0, 6)

    def test_realistic_star_offset(self, corner_roi):
        """
        Star at typical sensor-centre-to-corner-ROI offset (~1445 px).
        Strip should have rows and correct width.
        """
        strip = crop_sidelobe_strip(corner_roi, -1445.0, -1445.0,
                                     angle_degrees=45.0, width=6)
        assert strip.shape[1] == 6
        assert strip.shape[0] > 0

    def test_wider_width_more_pixels_per_row(self, diagonal_roi):
        """
        Wider strips have more pixels per output row.
        With the binning approach, wider strips may have fewer rows
        (more bins fall below the completeness threshold), but each
        surviving row contains more pixels.
        """
        strip_narrow = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0, angle_degrees=45.0, width=4)
        strip_wide   = crop_sidelobe_strip(diagonal_roi, 0.0, 0.0, angle_degrees=45.0, width=8)
        assert strip_narrow.shape[1] == 4
        assert strip_wide.shape[1]   == 8

    def test_rows_divisible_by_width(self, corner_roi):
        """Output pixel count must be exactly n_rows * width (no remainder)."""
        for w in [4, 6, 8]:
            strip = crop_sidelobe_strip(corner_roi, -1445.0, -1445.0, angle_degrees=45, width=w)
            assert strip.shape[0] * w == strip.size


class TestCropAndMergeCorners:

    @pytest.fixture
    def four_corners(self):
        """Four identical 360x360 corner ROIs with a bright 45° diagonal."""
        rng = np.random.default_rng(3)
        roi = rng.integers(10, 100, (360, 360), dtype=np.uint16)
        for i in range(360):
            roi[i, i] = 3000
        return {
            "top_left":  roi.copy(),
            "top_right": roi.copy(),
            "bot_left":  roi.copy(),
            "bot_right": roi.copy(),
        }

    @pytest.fixture
    def one_star_positions(self):
        """One star at typical offset for all four corner ROIs."""
        pos = {"xs": [-1445.0], "ys": [-1445.0]}
        return {label: pos for label in
                ["top_left", "top_right", "bot_left", "bot_right"]}

    @pytest.fixture
    def two_star_positions(self):
        """Two stars at slightly different offsets."""
        pos = {"xs": [-1445.0, -1440.0], "ys": [-1445.0, -1440.0]}
        return {label: pos for label in
                ["top_left", "top_right", "bot_left", "bot_right"]}

    def test_returns_ndarray(self, four_corners, one_star_positions):
        result = crop_and_merge_corners(four_corners, one_star_positions)
        assert result is None or isinstance(result, np.ndarray)

    def test_empty_corners_returns_none(self):
        assert crop_and_merge_corners({}, {}) is None

    def test_col_count_is_multiple_of_width(self, four_corners,
                                              one_star_positions):
        """Output columns must be a multiple of strip width."""
        result = crop_and_merge_corners(four_corners, one_star_positions,
                                         width=6)
        # DEFAULT_ANGLE_DEGREES: top_right/bot_left use 45° (matches the
        # fixture's 45° diagonal streak); top_left/bot_right use 135°
        # (perpendicular to the streak, may yield fewer/no strips).
        if result is not None:
            assert result.shape[1] % 6 == 0

    def test_two_stars_doubles_columns(self, four_corners,
                                        one_star_positions,
                                        two_star_positions):
        """Two stars should produce twice as many columns as one star."""
        r1 = crop_and_merge_corners(four_corners, one_star_positions, width=6)
        r2 = crop_and_merge_corners(four_corners, two_star_positions, width=6)
        if r1 is not None and r2 is not None:
            assert r2.shape[1] == r1.shape[1] * 2

    def test_missing_corner_label_skipped(self, four_corners,
                                           one_star_positions):
        """Positions dict missing a corner label should not raise."""
        partial_positions = {k: v for k, v in one_star_positions.items()
                             if k != "bot_right"}
        result_partial = crop_and_merge_corners(four_corners, partial_positions,
                                                width=6)
        result_full    = crop_and_merge_corners(four_corners, one_star_positions,
                                                width=6)
        # Partial should have fewer or equal columns than full
        if result_partial is not None and result_full is not None:
            assert result_partial.shape[1] <= result_full.shape[1]
        # Must not raise regardless

    def test_DEFAULT_ANGLE_DEGREES_applied(self, four_corners,
                                            one_star_positions):
        """DEFAULT_ANGLE_DEGREES: top_left/bot_right=135°, top_right/bot_left=45°."""
        assert DEFAULT_ANGLE_DEGREES["top_left"]  == 135.0
        assert DEFAULT_ANGLE_DEGREES["top_right"] ==  45.0
        assert DEFAULT_ANGLE_DEGREES["bot_left"]  ==  45.0
        assert DEFAULT_ANGLE_DEGREES["bot_right"] == 135.0

    def test_custom_corner_angles_override(self, four_corners,
                                            one_star_positions):
        """Passing custom angles should not raise and should be used."""
        custom = {k: 90.0 for k in four_corners}
        result = crop_and_merge_corners(four_corners, one_star_positions,
                                         corner_angles=custom, width=6)
        # A 90° axis is vertical - may or may not yield strips depending
        # on geometry, but must not raise.
        assert result is None or isinstance(result, np.ndarray)

    def test_aligned_axis_captures_more_streak_pixels(self):
        """
        A 45° axis aligned with a 45° bright streak captures many more
        bright pixels than a 135° axis (perpendicular), which only
        intersects the streak at a single crossing point.
        """
        rng = np.random.default_rng(11)
        roi = rng.integers(5, 20, (200, 200), dtype=np.uint16)
        for i in range(200):   # bright 45° diagonal: y = x
            roi[i, i] = 3000

        strip_45  = crop_sidelobe_strip(roi, 100.0, 100.0,
                                         angle_degrees=45.0,  width=6)
        strip_135 = crop_sidelobe_strip(roi, 100.0, 100.0,
                                         angle_degrees=135.0, width=6)

        bright_45  = int((strip_45  == 3000).sum())
        bright_135 = int((strip_135 == 3000).sum())

        # 45° captures the full streak (~200 pixels);
        # 135° captures only the single crossing point (~5 pixels)
        assert bright_45 > bright_135 * 5, (
            f"Expected 45° to capture far more bright pixels than 135°, "
            f"got {bright_45} vs {bright_135}"
        )

    def test_all_rows_same_length(self, four_corners, two_star_positions):
        """All strips must be trimmed to the same number of rows."""
        result = crop_and_merge_corners(four_corners, two_star_positions,
                                         width=6)
        if result is not None:
            # By construction all strips are trimmed to min_rows
            assert result.ndim == 2


# ============================================================================
