"""
tests/unit/test_compressor.py — Unit tests for workers.compression
"""

import pytest
import numpy as np

from workers.compression import _make_diffs

pytestmark = pytest.mark.unit


class TestMakeDiffs:

    def test_output_shape(self):
        stack = np.zeros((5, 32, 32), dtype=np.uint16)
        diffs, _, _ = _make_diffs(stack)
        assert diffs.shape == (4, 32, 32)

    def test_small_diffs_int8(self):
        stack = np.zeros((5, 32, 32), dtype=np.uint16)
        for i in range(5):
            stack[i] = i * 10
        diffs, dtype_name, scale = _make_diffs(stack)
        assert dtype_name == "int8"
        assert diffs.dtype == np.int8
        assert scale == 1.0

    def test_large_diffs_int16(self):
        stack = np.zeros((5, 32, 32), dtype=np.uint16)
        stack[0] = 4000
        stack[1] = 0
        diffs, dtype_name, _ = _make_diffs(stack)
        assert dtype_name == "int16"
        assert diffs.dtype == np.int16

    def test_diff_values_correct(self):
        stack = np.array([[[100, 200]], [[150, 180]]], dtype=np.uint16)
        diffs, _, _ = _make_diffs(stack)
        assert diffs[0, 0, 0] == 50
        assert diffs[0, 0, 1] == -20

    def test_no_overflow_across_uint16_boundary(self):
        """frame[1]=0, frame[0]=4095 → diff should be -4095, not wrapping."""
        stack = np.zeros((2, 1, 1), dtype=np.uint16)
        stack[0, 0, 0] = 4095
        stack[1, 0, 0] = 0
        diffs, dtype_name, _ = _make_diffs(stack)
        assert diffs[0, 0, 0] == -4095
        assert dtype_name == "int16"

    def test_boundary_int8_at_127(self):
        stack = np.zeros((2, 1, 1), dtype=np.uint16)
        stack[0, 0, 0] = 0
        stack[1, 0, 0] = 127
        diffs, dtype_name, _ = _make_diffs(stack)
        assert dtype_name == "int8"
        assert diffs[0, 0, 0] == 127

    def test_boundary_int16_at_128(self):
        stack = np.zeros((2, 1, 1), dtype=np.uint16)
        stack[0, 0, 0] = 0
        stack[1, 0, 0] = 128
        diffs, dtype_name, _ = _make_diffs(stack)
        assert dtype_name == "int16"

    def test_constant_stack_zero_diffs(self):
        stack = np.full((5, 32, 32), 1000, dtype=np.uint16)
        diffs, dtype_name, _ = _make_diffs(stack)
        assert np.all(diffs == 0)
        assert dtype_name == "int8"
