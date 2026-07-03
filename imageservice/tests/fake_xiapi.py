"""
fake_xiapi.py - Fake implementation of the Ximea xiapi module.

Provides FakeCamera, FakeImage, and FakeXiapi, which together replace
the real ximea.xiapi C extension in tests. The fake:

  - Records every set_* and param call in self.calls for assertion
  - Returns configurable sensor dimensions from get_*_maximum()
  - Simulates frame delivery via FakeImage.set_data()
  - Advances acq_nframe and timestamps on each get_image() call
  - Can simulate Xi_errors on demand via inject_error()

Usage in tests
--------------
    from tests.fake_xiapi import FakeXiapi

    fake = FakeXiapi()
    config = single_roi_config(...)
    cam = XimeaCamera(config, _xiapi=fake)

    # Pre-load frame data
    cam._img.set_data(np.zeros((64, 128), dtype=np.uint16))

    with cam:
        frame = cam.acquire_frame()
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Simulated Xi_error
# ---------------------------------------------------------------------------

class Xi_error(Exception):
    """Fake Ximea API error, matching the real xiapi.Xi_error interface."""
    def __init__(self, status: int = 10, message: str = "Simulated Xi_error"):
        self.status = status
        super().__init__(f"Xi_error status={status}: {message}")


# ---------------------------------------------------------------------------
# FakeImage
# ---------------------------------------------------------------------------

class FakeImage:
    """
    Mimics xiapi.Image.

    Call set_data() to load synthetic pixel data before each acquire_frame()
    call. The fake auto-increments acq_nframe and tsSec/tsUSec on each
    get_image() call (done by FakeCamera.get_image()).
    """

    def __init__(self) -> None:
        self._data: Optional[np.ndarray] = None
        self.acq_nframe: int = 0
        self.nframe:     int = 0
        self.tsSec:      int = 0
        self.tsUSec:     int = 0

    def set_data(self, array: np.ndarray) -> None:
        """Load synthetic image data for the next acquisition."""
        self._data = array.copy()

    def get_image_data_numpy(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError(
                "FakeImage has no data - call set_data() before acquire_frame()."
            )
        return self._data


# ---------------------------------------------------------------------------
# FakeCamera
# ---------------------------------------------------------------------------

class FakeCamera:
    """
    Mimics xiapi.Camera.

    Records all API calls for assertion. Returns configurable defaults
    for all get_* queries. Supports error injection via inject_error().
    """

    def __init__(
        self,
        sensor_width:  int = 4504,
        sensor_height: int = 4504,
    ) -> None:
        self.calls: List[Tuple] = []
        self._params: dict = {}
        self._sensor_width  = sensor_width
        self._sensor_height = sensor_height
        self._framerate:    float = 0.0
        self._exposure:     int   = 0
        self._error_queue:  List[Xi_error] = []
        self._frame_counter: int = 0
        self._img_ref: Optional[FakeImage] = None

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def was_called(self, method: str, *args) -> bool:
        """Return True if method was called with exactly the given args."""
        return any(
            c[0] == method and c[1:] == args
            for c in self.calls
        )

    def call_args_for(self, method: str) -> List[Tuple]:
        """Return list of arg tuples for all calls to method."""
        return [c[1:] for c in self.calls if c[0] == method]

    def inject_error(self, error: Optional[Xi_error] = None) -> None:
        """
        Queue an error to be raised on the next get_image() call.
        Call multiple times to inject multiple consecutive errors.
        """
        self._error_queue.append(
            error or Xi_error(10, "Injected timeout")
        )

    def reset_calls(self) -> None:
        """Clear the call log (useful between test phases)."""
        self.calls.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_device(self) -> None:
        self.calls.append(("open_device",))

    def open_device_by_SN(self, sn: str) -> None:
        self.calls.append(("open_device_by_SN", sn))

    def close_device(self) -> None:
        self.calls.append(("close_device",))

    def start_acquisition(self) -> None:
        self.calls.append(("start_acquisition",))

    def stop_acquisition(self) -> None:
        self.calls.append(("stop_acquisition",))

    def get_device_name(self) -> bytes:
        return b"MC203MG-SY-UB"

    def get_device_sn(self) -> bytes:
        return b"TEST000001"

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_imgdataformat(self, fmt: str) -> None:
        self.calls.append(("set_imgdataformat", fmt))

    def set_output_bit_depth(self, bd: str) -> None:
        self.calls.append(("set_output_bit_depth", bd))

    def set_acq_timing_mode(self, mode: str) -> None:
        self.calls.append(("set_acq_timing_mode", mode))

    def set_framerate(self, hz: float) -> None:
        self._framerate = hz
        self.calls.append(("set_framerate", hz))

    def set_exposure(self, us: int) -> None:
        self._exposure = us
        self.calls.append(("set_exposure", us))

    def set_buffers_queue_size(self, n: int) -> None:
        self.calls.append(("set_buffers_queue_size", n))

    def set_param(self, name: str, val) -> None:
        self._params[name] = val
        self.calls.append(("set_param", name, val))

    def set_width(self, v: int) -> None:
        self.calls.append(("set_width", v))

    def set_height(self, v: int) -> None:
        self.calls.append(("set_height", v))

    def set_offsetX(self, v: int) -> None:
        self.calls.append(("set_offsetX", v))

    def set_offsetY(self, v: int) -> None:
        self.calls.append(("set_offsetY", v))

    # ------------------------------------------------------------------
    # Configuration getters
    # ------------------------------------------------------------------

    def get_framerate(self)  -> float: return self._framerate
    def get_width_maximum(self)  -> int: return self._sensor_width
    def get_height_maximum(self) -> int: return self._sensor_height
    def get_width_increment(self)   -> int: return 4
    def get_height_increment(self)  -> int: return 4
    def get_offsetX_increment(self) -> int: return 4
    def get_offsetY_increment(self) -> int: return 4
    def get_buffers_queue_size_minimum(self) -> int: return 2
    def get_buffers_queue_size_maximum(self) -> int: return 64

    # ------------------------------------------------------------------
    # Image acquisition
    # ------------------------------------------------------------------

    def get_image(self, img: FakeImage, timeout: int = 1000) -> None:
        """
        Simulate frame delivery. Advances frame counters and timestamps.
        Raises the next queued Xi_error if one has been injected.
        """
        if self._error_queue:
            raise self._error_queue.pop(0)

        self._frame_counter += 1
        img.acq_nframe = self._frame_counter
        img.nframe     = self._frame_counter
        img.tsSec      = self._frame_counter // 100
        img.tsUSec     = (self._frame_counter % 100) * 10_000


# ---------------------------------------------------------------------------
# FakeXiapi - top-level module replacement
# ---------------------------------------------------------------------------

class FakeXiapi:
    """
    Top-level fake module, passed as _xiapi= to XimeaCamera.

    Usage::

        fake = FakeXiapi()
        cam  = XimeaCamera(config, _xiapi=fake)
        # Access the underlying FakeCamera via fake.camera_instance
        # after cam.open() has been called.
    """
    Xi_error = Xi_error
    Camera   = FakeCamera
    Image    = FakeImage
