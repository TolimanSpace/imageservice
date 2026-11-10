"""
sim_xiapi.py - A simulated Ximea camera, usable at runtime in place of ximea.xiapi.

XimeaCamera (workers/camera.py) uses this module instead of the real xiapi
when TOLIMAN_CAMERA_SIM is set, so the whole service (supervisor,
acquisition, centroiding, writers, the centroid datagrams to the pointing
service) runs without a camera. Unlike tests/fake_xiapi.py, which replays
frames a test loads into it, this camera renders each frame itself:

  - a star with a Gaussian PSF (pixel-integrated, so sub-pixel positions are
    exact), at the position the pointing plant reports, or at a fixed one;
  - shot noise (Poisson) and read noise (Gaussian), then the camera model's
    QE, gain, bias and 12-bit clipping from sim_converter.py;
  - the configured ROI (width, height, offsets), exposure and frame rate.
    Frames are paced in real time on a fixed grid of 1 / frame rate; a
    reader that falls behind gets the latest frame and the skipped ones
    count as dropped (acq_nframe jumps), as when the real transport buffer
    overflows. The buffer's queueing is not modelled.
  - acq_nframe counts from 1 at start_acquisition(); tsSec/tsUSec are a
    camera clock that starts when the device is opened.

USB and the XIMEA protocol are not emulated: everything above xiapi runs
unchanged.

Star position
-------------
The star sits at (sensor_width / 2 + x, sensor_height / 2 + y) in sensor
pixel coordinates, in the convention of workers/centroid.py (pixel i is
centred on coordinate i), so compute_pointing_error() returns dx = x,
dy = y for a noiseless frame. x and y come from:

  TOLIMAN_CAMERA_SIM=<socket path>   the pointing plant's query socket
      (SEC repo: tools/pointing-sim, `pointing-sim --plant-socket <path>`).
      Queried once per frame at the end of its exposure; x and y are the
      plant's star offsets in pixels. If the plant does not answer,
      get_image() raises Xi_error like a camera timeout.
  TOLIMAN_CAMERA_SIM=static:<x>,<y>  a fixed position (no plant needed).

Other settings (environment, read when the camera is opened):

  TOLIMAN_CAMERA_SIM_MODEL       camera model in sim_converter.CAMERA_MODELS
                                 (default MC203MG-SY-UB); also the device name
  TOLIMAN_CAMERA_SIM_SENSOR      sensor size WxH (default 4512x4512); must match
                                 sensor.width/height in the service's config
  TOLIMAN_CAMERA_SIM_FLUX        star photons per second (default 4e6: a peak of
                                 about 2000 ADU above bias at 4 ms)
  TOLIMAN_CAMERA_SIM_SIGMA       PSF standard deviation in pixels (default 1.5)
  TOLIMAN_CAMERA_SIM_READ_NOISE  read noise in electrons RMS (default 2)
  TOLIMAN_CAMERA_SIM_SKY         background photons per pixel per second (default 0)
  TOLIMAN_CAMERA_SIM_SEED        random seed (default: unseeded)

Plant query protocol (little endian, Unix datagrams): send [0x10, 1, 0, 0]
from a bound socket; the reply is struct '<BBBBQ5f' = kind 0x11, version 1,
board state, 0, Unix time ns, star x px, star y px, three stack positions.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

ENV = "TOLIMAN_CAMERA_SIM"

XI_TIMEOUT = 10  # XI_TIMEOUT status of the real API

PLANT_REQUEST = bytes([0x10, 1, 0, 0])
_PLANT_REPLY = struct.Struct("<BBBBQ5f")


class Xi_error(Exception):
    """Matches the real xiapi.Xi_error interface (a .status code)."""

    def __init__(self, status: int = XI_TIMEOUT, message: str = "") -> None:
        self.status = status
        super().__init__(f"Xi_error status={status}: {message}")


# ---------------------------------------------------------------------------
# Settings and star sources
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimSettings:
    source: str                       # plant socket path or "static:x,y"
    model: str = "MC203MG-SY-UB"
    sensor_width: int = 4512
    sensor_height: int = 4512
    flux_photons_per_s: float = 4e6
    psf_sigma_px: float = 1.5
    read_noise_e: float = 2.0
    sky_photons_per_px_s: float = 0.0
    seed: Optional[int] = None

    @classmethod
    def from_env(cls, env=os.environ) -> "SimSettings":
        source = env.get(ENV, "")
        if not source:
            raise ValueError(f"{ENV} is not set")
        w, h = env.get(f"{ENV}_SENSOR", "4512x4512").lower().split("x")
        seed = env.get(f"{ENV}_SEED")
        return cls(
            source=source,
            model=env.get(f"{ENV}_MODEL", cls.model),
            sensor_width=int(w),
            sensor_height=int(h),
            flux_photons_per_s=float(env.get(f"{ENV}_FLUX", cls.flux_photons_per_s)),
            psf_sigma_px=float(env.get(f"{ENV}_SIGMA", cls.psf_sigma_px)),
            read_noise_e=float(env.get(f"{ENV}_READ_NOISE", cls.read_noise_e)),
            sky_photons_per_px_s=float(env.get(f"{ENV}_SKY", cls.sky_photons_per_px_s)),
            seed=int(seed) if seed else None,
        )


def enabled(env=os.environ) -> bool:
    """True if TOLIMAN_CAMERA_SIM selects the simulated camera."""
    return bool(env.get(ENV))


class PlantClient:
    """Asks the pointing plant where the star is (tools/pointing-sim/python/plant.py)."""

    def __init__(self, path: str, timeout_s: float = 0.2) -> None:
        self.path = path
        self._dir = tempfile.TemporaryDirectory(prefix="sim-xiapi-")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(os.path.join(self._dir.name, "camera.sock"))  # the plant replies here
        self.timeout_s = timeout_s

    def star(self) -> Tuple[float, float]:
        """(x, y) of the star in pixels from the centre; raises OSError or ValueError."""
        # Drop late replies to earlier queries that timed out
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recv(64)
        except (BlockingIOError, InterruptedError):
            pass
        self.sock.settimeout(self.timeout_s)
        self.sock.sendto(PLANT_REQUEST, self.path)
        data = self.sock.recv(64)
        if len(data) != _PLANT_REPLY.size:
            raise ValueError(f"plant reply has {len(data)} bytes")
        kind, version, _state, _, _t, x, y, *_ = _PLANT_REPLY.unpack(data)
        if (kind, version) != (0x11, 1):
            raise ValueError(f"unexpected plant reply kind {kind:#x} version {version}")
        return float(x), float(y)

    def close(self) -> None:
        self.sock.close()
        self._dir.cleanup()


def _star_source(spec: str) -> Tuple[Callable[[], Tuple[float, float]], Callable[[], None]]:
    """(get position, close) for a TOLIMAN_CAMERA_SIM value."""
    if spec.startswith("static:"):
        x, y = (float(v) for v in spec[len("static:"):].split(","))
        return (lambda: (x, y)), (lambda: None)
    plant = PlantClient(spec)
    return plant.star, plant.close


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_erf = np.frompyfunc(math.erf, 1, 1)


def _pixel_fractions(n: int, offset: int, centre: float, sigma: float) -> np.ndarray:
    """Fraction of a 1-D Gaussian falling on each of n pixels starting at `offset`."""
    edges = (np.arange(n + 1, dtype=np.float64) + offset - 0.5 - centre) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + _erf(edges).astype(np.float64))
    return np.diff(cdf)


def render_photons(
    width: int, height: int, offset_x: int, offset_y: int,
    star_x: float, star_y: float, settings: SimSettings, exposure_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Photon counts (int64, noise included) on a width x height region at
    (offset_x, offset_y), for a star at sensor coordinates (star_x, star_y).

    Read noise is added in photon units (electrons / QE) because
    sim_converter.photons_to_adu() applies QE after the simulator.
    """
    from sim_converter import get_camera_model

    fx = _pixel_fractions(width, offset_x, star_x, settings.psf_sigma_px)
    fy = _pixel_fractions(height, offset_y, star_y, settings.psf_sigma_px)
    mean = np.outer(fy, fx) * (settings.flux_photons_per_s * exposure_s)
    mean += settings.sky_photons_per_px_s * exposure_s
    photons = rng.poisson(mean).astype(np.float64)
    if settings.read_noise_e > 0:
        qe = get_camera_model(settings.model).quantum_efficiency
        photons += rng.normal(0.0, settings.read_noise_e / qe, size=photons.shape)
    return np.rint(photons).astype(np.int64)


# ---------------------------------------------------------------------------
# xiapi.Image / xiapi.Camera
# ---------------------------------------------------------------------------

class Image:
    """Mimics xiapi.Image: frame metadata and the pixel data of the last get_image()."""

    def __init__(self) -> None:
        self._data: Optional[np.ndarray] = None
        self.acq_nframe: int = 0
        self.nframe: int = 0
        self.tsSec: int = 0
        self.tsUSec: int = 0
        self.width: int = 0
        self.height: int = 0

    def get_image_data_numpy(self) -> np.ndarray:
        if self._data is None:
            raise Xi_error(XI_TIMEOUT, "no image acquired")
        return self._data


class Camera:
    """Mimics the subset of xiapi.Camera that workers/camera.py uses."""

    def __init__(self, settings: Optional[SimSettings] = None) -> None:
        self._settings = settings
        self._open = False
        self._acquiring = False
        self._framerate = 10.0
        self._exposure_us = 1000
        self._width = self._height = 0
        self._offset_x = self._offset_y = 0
        self._buffers = 8
        self._clock_origin_ns = 0
        self._t0 = 0.0
        self._last_index = -1
        self._star = None
        self._close_star = None
        self._rng: Optional[np.random.Generator] = None
        self._model = None

    # Lifecycle -------------------------------------------------------------

    def open_device(self) -> None:
        if self._open:
            raise Xi_error(57, "device already open")
        from sim_converter import get_camera_model

        if self._settings is None:
            self._settings = SimSettings.from_env()
        s = self._settings
        self._model = get_camera_model(s.model)
        self._star, self._close_star = _star_source(s.source)
        self._rng = np.random.default_rng(s.seed)
        self._width, self._height = s.sensor_width, s.sensor_height
        self._offset_x = self._offset_y = 0
        self._clock_origin_ns = time.monotonic_ns()
        self._open = True

    def open_device_by_SN(self, sn) -> None:
        self.open_device()

    def close_device(self) -> None:
        if self._close_star is not None:
            self._close_star()
        self._star = self._close_star = None
        self._open = False
        self._acquiring = False

    def start_acquisition(self) -> None:
        self._require_open()
        self._acquiring = True
        self._t0 = time.monotonic()
        self._last_index = -1

    def stop_acquisition(self) -> None:
        self._acquiring = False

    def get_device_name(self) -> bytes:
        return (self._settings.model if self._settings else SimSettings.model).encode()

    def get_device_sn(self) -> bytes:
        return b"SIM00001"

    # Settings -------------------------------------------------------------

    def set_imgdataformat(self, fmt: str) -> None:
        if fmt != "XI_MONO16":
            raise Xi_error(106, f"simulated camera supports XI_MONO16 only, not {fmt}")

    def set_output_bit_depth(self, bpp: str) -> None:
        if bpp != "XI_BPP_12":
            raise Xi_error(106, f"simulated camera supports XI_BPP_12 only, not {bpp}")

    def set_acq_timing_mode(self, mode: str) -> None:
        pass

    def set_framerate(self, hz: float) -> None:
        if hz <= 0:
            raise Xi_error(11, f"invalid frame rate {hz}")
        self._framerate = float(hz)

    def get_framerate(self) -> float:
        return self._framerate

    def set_exposure(self, us: int) -> None:
        self._exposure_us = int(us)

    def get_exposure(self) -> int:
        return self._exposure_us

    def set_buffers_queue_size(self, n: int) -> None:
        self._buffers = int(n)

    def get_buffers_queue_size_minimum(self) -> int:
        return 1

    def get_buffers_queue_size_maximum(self) -> int:
        return 64

    def set_param(self, name: str, value) -> None:
        pass  # multi-region parameters: not simulated (one region only)

    def set_width(self, v: int) -> None:
        self._width = int(v)

    def set_height(self, v: int) -> None:
        self._height = int(v)

    def set_offsetX(self, v: int) -> None:
        self._offset_x = int(v)

    def set_offsetY(self, v: int) -> None:
        self._offset_y = int(v)

    def get_width_maximum(self) -> int:
        return self._sensor_w

    def get_height_maximum(self) -> int:
        return self._sensor_h

    def get_width_increment(self) -> int:
        return 4

    def get_height_increment(self) -> int:
        return 4

    def get_offsetX_increment(self) -> int:
        return 4

    def get_offsetY_increment(self) -> int:
        return 4

    # Acquisition ----------------------------------------------------------

    def get_image(self, img: Image, timeout: int = 1000) -> None:
        if not self._acquiring:
            raise Xi_error(45, "acquisition not started")
        self._check_extent(self._width, self._offset_x, self._sensor_w, "width + offsetX")
        self._check_extent(self._height, self._offset_y, self._sensor_h, "height + offsetY")
        period = 1.0 / self._framerate
        now = time.monotonic()
        # Frame n is read out at t0 + (n + 1) * period; take the next one not yet
        # delivered, or the latest one if the reader has fallen behind.
        index = max(self._last_index + 1, int((now - self._t0) / period) - 1)
        ready = self._t0 + (index + 1) * period
        if ready - now > timeout / 1000.0:
            time.sleep(timeout / 1000.0)
            raise Xi_error(XI_TIMEOUT, "timeout")
        if ready > now:
            time.sleep(ready - now)

        try:
            x, y = self._star()
        except (OSError, ValueError) as e:
            raise Xi_error(XI_TIMEOUT, f"plant did not answer: {e}") from e

        from sim_converter import photons_to_adu

        photons = render_photons(
            self._width, self._height, self._offset_x, self._offset_y,
            self._sensor_w / 2.0 + x, self._sensor_h / 2.0 + y,
            self._settings, self._exposure_us / 1e6, self._rng,
        )
        img._data = photons_to_adu(photons, self._model)
        img.width, img.height = self._width, self._height
        self._last_index = index
        img.acq_nframe = index + 1
        img.nframe = index + 1
        ts_us = (time.monotonic_ns() - self._clock_origin_ns) // 1000
        img.tsSec, img.tsUSec = divmod(ts_us, 1_000_000)

    # Helpers --------------------------------------------------------------

    @property
    def _sensor_w(self) -> int:
        return self._settings.sensor_width if self._settings else SimSettings.sensor_width

    @property
    def _sensor_h(self) -> int:
        return self._settings.sensor_height if self._settings else SimSettings.sensor_height

    def _require_open(self) -> None:
        if not self._open:
            raise Xi_error(56, "device not open")

    @staticmethod
    def _check_extent(size: int, offset: int, sensor: int, name: str) -> None:
        if size + offset > sensor or offset < 0 or size <= 0:
            raise Xi_error(11, f"{name}: region {size}+{offset} exceeds sensor {sensor}")
