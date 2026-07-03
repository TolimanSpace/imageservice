"""
config.py - Configuration management for the image acquisition system.

Two configuration layers are provided:

  SystemConfig   - Static hardware and infrastructure parameters loaded from
                   a JSON file at startup. Covers only things that are fixed
                   for the lifetime of the instrument: camera serial number,
                   sensor dimensions, file paths, and pipeline tuning.
                   Does not include camera mode or ROI geometry, which vary
                   between imaging sessions.

  SessionConfig  - All per-session operational parameters, supplied
                   programmatically at the start of each imaging session
                   (e.g. from a ground command or script). Includes camera
                   mode, ROI geometry, frame rate, exposure, and session
                   termination conditions.

Typical usage
-------------
At startup::

    system = SystemConfig.from_file("/etc/imageservice/config.json")

At the start of each imaging session::

    roi = make_rois(...)   # or a single RoiDefinition for SINGLE_ROI

    session = SessionConfig(
        mode=CameraMode.SINGLE_ROI,
        rois=[roi_def],
        frame_rate_hz=50.0,
        exposure_us=4_000,
        n_frames=500,
        session_id="obs_2026_001",
    )

    camera_config = system.to_camera_config(session)

In-flight config updates
------------------------
SystemConfig is designed to be updatable in-flight if a parameter needs
correcting. Use SystemConfig.save() to write an updated config atomically:

    cfg = SystemConfig.from_file(path)
    new_cfg = dataclasses.replace(cfg, serial_port="/dev/ttyS1")
    new_cfg.save(path)

save() writes to a temporary file alongside the target, validates it by
reloading it, then renames it over the original. The previous file is
preserved as <path>.bak. This ensures the system is never left with a
half-written or invalid config file.

Session termination
-------------------
Sessions end when the first of these conditions is met:
  - n_frames frames have been acquired  (if n_frames is not None)
  - duration_s seconds have elapsed     (if duration_s is not None)
  - stop() is called externally         (always available)

If both n_frames and duration_s are None the session runs until stopped.

ROI geometry
------------
config.py imports RoiDefinition and make_rois() from camera.py.
SessionConfig holds ROI geometry directly as List[RoiDefinition], so
there is no parallel ROI type hierarchy - camera.py types are the
single source of truth.

JSON file structure
-------------------
Required fields (ConfigError if absent):
  camera.serial_number
  sensor.width, sensor.height
  paths.data_dir, paths.log_dir

Optional fields fall back to the defaults documented on SystemConfig.
Extra keys are ignored with a warning for forward compatibility.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from camera import (
    CameraConfig,
    CameraMode,
    RoiDefinition,
    make_rois,
    single_roi_config,
    full_frame_config,
    multi_roi_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised for any configuration loading or validation failure."""


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemConfig:
    """
    Static hardware and infrastructure configuration loaded from a JSON file.

    Contains only parameters that are fixed for the lifetime of the instrument.
    Camera mode and ROI geometry are NOT here - they belong in SessionConfig
    since they vary between imaging sessions.

    Required JSON fields:
      camera.serial_number
      sensor.width, sensor.height
      paths.data_dir, paths.log_dir

    Optional JSON fields (defaults shown):
      camera.transport_buffer_size    8
      camera.output_bit_depth         12
      centroid.min_total_intensity    1.0
      writer.ring_buffer_n_frames     256
      writer.flush_timeout_s          5.0
      compression.batch_size_n        50
      compression.deflate_level       5
      serial.port                     None  (publisher disabled if absent)
      serial.baud_rate                115200
    """

    # --- Camera hardware ---
    camera_serial_number: str
    camera_transport_buffer_size: int
    camera_output_bit_depth: int

    # --- Sensor ---
    sensor_width: int
    sensor_height: int

    # --- Centroid ---
    centroid_min_total_intensity: float

    # --- Writer ---
    writer_ring_buffer_n_frames: int
    writer_flush_timeout_s: float

    # --- Compression ---
    compression_batch_size_n: int
    compression_deflate_level: int

    # --- Serial ---
    serial_port: Optional[str]
    serial_baud_rate: int

    # --- Paths ---
    data_dir: str
    log_dir: str

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "SystemConfig":
        """
        Load and validate a SystemConfig from a JSON file.

        Raises
        ------
        ConfigError
            If the file cannot be read, contains invalid JSON, is missing
            required fields, or contains out-of-range values.
        """
        logger.info("Loading system config from: %s", path)
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file {path}: {e}")
        except OSError as e:
            raise ConfigError(f"Cannot read config file {path}: {e}")

        cfg = cls._from_dict(raw)
        logger.info("SystemConfig loaded successfully.")
        return cfg

    @classmethod
    def _from_dict(cls, raw: dict) -> "SystemConfig":
        _warn_extra_keys(raw, ["camera", "sensor", "centroid",
                                "writer", "compression", "serial", "paths"],
                         "root")

        # camera
        cam = _section(raw, "camera", required=True)
        _warn_extra_keys(cam, ["serial_number", "transport_buffer_size",
                                "output_bit_depth"], "camera")
        _require(cam, ["serial_number"], "camera")
        bit_depth = cam.get("output_bit_depth", 12)
        if bit_depth != 12:
            raise ConfigError(
                f"camera.output_bit_depth must be 12, got {bit_depth}"
            )

        # sensor
        sen = _section(raw, "sensor", required=True)
        _warn_extra_keys(sen, ["width", "height"], "sensor")
        _require(sen, ["width", "height"], "sensor")

        # centroid
        cen = _section(raw, "centroid", required=False) or {}
        _warn_extra_keys(cen, ["min_total_intensity"], "centroid")

        # writer
        wri = _section(raw, "writer", required=False) or {}
        _warn_extra_keys(wri, ["ring_buffer_n_frames", "flush_timeout_s"],
                         "writer")

        # compression
        com = _section(raw, "compression", required=False) or {}
        _warn_extra_keys(com, ["batch_size_n", "deflate_level"], "compression")
        deflate_level = com.get("deflate_level", 5)
        _validate_range(deflate_level, 0, 9, "compression.deflate_level")

        # serial
        ser = _section(raw, "serial", required=False) or {}
        _warn_extra_keys(ser, ["port", "baud_rate"], "serial")

        # paths
        pth = _section(raw, "paths", required=True)
        _warn_extra_keys(pth, ["data_dir", "log_dir"], "paths")
        _require(pth, ["data_dir", "log_dir"], "paths")

        cfg = cls(
            camera_serial_number=_str(cam, "serial_number", "camera"),
            camera_transport_buffer_size=cam.get("transport_buffer_size", 8),
            camera_output_bit_depth=bit_depth,
            sensor_width=_int(sen, "width", "sensor"),
            sensor_height=_int(sen, "height", "sensor"),
            centroid_min_total_intensity=float(
                cen.get("min_total_intensity", 1.0)),
            writer_ring_buffer_n_frames=int(
                wri.get("ring_buffer_n_frames", 256)),
            writer_flush_timeout_s=float(wri.get("flush_timeout_s", 5.0)),
            compression_batch_size_n=int(com.get("batch_size_n", 50)),
            compression_deflate_level=int(deflate_level),
            serial_port=ser.get("port", None),
            serial_baud_rate=int(ser.get("baud_rate", 115200)),
            data_dir=_str(pth, "data_dir", "paths"),
            log_dir=_str(pth, "log_dir", "paths"),
        )
        cfg._validate_paths()
        return cfg

    # ------------------------------------------------------------------
    # Saving (atomic, with backup)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Atomically save this SystemConfig to a JSON file.

        Writes to a temporary file in the same directory, validates it by
        reloading it, then renames it over the target. The previous file
        is preserved as <path>.bak.

        This ensures the system is never left with a half-written or invalid
        config file - if the write or validation fails the original is
        untouched.

        Parameters
        ----------
        path : str
            Destination file path (typically the same path it was loaded from).

        Raises
        ------
        ConfigError
            If the written file fails to reload cleanly (should not happen
            unless there is a serialisation bug).
        OSError
            If the file cannot be written or renamed.
        """
        dir_ = os.path.dirname(os.path.abspath(path))
        as_dict = self._to_dict()

        # Write to a temp file in the same directory (same filesystem → atomic rename)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(as_dict, f, indent=2)

            # Validate by reloading before committing
            try:
                SystemConfig.from_file(tmp_path)
            except ConfigError as e:
                raise ConfigError(
                    f"Validation of written config failed (this is a bug): {e}"
                )

            # Backup the existing file if present
            if os.path.exists(path):
                os.replace(path, path + ".bak")
                logger.info("Previous config backed up to: %s.bak", path)

            # Atomic rename
            os.replace(tmp_path, path)
            logger.info("SystemConfig saved to: %s", path)

        except Exception:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _to_dict(self) -> dict:
        """Serialise this SystemConfig back to a JSON-compatible dict."""
        d: dict = {
            "camera": {
                "serial_number": self.camera_serial_number,
                "transport_buffer_size": self.camera_transport_buffer_size,
                "output_bit_depth": self.camera_output_bit_depth,
            },
            "sensor": {
                "width": self.sensor_width,
                "height": self.sensor_height,
            },
            "centroid": {
                "min_total_intensity": self.centroid_min_total_intensity,
            },
            "writer": {
                "ring_buffer_n_frames": self.writer_ring_buffer_n_frames,
                "flush_timeout_s": self.writer_flush_timeout_s,
            },
            "compression": {
                "batch_size_n": self.compression_batch_size_n,
                "deflate_level": self.compression_deflate_level,
            },
            "paths": {
                "data_dir": self.data_dir,
                "log_dir": self.log_dir,
            },
        }
        serial: dict = {"baud_rate": self.serial_baud_rate}
        if self.serial_port is not None:
            serial["port"] = self.serial_port
        d["serial"] = serial
        return d

    # ------------------------------------------------------------------
    # Bridge to camera.py
    # ------------------------------------------------------------------

    def to_camera_config(self, session: "SessionConfig") -> CameraConfig:
        """
        Construct a CameraConfig from this SystemConfig and a SessionConfig.

        This is the single point where the two config layers are combined.
        acquisition.py calls this once at session start.

        Parameters
        ----------
        session : SessionConfig

        Returns
        -------
        CameraConfig ready to pass to XimeaCamera.
        """
        if session.mode is CameraMode.SINGLE_ROI:
            roi = session.rois[0]
            return single_roi_config(
                width=roi.width,
                height=roi.height,
                offset_x=roi.offset_x,
                offset_y=roi.offset_y,
                exposure_us=session.exposure_us,
                frame_rate_hz=session.frame_rate_hz,
                label=roi.label,
                serial_number=self.camera_serial_number,
                transport_buffer_size=self.camera_transport_buffer_size,
            )
        elif session.mode is CameraMode.MULTI_ROI:
            return multi_roi_config(
                rois=session.rois,
                exposure_us=session.exposure_us,
                frame_rate_hz=session.frame_rate_hz,
                serial_number=self.camera_serial_number,
                transport_buffer_size=self.camera_transport_buffer_size,
            )
        else:  # FULL_FRAME
            return full_frame_config(
                exposure_us=session.exposure_us,
                frame_rate_hz=session.frame_rate_hz,
                serial_number=self.camera_serial_number,
                transport_buffer_size=self.camera_transport_buffer_size,
            )

    # ------------------------------------------------------------------
    # Convenience properties for centroid module
    # ------------------------------------------------------------------

    def centre_roi_offset(
        self, session: "SessionConfig"
    ) -> tuple:
        """
        Return (offset_x, offset_y) of the centre ROI for a given session.

        For MULTI_ROI this is the centre region (index 4) of the ROI list.
        For SINGLE_ROI this is the single ROI's offset.
        For FULL_FRAME this is (0, 0).

        Returns
        -------
        (offset_x, offset_y) : tuple of int
        """
        if session.mode is CameraMode.MULTI_ROI and session.rois:
            centre = session.rois[4]   # index 4 is always "centre" per make_rois()
            return (centre.offset_x, centre.offset_y)
        if session.mode is CameraMode.SINGLE_ROI and session.rois:
            return (session.rois[0].offset_x, session.rois[0].offset_y)
        return (0, 0)

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        for label, path in [("data_dir", self.data_dir),
                             ("log_dir",  self.log_dir)]:
            if not os.path.isdir(path):
                logger.warning(
                    "paths.%s does not exist: %s "
                    "(will be created on first write if possible).",
                    label, path,
                )

    # ------------------------------------------------------------------
    # Example file generator
    # ------------------------------------------------------------------

    @staticmethod
    def write_example(path: str) -> None:
        """Write an annotated example JSON config file."""
        example = {
            "camera": {
                "_comment": "serial_number is required",
                "serial_number": "XXXXXXXX",
                "transport_buffer_size": 8,
                "output_bit_depth": 12,
            },
            "sensor": {
                "width": 4512,
                "height": 4512,
            },
            "centroid": {
                "min_total_intensity": 1.0,
            },
            "writer": {
                "ring_buffer_n_frames": 256,
                "flush_timeout_s": 5.0,
            },
            "compression": {
                "batch_size_n": 50,
                "deflate_level": 5,
            },
            "serial": {
                "_comment": "Omit 'port' to disable serial publishing",
                "port": "/dev/ttyS0",
                "baud_rate": 115200,
            },
            "paths": {
                "_comment": "data_dir and log_dir are required",
                "data_dir": "/data/imageservice",
                "log_dir": "/var/log/imageservice",
            },
        }
        with open(path, "w") as f:
            json.dump(example, f, indent=2)
        logger.info("Example config written to: %s", path)


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    """
    Per-session operational parameters, supplied programmatically.

    Not stored in the JSON file - these are commanded at the start of each
    imaging session.

    Session termination
    -------------------
    A session ends when the first of these conditions is met:
      - n_frames frames acquired    (if n_frames is not None)
      - duration_s seconds elapsed  (if duration_s is not None)
      - stop() called externally    (always available)

    If both n_frames and duration_s are None the session runs until stopped.
    If both are set, the session ends on whichever condition is reached first.

    Attributes
    ----------
    mode : CameraMode
        Acquisition mode for this session.
    rois : List[RoiDefinition] or None
        ROI geometry. Required for SINGLE_ROI (exactly 1) and MULTI_ROI
        (exactly 9, built with make_rois()). Must be None for FULL_FRAME.
    frame_rate_hz : float
        Target frame rate in Hz. Must be positive.
    exposure_us : int
        Exposure time in microseconds. Must be less than
        1_000_000 / frame_rate_hz.
    n_frames : int or None
        Maximum number of frames to acquire. None means no frame limit.
    duration_s : float or None
        Maximum session duration in seconds. None means no time limit.
    session_id : str
        Short identifier used as a prefix for output filenames,
        e.g. "obs_2026_177_001". Must not be empty.
    """
    mode: CameraMode
    rois: Optional[List[RoiDefinition]]
    frame_rate_hz: float
    exposure_us: int
    n_frames: Optional[int]
    duration_s: Optional[float]
    session_id: str

    def __post_init__(self) -> None:
        if self.frame_rate_hz <= 0.0:
            raise ConfigError(
                f"frame_rate_hz must be positive, got {self.frame_rate_hz}"
            )
        if self.exposure_us <= 0:
            raise ConfigError(
                f"exposure_us must be positive, got {self.exposure_us}"
            )
        frame_period_us = 1_000_000.0 / self.frame_rate_hz
        if self.exposure_us >= frame_period_us:
            raise ConfigError(
                f"exposure_us ({self.exposure_us} us) must be less than the "
                f"frame period ({frame_period_us:.0f} us at "
                f"{self.frame_rate_hz} Hz)."
            )
        if self.n_frames is not None and self.n_frames <= 0:
            raise ConfigError(
                f"n_frames must be positive or None, got {self.n_frames}"
            )
        if self.duration_s is not None and self.duration_s <= 0.0:
            raise ConfigError(
                f"duration_s must be positive or None, got {self.duration_s}"
            )
        if self.n_frames is None and self.duration_s is None:
            logger.warning(
                "SessionConfig: both n_frames and duration_s are None - "
                "session will run until stop() is called."
            )
        if not self.session_id:
            raise ConfigError("session_id must not be empty.")
        self._validate_rois()

    def _validate_rois(self) -> None:
        if self.mode is CameraMode.SINGLE_ROI:
            if self.rois is None or len(self.rois) != 1:
                raise ConfigError(
                    "SINGLE_ROI mode requires exactly 1 RoiDefinition in rois."
                )
            if not self.rois[0].is_data:
                raise ConfigError(
                    "SINGLE_ROI: rois[0].is_data must be True."
                )
        elif self.mode is CameraMode.MULTI_ROI:
            if self.rois is None or len(self.rois) != 9:
                raise ConfigError(
                    "MULTI_ROI mode requires exactly 9 RoiDefinition objects "
                    "in rois (use make_rois() to construct them)."
                )
        elif self.mode is CameraMode.FULL_FRAME:
            if self.rois is not None:
                raise ConfigError(
                    "FULL_FRAME mode requires rois=None."
                )

    @property
    def has_frame_limit(self) -> bool:
        """True if the session will stop after n_frames frames."""
        return self.n_frames is not None

    @property
    def has_time_limit(self) -> bool:
        """True if the session will stop after duration_s seconds."""
        return self.duration_s is not None

    @property
    def is_bounded(self) -> bool:
        """True if the session has at least one termination condition."""
        return self.has_frame_limit or self.has_time_limit


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _section(
    d: dict,
    key: str,
    required: bool,
    missing_msg: Optional[str] = None,
) -> Optional[dict]:
    val = d.get(key)
    if val is None:
        if required:
            raise ConfigError(missing_msg or
                              f"Required config section '{key}' is missing.")
        return None
    if not isinstance(val, dict):
        raise ConfigError(
            f"Config section '{key}' must be a JSON object, "
            f"got {type(val).__name__}."
        )
    return val


def _require(d: dict, keys: List[str], section: str) -> None:
    for key in keys:
        if key not in d:
            raise ConfigError(
                f"Required field '{section}.{key}' is missing from config."
            )


def _int(d: dict, key: str, section: str) -> int:
    val = d[key]
    if not isinstance(val, int):
        raise ConfigError(
            f"'{section}.{key}' must be an integer, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _str(d: dict, key: str, section: str) -> str:
    val = d[key]
    if not isinstance(val, str):
        raise ConfigError(
            f"'{section}.{key}' must be a string, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _validate_choice(value: str, choices: List[str], field: str) -> None:
    if value not in choices:
        raise ConfigError(
            f"'{field}' must be one of {choices}, got {value!r}."
        )


def _validate_range(value: float, lo: float, hi: float, field: str) -> None:
    if not (lo <= value <= hi):
        raise ConfigError(
            f"'{field}' must be in [{lo}, {hi}], got {value}."
        )


def _warn_extra_keys(d: dict, known: List[str], section: str) -> None:
    for key in d:
        if key not in known and not key.startswith("_"):
            logger.warning(
                "Unknown key '%s.%s' in config - ignored.", section, key
            )
