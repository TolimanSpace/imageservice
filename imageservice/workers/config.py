"""
config.py — Configuration management for the image acquisition system.

Two configuration layers are provided:

  SystemConfig   — Static hardware and system parameters loaded from a JSON
                   file at startup. Describes the camera, sensor geometry,
                   ROI layout, file paths, and pipeline tuning parameters.
                   Does not change between imaging sessions.

  SessionConfig  — Per-session acquisition parameters supplied programmatically
                   (e.g. from a ground command or command-line arguments) at
                   the start of each imaging session. Includes frame rate,
                   exposure, frame count, and a session identifier.

Typical usage
-------------
At startup::

    system = SystemConfig.from_file("/etc/startracker/config.json")

At the start of each imaging session::

    session = SessionConfig(
        frame_rate_hz=50.0,
        exposure_us=4_000,
        n_frames=500,
        session_id="obs_2026_001",
    )

    camera_config = system.to_camera_config(session)

The two configs are kept separate so that the JSON file can be written once
during instrument integration and never modified in flight, while session
parameters are commanded from the ground without touching the filesystem.

ROI geometry
------------
config.py imports RoiDefinition and make_rois() directly from camera.py.
SystemConfig.to_camera_config() calls make_rois() to produce the
List[RoiDefinition] that CameraConfig requires, so there is no parallel
ROI type hierarchy — camera.py types are the single source of truth.

JSON file structure
-------------------
Required fields have no default and will raise ConfigError if absent.
Optional fields fall back to the defaults documented in SystemConfig.
All fields are grouped under top-level keys matching the section names
below. An example file is produced by SystemConfig.write_example().

Error handling
--------------
ConfigError is raised for:
  - File not found or unreadable
  - Invalid JSON syntax
  - Missing required fields
  - Values that fail type or range validation

Unexpected extra keys in the JSON are ignored with a warning, not an error,
to allow forward compatibility as new fields are added.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

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
# MultiRoiConfig — compact geometry store for JSON persistence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiRoiConfig:
    """
    Compact representation of the multi-ROI quincunx geometry, as stored in
    JSON.  Not used directly by the camera — call to_roi_list() to produce
    the List[RoiDefinition] that CameraConfig and make_rois() expect.

    Mirrors the parameters of make_rois() exactly so that to_roi_list() is
    a direct pass-through with no translation logic.

    Attributes
    ----------
    sensor_width, sensor_height : int
        Full sensor dimensions in pixels.
    offset_x, offset_y : int
        Offset to the top-left corner of the quincunx pattern on the sensor.
    sidelobe_width, sidelobe_height : int
        Width and height of the corner ROIs.
    core_width, core_height : int
        Width and height of the centre ROI.
    full_width, full_height : int
        Total width and height of the quincunx pattern.
    """
    sensor_width: int
    sensor_height: int
    offset_x: int
    offset_y: int
    sidelobe_width: int
    sidelobe_height: int
    core_width: int
    core_height: int
    full_width: int
    full_height: int

    def to_roi_list(self) -> List[RoiDefinition]:
        """
        Construct the 9-element List[RoiDefinition] for CameraConfig.

        Delegates entirely to make_rois() in camera.py — no geometry logic
        is duplicated here.
        """
        return make_rois(
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            sidelobe_width=self.sidelobe_width,
            sidelobe_height=self.sidelobe_height,
            core_width=self.core_width,
            core_height=self.core_height,
            full_width=self.full_width,
            full_height=self.full_height,
        )

    @classmethod
    def from_dict(cls, d: dict, path: str) -> "MultiRoiConfig":
        _require(d, [
            "sensor_width", "sensor_height",
            "offset_x", "offset_y",
            "sidelobe_width", "sidelobe_height",
            "core_width", "core_height",
            "full_width", "full_height",
        ], path)
        return cls(
            sensor_width=_int(d, "sensor_width", path),
            sensor_height=_int(d, "sensor_height", path),
            offset_x=_int(d, "offset_x", path),
            offset_y=_int(d, "offset_y", path),
            sidelobe_width=_int(d, "sidelobe_width", path),
            sidelobe_height=_int(d, "sidelobe_height", path),
            core_width=_int(d, "core_width", path),
            core_height=_int(d, "core_height", path),
            full_width=_int(d, "full_width", path),
            full_height=_int(d, "full_height", path),
        )


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemConfig:
    """
    Static system configuration loaded from a JSON file.

    Required fields (ConfigError if absent):
      camera.serial_number   — Ximea camera serial number string.
      camera.mode            — "SINGLE_ROI", "MULTI_ROI", or "FULL_FRAME".
      sensor.width           — Full sensor width in pixels.
      sensor.height          — Full sensor height in pixels.
      paths.data_dir         — Directory for output data files.
      paths.log_dir          — Directory for log files.

    Optional fields (defaults shown):
      camera.transport_buffer_size  — 8
      camera.output_bit_depth       — 12 (must be 12; validated)
      roi.*                         — Required if mode is SINGLE_ROI or MULTI_ROI.
      centroid.min_total_intensity  — 1.0
      writer.ring_buffer_n_frames   — 256
      writer.flush_timeout_s        — 5.0
      compression.batch_size_n      — 50
      compression.deflate_level     — 5
      serial.port                   — None (serial publisher disabled if absent)
      serial.baud_rate              — 115200
    """

    # --- Camera ---
    camera_serial_number: str
    camera_mode: str                   # "SINGLE_ROI" | "MULTI_ROI" | "FULL_FRAME"
    camera_transport_buffer_size: int
    camera_output_bit_depth: int

    # --- Sensor ---
    sensor_width: int
    sensor_height: int

    # --- ROI geometry (mode-dependent) ---
    # For SINGLE_ROI: a single RoiDefinition (from camera.py)
    # For MULTI_ROI:  a MultiRoiConfig (compact store; call .to_roi_list())
    # For FULL_FRAME: both are None
    single_roi: Optional[RoiDefinition]
    multi_roi: Optional[MultiRoiConfig]

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
    # Construction
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

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "SystemConfig":
        _warn_extra_keys(raw, ["camera", "sensor", "roi", "centroid",
                                "writer", "compression", "serial", "paths"],
                         "root")

        # --- camera ---
        cam = _section(raw, "camera", required=True)
        _warn_extra_keys(cam, ["serial_number", "mode",
                                "transport_buffer_size", "output_bit_depth"],
                         "camera")
        _require(cam, ["serial_number", "mode"], "camera")
        camera_mode = _str(cam, "mode", "camera").upper()
        _validate_choice(camera_mode, ["SINGLE_ROI", "MULTI_ROI", "FULL_FRAME"],
                         "camera.mode")
        camera_bit_depth = cam.get("output_bit_depth", 12)
        if camera_bit_depth != 12:
            raise ConfigError(
                f"camera.output_bit_depth must be 12, got {camera_bit_depth}"
            )

        # --- sensor ---
        sen = _section(raw, "sensor", required=True)
        _warn_extra_keys(sen, ["width", "height"], "sensor")
        _require(sen, ["width", "height"], "sensor")
        sensor_width  = _int(sen, "width",  "sensor")
        sensor_height = _int(sen, "height", "sensor")

        # --- roi (structure depends on mode) ---
        single_roi_def: Optional[RoiDefinition] = None
        multi_roi_cfg: Optional[MultiRoiConfig] = None

        if camera_mode == "SINGLE_ROI":
            roi_raw = _section(raw, "roi", required=True,
                               missing_msg="roi section is required for SINGLE_ROI mode.")
            _require(roi_raw, ["width", "height", "offset_x", "offset_y"], "roi")
            single_roi_def = RoiDefinition(
                width=_int(roi_raw, "width", "roi"),
                height=_int(roi_raw, "height", "roi"),
                offset_x=_int(roi_raw, "offset_x", "roi"),
                offset_y=_int(roi_raw, "offset_y", "roi"),
                label=roi_raw.get("label", "single_roi"),
                is_data=True,
            )
        elif camera_mode == "MULTI_ROI":
            roi_raw = _section(raw, "roi", required=True,
                               missing_msg="roi section is required for MULTI_ROI mode.")
            multi_roi_cfg = MultiRoiConfig.from_dict(roi_raw, "roi")
        else:  # FULL_FRAME
            if "roi" in raw:
                logger.warning(
                    "roi section present in config but camera.mode is FULL_FRAME "
                    "— roi section will be ignored."
                )

        # --- centroid ---
        cen = _section(raw, "centroid", required=False) or {}
        _warn_extra_keys(cen, ["min_total_intensity"], "centroid")

        # --- writer ---
        wri = _section(raw, "writer", required=False) or {}
        _warn_extra_keys(wri, ["ring_buffer_n_frames", "flush_timeout_s"], "writer")

        # --- compression ---
        com = _section(raw, "compression", required=False) or {}
        _warn_extra_keys(com, ["batch_size_n", "deflate_level"], "compression")
        deflate_level = com.get("deflate_level", 5)
        _validate_range(deflate_level, 0, 9, "compression.deflate_level")

        # --- serial ---
        ser = _section(raw, "serial", required=False) or {}
        _warn_extra_keys(ser, ["port", "baud_rate"], "serial")

        # --- paths ---
        pth = _section(raw, "paths", required=True)
        _warn_extra_keys(pth, ["data_dir", "log_dir"], "paths")
        _require(pth, ["data_dir", "log_dir"], "paths")

        cfg = cls(
            camera_serial_number=_str(cam, "serial_number", "camera"),
            camera_mode=camera_mode,
            camera_transport_buffer_size=cam.get("transport_buffer_size", 8),
            camera_output_bit_depth=camera_bit_depth,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            single_roi=single_roi_def,
            multi_roi=multi_roi_cfg,
            centroid_min_total_intensity=float(
                cen.get("min_total_intensity", 1.0)),
            writer_ring_buffer_n_frames=int(
                wri.get("ring_buffer_n_frames", 256)),
            writer_flush_timeout_s=float(
                wri.get("flush_timeout_s", 5.0)),
            compression_batch_size_n=int(com.get("batch_size_n", 50)),
            compression_deflate_level=int(deflate_level),
            serial_port=ser.get("port", None),
            serial_baud_rate=int(ser.get("baud_rate", 115200)),
            data_dir=_str(pth, "data_dir", "paths"),
            log_dir=_str(pth, "log_dir", "paths"),
        )

        cfg._validate_paths()
        logger.info("SystemConfig loaded successfully (mode=%s).", cfg.camera_mode)
        return cfg

    # ------------------------------------------------------------------
    # Bridge to camera.py
    # ------------------------------------------------------------------

    def to_camera_config(self, session: "SessionConfig") -> CameraConfig:
        """
        Construct a CameraConfig from this SystemConfig and a SessionConfig.

        This is the single bridge between the two configuration layers.
        acquisition.py calls this once at session start; no other module
        needs to construct a CameraConfig directly.

        Parameters
        ----------
        session : SessionConfig
            Per-session parameters (frame rate, exposure, session ID).

        Returns
        -------
        CameraConfig ready to pass to XimeaCamera.
        """
        mode_str = self.camera_mode

        if mode_str == "SINGLE_ROI":
            return single_roi_config(
                width=self.single_roi.width,
                height=self.single_roi.height,
                offset_x=self.single_roi.offset_x,
                offset_y=self.single_roi.offset_y,
                exposure_us=session.exposure_us,
                frame_rate_hz=session.frame_rate_hz,
                label=self.single_roi.label,
                serial_number=self.camera_serial_number,
                transport_buffer_size=self.camera_transport_buffer_size,
            )

        elif mode_str == "MULTI_ROI":
            return multi_roi_config(
                rois=self.multi_roi.to_roi_list(),
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

    @property
    def centre_roi_offset_x(self) -> int:
        """
        X-offset of the centre ROI on the sensor, for use in
        compute_pointing_error(). Derived from whichever ROI config is active.
        """
        if self.camera_mode == "MULTI_ROI" and self.multi_roi is not None:
            # The centre ROI offset_x is the core_x computed inside make_rois()
            m = self.multi_roi
            return m.offset_x + (m.full_width - m.core_width) // 2
        if self.camera_mode == "SINGLE_ROI" and self.single_roi is not None:
            return self.single_roi.offset_x
        return 0

    @property
    def centre_roi_offset_y(self) -> int:
        """
        Y-offset of the centre ROI on the sensor, for use in
        compute_pointing_error(). Derived from whichever ROI config is active.
        """
        if self.camera_mode == "MULTI_ROI" and self.multi_roi is not None:
            m = self.multi_roi
            return m.offset_y + (m.full_height - m.core_height) // 2
        if self.camera_mode == "SINGLE_ROI" and self.single_roi is not None:
            return self.single_roi.offset_y
        return 0

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
    def write_example(path: str, mode: str = "SINGLE_ROI") -> None:
        """
        Write an annotated example JSON config file.

        Parameters
        ----------
        path : str
            Output file path.
        mode : str
            "SINGLE_ROI", "MULTI_ROI", or "FULL_FRAME". Controls which
            roi section is written.
        """
        single_roi_section = {
            "_comment": "Required for SINGLE_ROI mode",
            "width": 128,
            "height": 128,
            "offset_x": 2192,
            "offset_y": 2192,
            "label": "single_roi",
        }
        multi_roi_section = {
            "_comment": "Required for MULTI_ROI mode",
            "sensor_width": 4512,
            "sensor_height": 4512,
            "offset_x": 0,
            "offset_y": 0,
            "sidelobe_width": 360,
            "sidelobe_height": 360,
            "core_width": 128,
            "core_height": 128,
            "full_width": 4512,
            "full_height": 4512,
        }

        example: dict = {
            "camera": {
                "_comment": "serial_number and mode are required",
                "serial_number": "XXXXXXXX",
                "mode": mode.upper(),
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
                "data_dir": "/data/startracker",
                "log_dir": "/var/log/startracker",
            },
        }

        if mode.upper() == "SINGLE_ROI":
            example["roi"] = single_roi_section
        elif mode.upper() == "MULTI_ROI":
            example["roi"] = multi_roi_section

        with open(path, "w") as f:
            json.dump(example, f, indent=2)
        logger.info("Example config written to: %s", path)


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    """
    Per-session acquisition parameters, supplied programmatically.

    These are not stored in the JSON file — they are commanded at the start
    of each imaging session.

    Attributes
    ----------
    frame_rate_hz : float
        Target acquisition frame rate in Hz. Must be positive.
    exposure_us : int
        Exposure time in microseconds. Must be less than
        1_000_000 / frame_rate_hz.
    n_frames : int or None
        Number of frames to acquire before stopping. Pass None to run
        until explicitly stopped.
    session_id : str
        Short identifier used as a prefix for output filenames,
        e.g. "obs_2026_177_001". Must not be empty.
    """
    frame_rate_hz: float
    exposure_us: int
    n_frames: Optional[int]
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
                f"frame period ({frame_period_us:.0f} us at {self.frame_rate_hz} Hz)."
            )
        if self.n_frames is not None and self.n_frames <= 0:
            raise ConfigError(
                f"n_frames must be positive or None, got {self.n_frames}"
            )
        if not self.session_id:
            raise ConfigError("session_id must not be empty.")


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
            f"Config field '{section}.{key}' must be an integer, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _str(d: dict, key: str, section: str) -> str:
    val = d[key]
    if not isinstance(val, str):
        raise ConfigError(
            f"Config field '{section}.{key}' must be a string, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _validate_choice(value: str, choices: List[str], field: str) -> None:
    if value not in choices:
        raise ConfigError(
            f"Config field '{field}' must be one of {choices}, got {value!r}."
        )


def _validate_range(value: float, lo: float, hi: float, field: str) -> None:
    if not (lo <= value <= hi):
        raise ConfigError(
            f"Config field '{field}' must be in [{lo}, {hi}], got {value}."
        )


def _warn_extra_keys(d: dict, known: List[str], section: str) -> None:
    for key in d:
        if key not in known and not key.startswith("_"):
            logger.warning(
                "Unknown key '%s.%s' in config — ignored.", section, key
            )
