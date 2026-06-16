from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict

import numpy as np

from ximea import xiapi

logger = logging.getLogger(__name__)

#-----------------------------
# Operating Mode
#-----------------------------

class CameraMode(Enum):
    """
    Selects the camera operating mode

    MULTI_ROI - 3x3 multi-region grid; frame rate 10-100 Hz
    SINGLE_ROI - Single Region of Interest
    FULL_FRAME - Entire sensor as a single region; diagnostic use only;
                 no frame rate lower bound enforced
    """
    MULTI_ROI = auto()
    SINGLE_ROI = auto()
    FULL_FRAME = auto()

#----------------------------
# Data Structures
#----------------------------

@dataclass(frozen=True)
class RoiDefinition:
    """
    Geometry for one region

    Attributes
    ----------
    height : int
        Height of this region in pixels. Must be divisible by the camera's height increment.
        (query with get_height_increment())
    offset_y : int
        Y-offset of this region form the top of the sensor in pixels
    width : int
        Width of this region in pixels. Must be divisible by the camera's width increment.
    offset_x : int
        X-offset from the left of the sensor in pixels
    label : str
        Human-readable identifier, e.g. "top_left", "centre"
    is_data : bool
        True for regions to be saved; False for interstitial regions that are acquired but
        immediately discarded
    """
    height: int
    offset_y: int
    width: int
    offset_x: int
    label: str
    is_data: bool = True

@dataclass(frozen=True)
class CameraConfig:
    """
    Complete camera configuration. Validated on construction

    Attributes
    ----------
    serial_number : str or None
        Camera serial number for open_device_by_SN(). Pass None to open the first
        available device
    mode: CameraMode
        MULTI_ROI for normal science acquisition; FULL_FRAME for diagnostics
    exposure_us : int
        Exposure time in microseconds
    frame_rate_hz : float
        Target frame rate in Hz.
    rois : List[RoiDefinition] or None
        Required for MULTI_ROI or SINGLE_ROI, must be None for FULL_FRAME
    output_bit_depth : int
        Sensor output bit depth. 12 for this application
    transport_buffer_size : int
        Number of frames to allocate in the Ximea transport buffer.
        Larger values reduce the risk of dropping frames at high rates
        at the cost of latency and memory. Default of 8 as a starting point.
        The value is clamped at runtime to the camera's own reported [min,max] range,
        so the actual size of the buffer may differ from what is requested. A warning
        is logged if clamping occurs.
    """
    serial_number: Optional[str]
    mode: CameraMode
    exposure_us: int
    frame_rate_hz: float
    rois: Optional[List[RoiDefinition]] = None
    output_bit_depth: int = 12
    transport_buffer_size: int = 8

    def __post_init__(self) -> None:
        if self.output_bit_depth != 12:
            raise ValueError(
                f"output_bit_depth must be 12, got {self.output_bit_depth}"
            )

        if self.frame_rate_hz <= 0.0:
            raise ValueError(
                f"frame_rate_hz must be positive, got {self.frame_rate_hz}"
            )
        
        if self.mode is CameraMode.MULTI_ROI:
            self._validate_multi_roi()
        elif self.mode is CameraMode.SINGLE_ROI:
            self._validate_single_roi()
        elif self.mode is CameraMode.FULL_FRAME:
            self._validate_full_frame()
        else:
            raise ValueError(
                f"Camera mode not recognised, got {self.mode}"
            )
        
    def _validate_multi_roi(self) -> None:
        if self.rois is None:
            raise ValueError("rois must be provided for MULTI_ROI mode")
        if len(self.rois) != 9:
            raise ValueError(
                f"Exactly 9 ROI definitions required for MULTI_ROI mode, got {len(self.rois)}"
            )
        
        # Validate data discard pattern
        expected_data = {0, 2, 4, 6, 8}
        expected_discard = {1, 3, 5, 7}
        for idx, roi in enumerate(self.rois):
            if idx in expected_data and not roi.is_data:
                raise ValueError(
                    f"ROI at index {idx} must be a data region (is_data=True)"
                )
            if idx in expected_discard and roi.is_data:
                raise ValueError(
                    f"ROI at index {idx} must be a discard region (is_data=False)"
                )
        
        # TODO: Validate ROIS for width/height consistency


    def _validate_single_roi(self) -> None:
        if self.rois is None:
            raise ValueError("roi must be provided for SINGLE_ROI mode")
        if len(self.rois) != 1:
            raise ValueError(
                f"Exactly 1 ROI definitions required for SINGLE_ROI mode, got {len(self.rois)}"
            )
        
        # Validate data discard pattern
        if not self.rois[0].is_data:
            raise ValueError(
                f"ROI must be a data region (is_data=True)"
            )

    def _validate_full_frame(self) -> None:
        if self.rois is not None:
            raise ValueError(
                "rois must be None for FULL_FRAME mode; image geometry is determined"
                "automatically from the sensor dimensions"
            )

@dataclass
class AcquiredFrame:
    """
    Output of a single camera acquisition cycle

    Attributes
    ----------
    frame_id : int
        Monotonically increasing frame counter from the camera
    timestamp_ns : int
        Camera hardware timestamp in nanoseconds (from img.tsSec / tsUSec)
    host_time : float
        time.monotonic() recorded immediately after get_image() returns,
        for latency diagnostics
    mode : CameraMode
        The operationg mode active when this frame was acquired
    rois : dict[str, np.ndarray]
        MULTI_ROI: mapping of label -> 2-D uint16 array for the five data-bearing 
        regions. Interstitual regions are absent
        SINGLE_ROI: single entry keyed "single_roi" containing the single region image
        as a 2-D uint array
        FULL_FRAME: single entry keyed "full_frame" containing the complete sensor image
        as a 2-D uint array (sensor_height x sensor_width)
    nframes_dropped : int
        Cummulative dropped frame count reported by the camera at this frame
    """
    frame_id: int
    timestamp_ns: int
    host_time: float
    mode: CameraMode
    rois: Dict[str, np.ndarray]
    nframes_dropped: int

#---------------------------------
# ROI strip parsing
#---------------------------------

def _parse_roi_strip(
        strip: np.ndarray,
        rois: List[RoiDefinition],
) -> Dict[str, np.ndarray]:
    """
    TODO: Write this code for multi-ROI, discarding unwanted regions

    Slice the concatenated multi-ROI strip into individual region arrays

    The Ximea drive returns all active regions stacked vertically in a single
    image buffer. Region heights accumulate from top to bottom in region index order.
    Only data-bearing regions are included in the returned dict.

    Parameters
    ----------
    strip : np.ndarray
        2-D uint16 array of shape (total_height, width) as returned by
        img.get_image_data_numpy().
    rois : List[RoiDefinition]
        The 9 RoiDefinition objects in index order

    Returns
    -------
    dict mapping label -> 2-D uint16 subarray (a view, not a copy)
    """

    pass

#--------------------------------------
# Camera class
#--------------------------------------

class XimeaCamera:
    """
    Lifecycle manager and acquisition interface for a single Ximea camera

    Supports three operating modes selected via CameraConfig.mode:

        MULTI_ROI - 3x3 multi-region grid for high-rate science acquisition
        SINGLE_ROI - single region for testing
        FULL_FRAME - single full-sensor region for diagnostic image capture

    Usage
    -----
    Intended as a context manager:

        config = CameraConfig(mode = CameraMode.MULTI_ROI, ...)
        with XimeaCamera(config) as cam:
            for _ in range(n_frames):
                frame = cam.acquire_frame()

        # Diagnostic session, seperate config object, same camera obejct:
        diag_config = CameraConfig(mode = CameraMode.FULL_FRAME, ...)
        with XimeaCamera(diag_config) as cam:
            frame = cam.acquire_frame()

    The camera is opened, configured, and started in __enter__; stopped and
    closed in __exit__. Calling acquired_frame() ourside the context manager
    raises RuntimeError.

    Thread safety
    -------------
    Not thread-safe. Intended to be owned by a signle acquisition thread or process.
    The returned AcquiredFrame contains numpy arrays that are copies of the
    camera buffer and are safe to hand off to another thread/process
            
    """

    MAX_PIXEL_VALUE: int = 4512

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cam: Optional[xiapi.Camera] = None
        self._img: Optional[xiapi.Image] = None
        self._frame_counter: int = 0
        self._acquiring: bool = False
        self._last_acq_nframe: int = -1 # for dropped-frame detection

    def __enter__(self) -> XimeaCamera:
        self.open()
        self.configure()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            self.stop()
        finally:
            self.close()
        return False # do not suppress exceptions
    
    #-----------------------------------------------
    # Lifecycle methods (can be called manually)
    #-----------------------------------------------

    def open(self) -> None:
        """Open the camera device."""
        if self._cam is not None:
            raise RuntimeError("Camera is already open.")
        self._cam = xiapi.Camera()
        if self._config.serial_number is not None:
            logger.info("Opening camera by SN: %s", self._config.serial_number)
            self._cam.open_device_by_SN(self._config.serial_number)
        else:
            logger.info("Opening first available camera.")
            self._cam.open_device()
        self._img = xiapi.Image()
        model = self._cam.get_device_name()
        sn = self._cam.get_device_sn()
        logger.info("Opened camera: model=%s SN=%s", model, sn)

    def configure(self) -> None:
        """
        Apply CameraConfig to the open camera

        Branches on the CameraConfig.mode:
            MULTI_ROI -> calls _configure_rois() for the 3x3 region grid
            SINGLE_ROI -> TODO: might need a separate configuration function
            FULL_FRAME -> calls_configure_full_frame() for the single-region readout

        Order of operations:
            1. Set image data format
            2. Set output bit depth
            3. Configure ROI geometry
            4. Set timing mode and frame rate
            5. Set exposure.
            6. Set transport buffer size
        """

        if self._cam is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        if self._acquiring:
            raise RuntimeError("Cannot reconfigure during acquisition. Call stop() first.")
        
        cfg = self._config
        cam = self._cam

        logger.info("Configuring camera (mode=%s)...", cfg.mode.name)

        # 1. Image data format
        cam.set_imgdataformat("XI_MONO16")
        logger.debug("imgdataformat = XI_MONO16")

        # 2. Output bit depth
        cam.set_output_bit_depth("XI_BPP_12")
        logger.debug("output_bit_depth = XI_BPP_12")

        # 3. ROI geometry: mode-dependent
        if cfg.mode is CameraMode.MULTI_ROI:
            self._configure_rois()
        if cfg.mode is CameraMode.SINGLE_ROI:
            self._configure_single_roi()
        elif cfg.mode is CameraMode.FULL_FRAME:
            self._configure_full_frame()

        # 4. Frame rate
        timing_mode = _select_timing_mode(cam.get_device_name())
        cam.set_acq_timing_mode(timing_mode)
        cam.set_framerate(cfg.frame_rate_hz)
        actual_fps = cam.get_framerate()
        logger.info(
            "Frame rate: timing_mode=%s requested=%.2f Hz reported=%.2f Hz",
            timing_mode, cfg.frame_rate_hz, actual_fps
        )
        if timing_mode == "XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT":
            logger.info(
                "Note: FRAME_RATE_LIMIT is a rate ceiling not a locked clock."
                "Use frame timestamps to measure the actual acquisition rate."
            )

        # 5. Exposure
        frame_period_us = 1_000_000.0 / cfg.frame_rate_hz
        if cfg.exposure_us >= frame_period_us:
            raise ValueError(
                f"exposure_us ({cfg.exposure_us} us) must be less than"
                f"frame period ({frame_period_us:.0f} us) at "
                f"{cfg.frame_rate_hz} Hz."
            )
        cam.set_exposure(cfg.exposure_us)
        logger.info("Exposure %d us", cfg.exposure_us)

        # 6. Transport buffer
        buf_min = cam.get_buffers_queue_size_minimum()
        buf_max = cam.get_buffers_queue_size_maximum()
        buf_size = max(buf_min, min(cfg.transport_buffer_size, buf_max))
        if buf_size != cfg.transport_buffer_size:
            logger.warning(
                "Requested transport_buffer_size=%d is outside the camera's "
                "valid range [%d, %d]; clamped to %d.",
                cfg.transport_buffer_size, buf_min, buf_max, buf_size,
            )
        cam.set_buffers_queue_size(buf_size)
        logger.debug(
            "Transport buffer size = %d frames (range [%d, %d])", 
            buf_size, buf_min, buf_max
        )

        logger.info("Camera configuration complete")

    def start(self) -> None:
        """
        Begin image acquisition
        """
        if self._cam is None:
            raise RuntimeError("Camera is not open.")
        if self._acquiring:
            logger.warning("start() called but acquisition already running")
            return
        self._cam.start_acquisition()
        self._acquiring = True
        self._frame_counter = 0
        self._last_acq_nframe = -1
        logger.info("Acquisition started")

    def stop(self) -> None:
        """
        Stop image acquisition
        """
        if self._cam is None or not self._acquiring:
            return
        self._cam.stop_acquisition()
        self._acquiring = False
        logger.info(
            "Acquisition stopped. Total frames acquired %d",
            self._frame_counter
        )
    
    def close(self) -> None:
        """
        Close the camera device and release resources
        """
        if self._cam is None:
            return
        try:
            self._cam.close_device()
            logger.info("Camera closed.")
        except Exception:
            logger.exception("Exception while closing camera.")
        finally:
            self._cam = None
            self._img = None


    #-------------------
    # Image acquisition
    #-------------------

    def acquire_frame(self, timeout_ms: int = 1000) -> AcquiredFrame:
        """
        Block until the next frame is available and return it as an AcquiredFrame

        Parameters
        ----------
        timeout_ms : int
            Maximum time to wait for the next frame in milliseconds.
            At 100 Hz the nominal inter-frame intervale is 10 ms; 1000 ms is a generous
            default to catch genuine hardware faults.

        Returns
        -------
        AcquiredFrame
            Contains copies of the five date-bearing ROI arrays (uint16) and frame metadata.
            Arrays are independent of the camera's internal buffer and safe to pass to other
            threads.

        Raises
        ------
        RuntimeError
            If called outside an active acquisition session.
        xiapi.Xi_error
            On camera hardware or transport errors.
        """

        if not self.is_acquiring:
            raise RuntimeError(
                "acquire_frame() called outside of an active acquisition session."
            )
        
        self._cam.get_image(self._img, timeout=timeout_ms)
        host_time = time.monotonic()

        # Reconstruct hardware timestamp in nanoseconds from seconds + microseconds
        timestamp_ns = (
            int(self._img.tsSec) * 1_000_000_000
            + int(self._img.tsUSec) * 1_000
        )

        # Get the full multi-ROI strip as a numpy array
        strip: np.ndarray = self._img.get_image_data_numpy()

        # Defensive copy: the underlying buffer may be reused by the next get_image() call
        strip = strip.copy()

        # Slice into individual ROIs; discard interstitual regions
        if self._config.mode is CameraMode.MULTI_ROI:
            roi_arrays = _parse_roi_strip(strip, self._config.rois)
        elif self.config.mode is CameraMode.SINGLE_ROI:
            roi_arrays = {"single_roi": strip}
        else:
            roi_arrays = {"full_frame": strip}

        # acq_nframe is a monotonically increasing acquisition counter that
        # does not reset on parameter changes. Gaps between consecutive values
        # indicate dropped frames
        acq_nframe = int(self._img.acq_nframe)
        if self._last_acq_nframe == -1:
            nframes_dropped = 0
        else:
            nframes_dropped = max(0, acq_nframe - self._last_acq_nframe -1)
        if nframes_dropped > 0:
            logger.warning(
                "Dropped %d frame(s) detected between acq_nframe %d and %d.",
                nframes_dropped, self._last_acq_nframe, acq_nframe
            )
        self._last_acq_nframe = acq_nframe
        self._frame_counter += 1

        return AcquiredFrame(
            frame_id=acq_nframe,
            timestamp_ns=timestamp_ns,
            host_time=host_time,
            mode=self._config.mode,
            rois=roi_arrays,
            nframes_dropped=nframes_dropped,
        )

    #------------------
    # Properties
    #------------------

    @property
    def is_acquiring(self) -> bool:
        return self._acquiring
    
    @property
    def frames_acquired(self) -> int:
        return self._frame_counter
    
    @property
    def config(self) -> CameraConfig:
        return self._config
    
    @property
    def mode(self) -> CameraMode:
        return self._config.mode


    #-------------------
    # Private helpers
    #-------------------

    def _configure_full_frame(self) -> None:
        """
        Configure the camera for full-sensor single-region readout.

        Ensures only region 0 is active and sets it to the maximum sensor dimensions
        Then resets offset to (0,0). Any previously configured regions are deactviated

        The resulting AcquiredFrame will contain a single "full_frame" entry in its
        rois dict with shape (sensor_height, sensor_width)
        """
        cam = self._cam

        max_width = cam.get_width_maximum()
        max_height = cam.get_height_maximum()

        cam.set_offsetX(0)
        cam.set_offsetY(0)
        cam.set_width(max_width)
        cam.set_height(max_height)

        logger.info(
            "Full-frame configured: %d x %d pixels (width x height)",
            max_width, max_height,
        )
        
    def _configure_single_roi(self) -> None:
        """
        Configure a single ROI on the camera.
        """
        cam = self._cam
        roi = self._config.rois[0]

        # Validate camera alignment constraints before touching hardware
        # These incremental values are camera-model dependent

        _check_alignment("width", roi.width, cam.get_width_increment())
        _check_alignment("height", roi.height, cam.get_height_increment())
        _check_alignment("offset_x", roi.offset_x, cam.get_offsetX_increment())
        _check_alignment("offset_y", roi.offset_y, cam.get_offsetY_increment())

        cam.set_width(roi.width)
        cam.set_height(roi.height)
        cam.set_offsetX(roi.offset_x)
        cam.set_offsetY(roi.offset_y)

        logger.debug(
            "ROI %s configured: width=%d offset_x=%d height=%d offset_y=%d",
            roi.label, roi.width, roi.offset_x, roi.height, roi.offset_y, 
        )


    def _configure_rois(self) -> None:
        """
        TODO: Configure the 3x3 multi-ROI grid on the camera
        """
        pass


def _check_alignment(name: str, value: int, increment: int) -> None:
    """
    Raise ValueError if value is not divisible by increment
    """
    if increment > 0 and value % increment != 0:
        raise ValueError(
            f"{name}={value} is not divisible by the camera's required "
            f"increment ({increment}). Adjust to a multiple of {increment}"
        )

# Camera families that support XI_ACQ_TIMING_MODE_FRAME_RATE (exact FPGA clock).
#All other families use XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT (rate ceiling).
_EXACT_TIMING_PREFIXES = ("MQ", "MD")

def _select_timing_mode(device_name: bytes) -> str:
    """
    Return the correct acq_timing_mode string for the connected camera.

    XI_ACQ_TIMING_MODE_FRAME_RATE       -- exact period FPGA clock.
                                           Supported by MQ and MD families only.
    XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT -- frame rate ceiling.
                                           Supported by CB, MC, MT, MX families

    Parameters
    ----------
    device_name : bytes
        Value returned by cam.get_device_name(), e.g. b"MC203MG-SY-UB"
    """
    name_str = device_name.decode("ascii", errors="ignore").upper()
    for prefix in _EXACT_TIMING_PREFIXES:
        if name_str.startswith(prefix):
            return "XI_ACQ_TIMING_MODE_FRAME_RATE"
    return "XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT"


#-----------------------------------
# Config factories
#-----------------------------------

def single_roi_config(
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
    exposure_us: int,
    frame_rate_hz: float,
    label: str = "single_roi",
    serial_number: Optional[str] = None,
    transport_buffer_size: int = 8,
) -> CameraConfig:
    """
    Convenience function for creating a SINGLE_ROI config

    Example
    -------
    >>> cfg = single_roi_config(
    ...      width=512, height=512,
    ...      offset_x=1992, offset_y=1992,
    ...      exposure_us=5_000, frame_rate_hz=10.0,
    ... )
    >>> with XimeraCamera(cfg) as cam:
    ...    frame = cam.acquire_frame()
    ...    image = frame.rois["single_roi"]   # unit16, 512 x 512
    """
    return CameraConfig(
        serial_number = serial_number,
        mode = CameraMode.SINGLE_ROI,
        exposure_us = exposure_us,
        frame_rate_hz = frame_rate_hz,
        rois = [RoiDefinition(
            height = height,
            offset_y = offset_y,
            width = width,
            offset_x = offset_x,
            label = label,
            is_data = True,
        )],
        transport_buffer_size = transport_buffer_size
    )


def full_frame_config(
    exposure_us: int,
    frame_rate_hz: float,
    serial_number: Optional[str] = None,
    transport_buffer_size: int = 4,
) -> CameraConfig:
    """
    Convenience function for creating a FULL_FRAME config

    Example
    -------
    >>> cfg = full_frame_config(exposure_us=50_000, frame_rate_hz=2.0)
    >>> with XimeraCamera(cfg) as cam:
    ...    frame = cam.acquire_frame()
    ...    image = frame.rois["full_frame"]   # unit16, 4512 x 4512
    """
    return CameraConfig(
        serial_number = serial_number,
        mode = CameraMode.FULL_FRAME,
        exposure_us = exposure_us,
        frame_rate_hz = frame_rate_hz,
        rois = None,
        transport_buffer_size = transport_buffer_size,
    )

def multi_roi_config(
    rois: List[RoiDefinition],
    exposure_us: int,
    frame_rate_hz: float,
    serial_number: Optional[str] = None,
    transport_buffer_size: int = 8,
) -> CameraConfig:
    """
    Convenience function for creating a MULTI_ROI configuration.

    The rois list is most conveniently built with the make_rois() helper function,
    which creates the required 9-element list with the correct pattern (data regions
    at indices 0, 2, 4, 6, 9; discard regions at 1, 3, 5, 7)

    Parameters
    ----------
    rois : List[RoiDefinition]
        Exactly 9 RoiDefinition objects in region-index order.
    exposure_us : int
        Exposure time in microseconds. Must be less than 1_000_000 / frame_rate_hz
    frame_rate_hz : float
        Target frame rate in Hz.
    serial_number : str or None
        Camera serial number, or None to open the first available device.
    transport_buffer_size : int
        Requested Ximea transport buffer depth. Default 8; clamped at runtime to
        camera's supported range.
    """
    return CameraConfig(
        serial_number = serial_number,
        mode = CameraMode.MULTI_ROI,
        exposure_us = exposure_us,
        frame_rate_hz = frame_rate_hz,
        rois = rois,
        transport_buffer_size = transport_buffer_size,
    )

def make_rois(
    sensor_width: int = 4512,
    sensor_height: int = 4512,
    offset_x: int = 0,
    offset_y: int = 0,
    sidelobe_width: int = 512,
    sidelobe_height: int = 512,
    core_width: int = 128,
    core_height: int = 128,
    full_width: int = 4512,
    full_height: int = 4512,
) -> List[RoiDefinition]:
    """
    Build a 9-element RoiDefinition list with a "quincunx" (⁙) arrangement.

    The 3x3 grid is:
        [0] top_left            [1] top_mid (discard)   [2] top_right
        [3] mid_left (discard)  [4] centre              [5] mid_right (discard)
        [6] bot_left            [7] bot_mid (discard)   [8] bot_right

    Parameters
    ----------
    sensor_width, sensor_height : int
        Full sensor dimensions in pixels
    offset_x, offset_y : int
        Offset in pixels to the top-leftmost pixel of the top_left region
    sidelobe_width : int
        The width of the left and right columns in pixels
    sidelobe_height : int
        The height of the top and bottom rows in pixels
    core_width : int
        The width of the central column in pixels
    core_height : int
        The height of the middle row in pixels
    full_width : int
        The full width of the pattern in pixels
    full_height : int
        The full height of the pattern in pixels
    
    """

    if offset_x + full_width > sensor_width:
        raise ValueError(
            f"Pattern is expected to be exceed width of sensor ({offset_x + full_width} > {sensor_width})"
        )

    if offset_y + full_height > sensor_height:
        raise ValueError(
            f"Pattern is expected to be exceed height of sensor ({offset_y + full_height} > {sensor_height})"
        )

    core_x = offset_x + (full_width - core_width) // 2
    right_x = offset_x + full_width - sidelobe_width
    core_y = offset_y + (full_height - core_height) // 2
    bottom_y = offset_y + full_height - sidelobe_height

    return [
        RoiDefinition(sidelobe_height, offset_y, sidelobe_width, offset_x, "top_left", True),
        RoiDefinition(sidelobe_height, offset_y, core_width, core_x, "top_mid", False),
        RoiDefinition(sidelobe_height, offset_y, sidelobe_width, right_x, "top_right", True),
        RoiDefinition(core_height, core_y, sidelobe_width, offset_x, "mid_left", False),
        RoiDefinition(core_height, core_y, core_width, core_x, "centre", True),
        RoiDefinition(core_height, core_y, sidelobe_width, right_x, "mid_right", False),
        RoiDefinition(sidelobe_height, bottom_y, sidelobe_width, offset_x, "bot_left", True),
        RoiDefinition(sidelobe_height, bottom_y, core_width, core_x, "bot_mid", False),
        RoiDefinition(sidelobe_height, bottom_y, sidelobe_width, right_x, "bot_right", True),
    ]
