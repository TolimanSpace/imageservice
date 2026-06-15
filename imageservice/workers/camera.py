import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

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
    rois : list[RoiDefinition] or None
        Required for MULTI_ROI or SINGLE_ROI, must be None for FULL_FRAME
    output_bit_depth : int
        Sensor output bit depth. 12 for this application
    transport_buffer_size : int
        Number of frames to allocate in the Ximea transport buffer.
        Larger values reduce the risk of dropping frames at high rates
        at the cost of latency and memory. Default of 8 as a starting point.
    """
    serial_number: Optional[str]
    mode: CameraMode
    exposure_us: int
    frame_rate_hz: float
    rois: Optional[list[RoiDefinition]] = None
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
    rois: dict[str, np.ndarray]
    nframes_dropped: int

#---------------------------------
# ROI strip parsing
#---------------------------------

def _parse_roi_strip(
        strip: np.ndarray,
        rois: list[RoiDefinition],
) -> dict[str, np.ndarray]:
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
    rois : list[RoiDefinition]
        The 9 RoiDefinition objects in index order

    Returns
    -------
    dict mapping label -> 2-D uint16 subarray (a view not a copy)
    """

    pass


class BaseCameraInterface:
    """
    A base interface for a PySpin or Simulated Camera

    This class provides a template for camera interfaces, including methods for starting, stopping
    applying settings, and capturing frames.
    """
    def __init__(self):
        pass

    def __enter__(self):
        return self
 
    def start(self):
        """
        Start the camera interface
        """
        pass

    def apply_settings(self, settings):
        """
        Apply settings to the camera.

        Args:
            settings (dict): A dictionary of settings to apply.
        """
        pass

    def apply_pattern(self):
        """
        Set camera to use test pattern
        """
        pass

    def capture_frame(self):
        """
        Capture a frame from the camera.

        Raises:
            NotImplementedError: This method should be implemented by subclassses.
        """
        raise NotImplementedError
    
    def stop(self):
        """
        Stop the camera interface
        """
        pass

    def __exit__(self, type, value, traceback):
        pass

class SimulatedCameraInterface(BaseCameraInterface):
    """
    A simulated camera interface for testing purposes.

    This class simulates a camera by loading images from a specified directory
    and returning them as if they were captured by a real camera
    """
    def __init__(self, image_dir = "images/simulated"):
        """
        Initialize the simulated camera interface.

        Args:
            image_dir (str): The directory containing simulated images.
        """
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith('.npy')
        ])

        self.index = 0
        self.exposure = 'Simulated Image'
        self.last_frame_time = time.time()
        self.frame_interval = 0.1

    def capture_frame(self):
        """
        Capture a frame from the simulated camera.

        This method loads an image from the specified directory and returns it along with metadata.

        Returns:
            tuple: A tuple containing metadata (dict) and image data (numpy array) .
        """
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)

        self.last_frame_time = time.time()
        comptime = str(datetime.now())

        if self.index >= len(self.image_paths):
            # Loop to first image
            self.index = 0
        image_data = np.load(self.image_paths[self.index])
        self.index += 1

        # Create metadata
        i = self.index
        camtime = int(self.last_frame_time*1e6)
        pxlfmt = image_data.dtype.name
        xoff = 0
        xpad = 0
        yoff = 0
        ypad = 0

        filename = f'images/raw/frame_{camtime}.npy'

        # Package data in a dict
        metadata = {
            "i": i,
            "rawfile": filename,
            "camtime": camtime,
            "comptime": comptime,
            "pxlfmt": pxlfmt,
            "xoff": xoff,
            "xpad": xpad,
            "yoff": yoff,
            "ypad": ypad,
            "exposure": self.exposure
            }
        

        return metadata, image_data



class CameraInterface(BaseCameraInterface):
    """
    A class to interface with a PySpin Camera

    Attributes
    ---------
    _lock : threading.Lock
        A lock to ensure thread saftey.
    cam : PySpin.Camera
        The camera object
    running : bool
        Indicated if the camera is running.
    processor : PySpin.ImageProcessor
        The image processor for the camera

    Methods
    -------
    __init__():
        Initializes the camera interface.
    exposure:
        Gets or sets the camera's exposure time.
    framerate:
        Gets or sets the camera's frame rate.
    resolution:
        Gets or sets the camera's resolution.
    acquisitionmode:
        Gets or sets the camera's acquisition mode.
    pixelformat:
        Gets or sets the camera's pixel format.
    buffercount:
        Gets or sets the camera's buffer size.
    bufferhandlingmode:
        Gets or sets how the camera handles images in the buffer.
    offsetx:
        Gets or sets the X offset of the camera's images.
    offsety:
        Gets or sets the Y offset of the camera's images.
    width:
        Gets or sets the width of the camera's images.
    height:
        Gets or sets the height of the camera's images.
    init():
        Initializes the camera
    __enter__():
        Initializes the camera and returns the instance
    close():
        Closes the camera.
    __exit__(type, value, traceback):
        Closes the camera and releases the PySpin system instance.
    start():
        Starts recording images.
    stop():
        Stops recording images.
    apply_setting(settings):
        Applies settings to the camera.
    apply_pattern():
        Applies a test pattern to the camera.
    capture_frame():
        Captures a single image frame.
    capture_frames(n_frames):
        Captures multiple iamge frames.
    """

    def __init__(self):
        """
        Initializes the CameraInterface.

        This method initializes the camera interface, sets up the camera, and configures the image processor.
        """
        self._lock = threading.Lock()

        cam_list = list_cameras()
        if not cam_list.GetSize():
            _logger.error("No cameras detected")
            raise RuntimeError("No cameras detected")
        self.cam = cam_list.GetByIndex(0)
        cam_list.Clear()
        self.running = False

        self.processor = PySpin.ImageProcessor()
        # Might need to move this next line to a property that can be varied
        self.processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_NONE)
        

    @property
    def exposure(self):
        """
        Gets the camera's exposure time.

        Returns:
            float: The current exposure time.
        """
        return self.cam.ExposureTime.GetValue()

    @exposure.setter
    def exposure(self, value):
        """
        Sets the camera's exposure time.

        Args:
            value (float): The desired exposure time.
        """
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(value)

    @property
    def framerate(self):
        """
        Gets the camera's frame rate.

        Returns:
            float: The current frame rate.
        """
        return self.cam.AcquisitionFrameRate.GetValue()
        
    @framerate.setter
    def framerate(self, value):
        """
        Sets the camera's frame rate.

        Args:
            value (float): The desired frame rate.
        """
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(value)

    @property
    def resolution(self):
        """
        Gets the camera's resolution.

        Returns:
            tuple: A tuple containing the height (int) and width (int) of images
        """
        width = self.cam.Width.GetValue()
        height = self.cam.Height.GetValue()
        return (height, width)

    @resolution.setter
    def resolution(self, value):
        """
        Sets the camera's resolution.

        Args:
            value (tuple): A tuple containing the desired height (int) and width (int) of images
        """
        self.apply_settings({'height': value[0], 'width': value[1]})

    @property
    def acquisitionmode(self):
        """
        Gets the camera's acquisition mode.

        Returns:
            int: The current acquisition mode.
        """
        return self.cam.AcquisitionMode.GetValue()
        
    @acquisitionmode.setter
    def acquisitionmode(self, value):
        """
        Sets the camera's acquisition mode

        Args:
            value (int): The desired acquisition mode.
        """

        self.cam.AcquisitionMode.SetValue(value)
        
    @property
    def pixelformat(self):
        """
        Gets the camera's pixel format.

        Returns:
            int: The current pixel format.
        """
        return self.cam.PixelFormat.GetValue()

    @pixelformat.setter
    def pixelformat(self,value):
        """
        Sets the camera's pixel format.

        Args:
            value (int): The desired pixel format.
        """
        self.cam.PixelFormat.SetValue(value)

    @property
    def buffercount(self):
        """
        Gets the camera's buffer count.

        Returns:
            int: The current buffer count.
        """
        return self.cam.TLStream.StreamBufferCountManual.GetValue()
    
    @buffercount.setter
    def buffercount(self,value):
        """
        Sets the camera's buffer count.

        Args:
            value (int): The desired buffer count.
        """

        self.cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)
        self.cam.TLStream.StreamBufferCountManual.SetValue(value)

    @property
    def bufferhandlingmode(self):
        """
        Gets the camera's buffer handling mode.

        Returns:
            int: The current buffer handling mode.
        """
        return self.cam.TLStream.StreamBufferHandlingMode.GetValue()
    
    @bufferhandlingmode.setter
    def bufferhandlingmode(self,value):
        """
        Sets the camera's buffer handling mode.

        Args:
            value (int): The current buffer handling mode.
        """
        self.cam.TLStream.StreamBufferHandlingMode.SetValue(value)

    @property
    def offsetx(self):
        """
        Gets the camera's image X offset.

        Returns:
            int: The current X offset.
        """
        return self.cam.OffsetX.GetValue()
    
    @offsetx.setter
    def offsetx(self,value):
        """
        Sets the camera's image X offset.

        The image width should be set prior to adjusting the X offset. 

        Args:
            value (int): The desired X offset.
        """
        inc = self.cam.OffsetX.GetInc()
        minval = self.cam.OffsetX.GetMin()
        maxval = self.cam.OffsetX.GetMax()
        if value < minval:
            _logger.warning(f"Value = {value} must be equal or greater than Min = {minval}")
            value = minval
            _logger.warning(f"Setting OffsetX to Value = {value} instead")
        if value > maxval:
            _logger.warning(f"Value = {value} must be equal or smaller than Max = {maxval}")
            value = maxval
            _logger.warning(f"Setting OffsetX to Value = {value} instead")
        if (value-minval) % inc != 0:
            _logger.warning(f"The difference between Value = {value} and Min = {minval} must be dividable without rest by Inc = {inc}")
            value = min(value + ((value-minval) % inc), maxval)
            _logger.warning(f"Setting OffsetX to Value = {value} instead")
        self.cam.OffsetX.SetValue(value)

    @property
    def offsety(self):
        """
        Gets the camera's image Y offset.

        Returns:
            int: The current Y offset.
        """
        return self.cam.OffsetY.GetValue()
    
    @offsety.setter
    def offsety(self,value):
        """
        Sets the camera's image Y offset.

        The image height should be set prior to adjusting the Y offset.

        Args:
            value (int): The desired Y offset.
        """
        inc = self.cam.OffsetY.GetInc()
        minval = self.cam.OffsetY.GetMin()
        maxval = self.cam.OffsetY.GetMax()
        if value < minval:
            _logger.warning(f"Value = {value} must be equal or greater than Min = {minval}")
            value = minval
            _logger.warning(f"Setting OffsetY to Value = {value} instead")
        if value > maxval:
            _logger.warning(f"Value = {value} must be equal or smaller than Max = {maxval}")
            value = maxval
            _logger.warning(f"Setting OffsetY to Value = {value} instead")
        if (value-minval) % inc != 0:
            _logger.warning(f"The difference between Value = {value} and Min = {minval} must be dividable without rest by Inc = {inc}")
            value = min(value + ((value-minval) % inc), maxval)
            _logger.warning(f"Setting OffsetY to Value = {value} instead")
        self.cam.OffsetY.SetValue(value)

    @property
    def width(self):
        """
        Gets the camera's image width.

        Returns:
            int: The current width.
        """
        return self.cam.Width.GetValue()
    
    @width.setter
    def width(self,value):
        """
        Sets the camera's image width.

        Returns:
            int: The desired width.
        """
        inc = self.cam.Width.GetInc()
        minval = self.cam.Width.GetMin()
        maxval = self.cam.Width.GetMax()
        if value < minval:
            _logger.warning(f"Value = {value} must be equal or greater than Min = {minval}")
            value = minval
            _logger.warning(f"Setting Width to Value = {value} instead")
        if value > maxval:
            _logger.warning(f"Value = {value} must be equal or smaller than Max = {maxval}")
            value = maxval
            _logger.warning(f"Setting Width to Value = {value} instead")
        if (value-minval) % inc != 0:
            _logger.warning(f"The difference between Value = {value} and Min = {minval} must be dividable without rest by Inc = {inc}")
            value = min(value + ((value-minval) % inc), maxval)
            _logger.warning(f"Setting Width to Value = {value} instead")
        self.cam.Width.SetValue(value)
    
    @property
    def height(self):
        """
        Gets the camera's image height.

        Returns:
            int: The current height.
        """
        return self.cam.Height.GetValue()
    
    @height.setter
    def height(self,value):
        """
        Sets the camera's image height.

        Returns:
            int: The desired height.
        """
        inc = self.cam.Height.GetInc()
        minval = self.cam.Height.GetMin()
        maxval = self.cam.Height.GetMax()
        if value < minval:
            _logger.warning(f"Value = {value} must be equal or greater than Min = {minval}")
            value = minval
            _logger.warning(f"Setting Height to Value = {value} instead")
        if value > maxval:
            _logger.warning(f"Value = {value} must be equal or smaller than Max = {maxval}")
            value = maxval
            _logger.warning(f"Setting Height to Value = {value} instead")
        if (value-minval) % inc != 0:
            _logger.warning(f"The difference between Value = {value} and Min = {minval} must be dividable without rest by Inc = {inc}")
            value = min(value + ((value-minval) % inc), maxval)
            _logger.warning(f"Setting Height to Value = {value} instead")
        self.cam.Height.SetValue(value)


    # ... add more properties as required

    def init(self):
        """
        Initializes the camera
        """
        self.cam.Init()
        _logger.info("Camera initialized")
        self.initialized = True

    def __enter__(self):
        self.init()
        return self
    
    def close(self):
        """
        Closes the camera
        """
        _logger.info("Stoping camera")
        self.stop()
        self.cam.DeInit()
        del self.cam
        _logger.info("Camera stopped")

    def __exit__(self, type, value, traceback):

        global _SYSTEM

        self.close()
        if _SYSTEM is not None:
            _SYSTEM.ReleaseInstance()
            _SYSTEM = None
            _logger.info("Releasing PySpin system instance")

    def start(self):
        """
        Start recording images
        """
        if not self.running:
            self.cam.BeginAcquisition()
            self.running = True
            _logger.info("Beginning image acquisition")

    def stop(self):
        """
        Stop recording images
        """
        if self.running:
            self.cam.EndAcquisition()
            _logger.info("Ending image acquisiton")
            print("Ending image acquisiton")
        self.running = False


    def apply_settings(self, settings):
        """
        Applies settings to the camera.

        Args:
            settings (dict): A dictionary of settings to apply
        """
        with self._lock:
            _logger.info(f"Setting camera properties")
            for setting, value in settings.items():
                if hasattr(self, setting):
                    try:
                        setattr(self, setting, value)
                        _logger.info(f"Set {setting} to {value}")
                    except Exception as err:
                        _logger.error(f"Failed to set {setting}: {err}")
                else:
                    _logger.warning(f"Unknown setting: {setting}")

    def apply_pattern(self):
        """
        Applies a test pattern to the camera
        """
        with self._lock:
            self.cam.TestPatternGeneratorSelector.SetValue(PySpin.TestPatternGeneratorSelector_Sensor)
            self.cam.TestPattern.SetValue(PySpin.TestPattern_SensorTestPattern)

    def capture_frame(self):
        """
        Capture a single image frame

        Returns:
            tuple: A tuple containing metadata (dict) and image data (numpy array)
        """
        with self._lock:
            comptime = str(datetime.now())
            image_result = self.cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                _logger.error(f"Image incomplete with status {image_result.GetImageStatus()}")
                return None
            
            # Right shift to get 12 bit
            image_converted = self.processor.Convert(image_result, PySpin.PixelFormat_Mono16)
            image_data = image_converted.GetNDArray() >> 4

            image_result.Release()

            # Get metadata
            # TODO: add more settings to metadata dictionary as required
            i = image_converted.GetFrameID()
            camtime = image_converted.GetTimeStamp()
            pxlfmt = image_converted.GetPixelFormatName()
            xoff = image_converted.GetXOffset()
            xpad = image_converted.GetXPadding()
            yoff = image_converted.GetYOffset()
            ypad = image_converted.GetYPadding()

            filename = f'images/raw/frame_{camtime}.npy'

            # Package data in a dict
            metadata = {
                "i": i,
                "rawfile": filename,
                "camtime": camtime,
                "comptime": comptime,
                "pxlfmt": pxlfmt,
                "xoff": xoff,
                "xpad": xpad,
                "yoff": yoff,
                "ypad": ypad,
                "exposure": self.exposure
                }

            return metadata, image_data

    def capture_frames(self, n_frames):
        """
        Capture multiple image frames.

        Args:
            n_frames (int): The number of frames to capture.

        Returns:
            list: A list of dictionaries containing metadata and image data for each frame.
        """

        results = []

        with self._lock:
            for frame in range(n_frames):
                # TODO: this duplicates code in capture_frame and can be refactored
                comptime = str(datetime.now())
                image_result = self.cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    _logger.error(f"Image incomplete with status {image_result.GetImageStatus()}")
                    return None

                # Convert image to correct format and release result
                image_converted = self.processor.Convert(image_result, PySpin.PixelFormat_Mono16)
                image_result.Release()

                # Right shift to get 12 bit
                image_data = image_converted.GetNDArray() >> 4

                # Get metadata
                # TODO: add more settings to metadata dictionary as required
                i = image_converted.GetFrameID()
                camtime = image_converted.GetTimeStamp()
                pxlfmt = image_converted.GetPixelFormatName()
                xoff = image_converted.GetXOffset()
                xpad = image_converted.GetXPadding()
                yoff = image_converted.GetYOffset()
                ypad = image_converted.GetYPadding()

                # Get data
                image_data = image_converted.GetNDArray()

                # Package data in a dict
                data = {
                    "i": i,
                    "frame": image_data,
                    "camtime": camtime,
                    "comptime": comptime,
                    "pxlfmt": pxlfmt,
                    "xoff": xoff,
                    "xpad": xpad,
                    "yoff": yoff,
                    "ypad": ypad,
                    "exposure": self.exposure
                    }

                results.append(data)

        return results