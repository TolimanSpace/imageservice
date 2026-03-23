import logging
import threading
import os
import time
from datetime import datetime
import numpy as np

import PySpin

_logger = logging.getLogger(__name__)

_SYSTEM = None

def list_cameras():
    """
    Return a list of Spinnaker cameras

    Initializes the PySpin system instance if it is not already initialized,
    and returns a list of available cameras.

    Returns:
        PySpin.CameraList: A list of available Spinnaker cameras.
    """

    global _SYSTEM

    if _SYSTEM is None:
        _SYSTEM = PySpin.System.GetInstance()
        _logger.info("Starting PySpin system instance")

    return _SYSTEM.GetCameras()

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