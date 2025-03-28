import logging
import threading
import psutil
import os
from datetime import datetime
import numpy as np

# from simple_pyspin import Camera
import PySpin

_logger = logging.getLogger(__name__)

_SYSTEM = None

def list_cameras():
    """
    Return a list of Spinnaker cameras
    """

    global _SYSTEM

    if _SYSTEM is None:
        _SYSTEM = PySpin.System.GetInstance()
        _logger.info("Starting PySpin system instance")

    return _SYSTEM.GetCameras()


class CameraInterface:
    """
    A class to interface with a PySpin Camera

    Atributes
    ---------

    Methods
    -------

    """
    def __init__(self):
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
        return self.cam.ExposureTime.GetValue()

    @exposure.setter
    def exposure(self, value):
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(value)

    @property
    def framerate(self):
        return self.cam.AcquisitionFrameRate.GetValue()
        
    @framerate.setter
    def framerate(self, value):
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(value)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value

    @property
    def acquisitionmode(self):
        return self.cam.AcquisitionMode.GetValue()
        
    @acquisitionmode.setter
    def acquisitionmode(self, value):
        self.cam.AcquisitionMode.SetValue(value)
        
    @property
    def pixelformat(self):
        return self.cam.PixelFormat.GetValue()

    @pixelformat.setter
    def pixelformat(self,value):
        self.cam.PixelFormat.SetValue(value)

    @property
    def buffercount(self):
        return self.cam.TLStream.StreamBufferCountManual.GetValue()
    
    @buffercount.setter
    def buffercount(self,value):
        self.cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)
        self.cam.TLStream.StreamBufferCountManual.SetValue(value)

    @property
    def bufferhandlingmode(self):
        return self.cam.TLStream.StreamBufferHandlingMode.GetValue()
    
    @bufferhandlingmode.setter
    def bufferhandlingmode(self,value):
        self.cam.TLStream.StreamBufferHandlingMode.SetValue(value)

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
        self.running = False


    def apply_settings(self, settings):
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
        with self._lock:
            self.cam.TestPatternGeneratorSelector.SetValue(PySpin.TestPatternGeneratorSelector_Sensor)
            self.cam.TestPattern.SetValue(PySpin.TestPattern_SensorTestPattern)

    def capture_frame(self):
        """
        Capture an image
        """
        with self._lock:
            comptime = str(datetime.now())
            image_result = self.cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                _logger.error(f"Image incomplete with status {image_result.GetImageStatus()}")
                return None
            
            # Right shift to get 12 bit
            image_converted = self.processor.Convert(image_result, PySpin.PixelFormat_Mono16)
            image_data = image_converted.GetNDArray() # >> 4

            image_result.Release()

            # Get metadata
            i = image_converted.GetFrameID()
            # width = image_converted.GetWidth()
            # height = image_converted.GetHeight()
            camtime = image_converted.GetTimeStamp()
            pxlfmt = image_converted.GetPixelFormatName()
            xoff = image_converted.GetXOffset()
            xpad = image_converted.GetXPadding()
            yoff = image_converted.GetYOffset()
            ypad = image_converted.GetYPadding()

            filename = f'images/raw/frame_{camtime}.npy'

            # Package data in a dict
            data = {
                "i": i,
                "rawfile": filename,
                "camtime": camtime,
                "comptime": comptime,
                "pxlfmt": pxlfmt,
                "xoff": xoff,
                "xpad": xpad,
                "yoff": yoff,
                "ypad": ypad,
                # "exposure": self.exposure
                }
            
            # _logger.info(f"Grabbed Image {i}, width = {width}, height = {height} at time {camtime}")

            # Dump data
            # image_result.Save(filename)
            # image_data.tofile(filename)
            # np.save(filename, image_data)

            return data, image_data

    def capture_frames(self, n_frames):
        """
        Capture n_images
        """

        # Get n images
        # Calculate centroids and push to queue?
        # Store images and metadata
        # Return images and metadata in bulk

        results = []

        with self._lock:
            for frame in range(n_frames):
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
                i = image_converted.GetFrameID()
                width = image_converted.GetWidth()
                height = image_converted.GetHeight()
                camtime = image_converted.GetTimeStamp()
                pxlfmt = image_converted.GetPixelFormatName()
                xoff = image_converted.GetXOffset()
                xpad = image_converted.GetXPadding()
                yoff = image_converted.GetYOffset()
                ypad = image_converted.GetYPadding()

                _logger.info(f"Grabbed Image {i}, width = {width}, height = {height} at time {camtime}")

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


# camera = CameraInterface()


def handle_packet(packet):
    """Handle incoming CSP packets."""
    # Extract command data
    command_data = packet.data()

    # Check if it's a camera setting command
    if command_data.startswith("camera_setting:"):
        _, setting, value = command_data.split(":")
        setattr(camera, setting, value)

    # ... Handle other commands as necessary

    # Send response
    response = "OK"  # or any other appropriate response
    csp.sendto(packet.source, packet.destination, packet.port, response)


def camera_loop():
    """Continuously acquire frames from the camera using the provided settings."""
    while True:
        # Check and apply camera settings
        camera.apply_settings()

        # Capture frame
        frame = camera.capture_frame()

        # ... Do anything else required with the frame


def csp_listen_loop():
    """Continuously listen for incoming CSP packets."""
    while True:
        packet = csp.recv()
        if packet:
            handle_packet(packet)


if __name__ == "__main__":
    # Start the camera loop in its own thread
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.start()

    # Start the CSP listening loop in its own thread
    csp_thread = threading.Thread(target=csp_listen_loop)
    csp_thread.start()

    # Join the threads (optional, if you want the main thread to wait until both threads have finished)
    camera_thread.join()
    csp_thread.join()
