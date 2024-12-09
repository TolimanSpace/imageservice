import logging
import threading

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
        self._exposure = None
        self._resolution = None
        self._lock = threading.Lock()

        cam_list = list_cameras()
        if not cam_list.GetSize():
            _logger.error("No cameras detected")
            raise RuntimeError("No cameras detected")
        self.cam = cam_list.GetByIndex(0)
        cam_list.Clear()
        self.running = False
        

    @property
    def exposure(self):
        with self._lock:
            return self._exposure

    @exposure.setter
    def exposure(self, value):
        with self._lock:
            self._exposure = value

    @property
    def resolution(self):
        with self._lock:
            return self._resolution

    @resolution.setter
    def resolution(self, value):
        with self._lock:
            self._resolution = value

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


    def apply_settings(self):
        # ... pseudocode to apply the camera settings
        pass

    def capture_frame(self):
        """
        Capture an image
        """
        with self._lock:
            image_result = self.cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                _logger.error(f"Image incomplete with status {image_result.GetImageStatus()}")
                return None
            
            i = image_result.GetFrameID()
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            _logger.info(f"Grabbed Image {i}, width = {width}, height = {height}")
            image_data = image_result.GetNDArray()
            image_result.Release()
            return image_data


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
