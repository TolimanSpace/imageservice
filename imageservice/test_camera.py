import logging
import PySpin
from workers.camera import CameraInterface

logging.basicConfig(level=logging.DEBUG)

camera_settings = {
        'acquisitionmode': PySpin.AcquisitionMode_Continuous,
        'exposure': 50,
        'framerate': 10,
        'pixelformat': PySpin.PixelFormat_Mono12Packed
    }

with CameraInterface() as cam:

    cam.apply_settings(camera_settings)
    # cam.apply_pattern()

    cam.start()
    data = cam.capture_frame()
    cam.stop()

# with CameraInterface() as cam:
#     cam.start()
#     data = [cam.capture_frame() for n in range(10)]
#     cam.stop()
