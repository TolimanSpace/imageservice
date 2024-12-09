import logging
from workers.camera import CameraInterface

logging.basicConfig(level=logging.DEBUG)

with CameraInterface() as cam:
    cam.start()
    data = cam.capture_frame()
    cam.stop()

with CameraInterface() as cam:
    cam.start()
    data = [cam.capture_frame() for n in range(10)]
    cam.stop()
