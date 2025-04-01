import cProfile
import logging
import numpy as np
import PySpin
import time
from workers.camera import CameraInterface
from datetime import datetime
import os

logging.basicConfig(level=logging.DEBUG)

# os.nice(-10)

camera_settings = {
        'acquisitionmode': PySpin.AcquisitionMode_Continuous,
        'exposure': 50,
        'pixelformat': PySpin.PixelFormat_Mono12p,
        'framerate': 10,
        'bufferhandlingmode': PySpin.StreamBufferHandlingMode_NewestOnly,
        'width': 3648,
        'offsetx': 912,
    }

# with CameraInterface() as cam:

#     cam.apply_settings(camera_settings)
#     # cam.apply_pattern()

#     cam.start()
#     data = cam.capture_frame()
#     cam.stop()

# total = []

with cProfile.Profile() as pr:
    with CameraInterface() as cam:
        cam.apply_settings(camera_settings)

        # time.sleep(3)
        cam.start()

        for n in range(100):
            data, frame = cam.capture_frame()
            # filename = data["rawfile"]
            # np.save(filename, frame, allow_pickle=False)

        cam.stop()

    pr.print_stats()

print(f"Last frame: {data['i']}")

# print("---")
# print(f"total median time: {np.median(total)} + {np.percentile(total, 84) - np.percentile(total, 50)} - {np.percentile(total, 50) - np.percentile(total, 16)}")
# print("---")

