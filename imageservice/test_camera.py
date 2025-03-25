import logging
import numpy as np
import PySpin
from workers.camera import CameraInterface

logging.basicConfig(level=logging.DEBUG)

camera_settings = {
        'acquisitionmode': PySpin.AcquisitionMode_Continuous,
        'exposure': 50,
        'framerate': 10,
        'pixelformat': PySpin.PixelFormat_Mono12Packed
        # 'pixelformat': PySpin.PixelFormat_Mono8
    }

# with CameraInterface() as cam:

#     cam.apply_settings(camera_settings)
#     # cam.apply_pattern()

#     cam.start()
#     data = cam.capture_frame()
#     cam.stop()

GetNextImage = []
Convert = []
GetNDArray = []
metadata = []
save = []
total = []

with CameraInterface() as cam:
    cam.apply_settings(camera_settings)

    cam.start()
    for n in range(300):
        data, timing = cam.capture_frame()
        GetNextImage.append(timing["GetNextImage"])
        Convert.append(timing["Convert"])
        GetNDArray.append(timing["GetNDArray"])
        metadata.append(timing["metadata"])
        save.append(timing["save"])
        total.append(timing["total"])
    cam.stop()

print("---")
print(f"GetNextImage median time: {np.median(GetNextImage)} + {np.percentile(GetNextImage, 84) - np.percentile(GetNextImage, 50)} - {np.percentile(GetNextImage, 50) - np.percentile(GetNextImage, 16)} ")
print("---")
print(f"Convert median time: {np.median(Convert)} + {np.percentile(Convert, 84) - np.percentile(Convert, 50)} - {np.percentile(Convert, 50) - np.percentile(Convert, 16)}")
print("---")
print(f"GetNDArray median time: {np.median(GetNDArray)} + {np.percentile(GetNDArray, 84) - np.percentile(GetNDArray, 50)} - {np.percentile(GetNDArray, 50) - np.percentile(GetNDArray, 16)}")
print("---")
print(f"metadata median time: {np.median(metadata)} + {np.percentile(metadata, 84) - np.percentile(metadata, 50)} - {np.percentile(metadata, 50) - np.percentile(metadata, 16)}")
print("---")
print(f"save median time: {np.median(save)} + {np.percentile(save, 84) - np.percentile(save, 50)} - {np.percentile(save, 50) - np.percentile(save, 16)}")
print("---")
print(f"total median time: {np.median(total)} + {np.percentile(total, 84) - np.percentile(total, 50)} - {np.percentile(total, 50) - np.percentile(total, 16)}")
print("---")

