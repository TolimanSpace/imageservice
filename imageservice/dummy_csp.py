import time
import PySpin

# CSP Listener Function
def csp_listener(shared_status):
    camera_settings = {
        'acquisitionmode': PySpin.AcquisitionMode_Continuous,
        'exposure': 50,
        'framerate': 10,
        'pixelformat': PySpin.PixelFormat_Mono8
    }
    shared_status["camera_settings"] = camera_settings
    shared_status["testing"] = True

    shared_status["enable_camera"] = True
    time.sleep(1)
    shared_status["begin_imaging"] = True
    time.sleep(10)
    shared_status["begin_imaging"] = False
    shared_status["enable_camera"] = False
    time.sleep(10)
    shared_status["enable_camera"] = True
    time.sleep(1)
    shared_status["begin_imaging"] = True
    time.sleep(10)
    shared_status["begin_imaging"] = False
    shared_status["enable_camera"] = False
