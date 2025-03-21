import time
import PySpin
import logging

_logger = logging.getLogger(__name__)

OBSERVING_TIME = 300

# CSP Listener Function
def csp_listener(shared_status):
    camera_settings = {
        'acquisitionmode': PySpin.AcquisitionMode_Continuous,
        'exposure': 50,
        'framerate': 10,
        'pixelformat': PySpin.PixelFormat_Mono12Packed
    }
    shared_status["camera_settings"] = camera_settings
    shared_status["testing"] = True

    _logger.info(f"Stop compression process")
    shared_status["begin_compression"] = False

    shared_status["enable_camera"] = True
    _logger.info(f"Camera enambled")
    time.sleep(1)
    _logger.info(f"Beginning imaging")
    shared_status["begin_imaging"] = True
    _logger.info(f"Imaging for {OBSERVING_TIME} seconds")
    time.sleep(OBSERVING_TIME)
    shared_status["begin_imaging"] = False
    _logger.info(f"Imaging stopped")
    shared_status["enable_camera"] = False
    _logger.info(f"Camera disabled")

    # time.sleep(10)

    # shared_status["enable_camera"] = True
    # _logger.info(f"Camera enabled")
    # time.sleep(1)
    # _logger.info(f"Beginning imaging")
    # shared_status["begin_imaging"] = True
    # time.sleep(60)
    # shared_status["begin_imaging"] = False
    # _logger.info(f"Imaging stopped")
    # shared_status["enable_camera"] = False
    # _logger.info(f"Camera disabled")
    
    _logger.info(f"Begin compression process")
    shared_status["begin_compression"] = True
    
    _logger.info("Initialize app shutdown")
    shared_status["close_app"] = True