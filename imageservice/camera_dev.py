import PySpin
from datetime import datetime

system = PySpin.System.GetInstance()

processor = PySpin.ImageProcessor()
processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_NONE)

def get_image():

    cam_list = system.GetCameras()
    cam = cam_list[0]

    cam.Init()

    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(50)
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(10)
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
    cam.TestPatternGeneratorSelector.SetValue(PySpin.TestPatternGeneratorSelector_Sensor)
    cam.TestPattern.SetValue(PySpin.TestPattern_SensorTestPattern)

    cam.BeginAcquisition()
    image_result = cam.GetNextImage(1000)
    time_0 = datetime.now()
    image = image_result.GetNDArray()
    time_1 = datetime.now()
    print(str(time_1-time_0))
    image_result.Release()
    cam.EndAcquisition()

    cam.DeInit()

    return image

def convert_16_to_12_bit(image):
    time_0 = datetime.now()
    image = image >> 4
    time_1 = datetime.now()
    print(str(time_1-time_0))

    return image


def save_image(image_result):
    time_0 = datetime.now()
    filename = f'images/raw/frame_{image_result.GetTimeStamp()}.raw'
    image_result.Save(filename)
    time_1 = datetime.now()

    print(str(time_1-time_0))

    return True

def clear_system():

    system.ReleaseInstance()

    return True
