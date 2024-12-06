import PySpin
import cv2

# TYPE = "pipeline"
TYPE = "sensor"

def main():
    
    system = PySpin.System.GetInstance()

    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No cameras detected")
        system.ReleaseInstance()
        return
    
    cam = cam_list[0]

    try:
        cam.Init()

        if TYPE == "pipeline":
            if cam.TestPatternGeneratorSelector.GetAccessMode() == PySpin.RW:
                cam.TestPatternGeneratorSelector.SetValue(PySpin.TestPatternGeneratorSelector_PipelineStart)

            if cam.TestPattern.GetAccessMode() == PySpin.RW:
                cam.TestPattern.SetValue(PySpin.TestPattern_Increment)
        elif TYPE == "sensor":
            if cam.TestPatternGeneratorSelector.GetAccessMode() == PySpin.RW:
                cam.TestPatternGeneratorSelector.SetValue(PySpin.TestPatternGeneratorSelector_Sensor)

            if cam.TestPattern.GetAccessMode() == PySpin.RW:
                cam.TestPattern.SetValue(PySpin.TestPattern_SensorTestPattern)
        else:
            print("No test pattern selected")

        cam.BeginAcquisition()

        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        for i in range(25):
            image_result = cam.GetNextImage()

            if image_result.IsIncomplete():
                print(f"Image incomplete with status {image_result.GetImageStatus()}")
            else:
                j = image_result.GetFrameID()
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                print('Grabbed Image %d, width = %d, height = %d' % (j, width, height))
                #
                image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
                frame = image_converted.GetData().reshape(height,width)
                cv2.imwrite(f'images/frame_{j}.png', frame)
            
            image_result.Release()

        cam.EndAcquisition()

        if cam.TestPattern.GetAccessMode() == PySpin.RW:
            cam.TestPattern.SetValue(PySpin.TestPattern_Off)

    except PySpin.SpinnakerException as err:
        print(f"Error: {err}")
    finally:
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
    
if __name__ == "__main__":
    main()