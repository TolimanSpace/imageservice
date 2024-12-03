import copy
import time
from multiprocessing import Process, Manager, Queue

import cv2
import serial

# from workers.csp import *
from workers.processing import *

import PySpin

_logger = logging.getLogger(__name__)

def acquire_frames(output_queue, shared_status):
    # Set up camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    camera = cam_list[0]

    camera.Init()

    camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    camera.ExposureTime.SetValue(50)
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(10)

    camera.BeginAcquisition()

    # Set up ImageProcessor instance
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    while True:
        # ret, frame = camera.capture_frame()
        image_result = camera.GetNextImage(1000)
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        else:
            # Testing
            i = image_result.GetFrameID()
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))
            #
            image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
            frame = image_converted.GetData().reshape(height,width)
            data = {"i": i, "frame": frame}
            output_queue.put(data)
            # Update shared_status
            shared_status['frame_acquisition'] = 'success'
            image_result.Release()
        # time.sleep(0.1)  # Maintain 10 Hz cadence
    else:
        camera.EndAcquisition()
        camera.DeInit()
        del camera
        cam_list.Clear()
        system.ReleaseInstance()


def frame_distributor(input_queue, process_queue, save_queue):
    while True:
        frame = input_queue.get()
        process_queue.put(copy.deepcopy(frame))
        save_queue.put(frame)


def process_frames(input_queue, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status):
    while True:
        frame = input_queue.get()
        centroid_data = find_centroid(frame["frame"])
        centroid_process_queue.put(copy.deepcopy(centroid_data))
        centroid_save_queue.put(copy.deepcopy(centroid_data))
        piezo_actuation_queue.put(centroid_data)


def save_to_disk(input_queue,shared_status):
    while True:
        frame = input_queue.get()
        cv2.imwrite(f'images/frame_{frame["i"]}.png', frame["frame"])


def serial_comm(centroid_queue,shared_status):
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    while True:
        centroid = centroid_queue.get()
        ser.write(str(centroid).encode())


def save_centroid(centroid_queue,shared_status):
    while True:
        centroid = centroid_queue.get()
        with open('centroids.txt', 'a') as file:
            file.write(f"{centroid}\n")


def actuate_piezo(centroid_queue,shared_status):
    while True:
        centroid = centroid_queue.get()
        # Actuate piezo system using centroid data


if __name__ == '__main__':
    manager = Manager()
    shared_status = manager.dict()

    frame_queue = Queue()
    process_queue = Queue()
    save_queue = Queue()
    centroid_process_queue = Queue()
    centroid_save_queue = Queue()
    piezo_actuation_queue = Queue()

    # Start the CSP processes
    # Process(target=csp_listener, args=(shared_status,)).start()
    # Process(target=csp_sender, args=(shared_status,)).start()

    # Start the frame acquisition and processing processes
    acquire_frame_process = Process(target=acquire_frames, args=(frame_queue, shared_status))
    acquire_frame_process.start()
    frame_distributor_process = Process(target=frame_distributor, args=(frame_queue, process_queue, save_queue))
    frame_distributor_process.start()
    process_frames_process = Process(target=process_frames,
            args=(process_queue, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status))
    process_frames_process.start()
    save_frame_process = Process(target=save_to_disk, args=(save_queue, shared_status))
    save_frame_process.start()
    # Process(target=serial_comm, args=(centroid_process_queue, shared_status)).start()
    save_centroid_process = Process(target=save_centroid, args=(centroid_save_queue, shared_status))
    save_centroid_process.start()
    Process(target=actuate_piezo, args=(piezo_actuation_queue, shared_status)).start()

    acquire_frame_process.join()
