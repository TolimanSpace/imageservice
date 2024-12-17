import copy
import time
from multiprocessing import Process, Manager, Queue
import logging

import cv2
import serial

# from workers.csp import *
from workers.images import create_fits
from workers.processing import *
from workers.camera import CameraInterface
from dummy_csp import csp_listener

import PySpin

_logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG)

def acquire_frames(output_queue, shared_status):
    while True:
        while shared_status["enable_camera"] == True:
            with CameraInterface() as camera:
                camera.apply_settings(shared_status["camera_settings"])
                if shared_status["testing"] == True:
                    camera.apply_pattern()
                camera.start()

                while shared_status["begin_imaging"] == True:
                    data = camera.capture_frame()
                    output_queue.put(data)

                    shared_status['frame_acquisition'] = 'success'

                camera.stop()


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

        result = create_fits(frame)

        cv2.imwrite(f'images/raw/frame_{frame["camtime"]}.png', frame["frame"])

        if not result:
            _logger.error(f"FITS file not created")

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
    Process(target=csp_listener, args=(shared_status,)).start()
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
    frame_distributor_process.join()
    process_frames_process.join()
    save_frame_process.join()
    save_centroid_process.join()
