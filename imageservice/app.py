import copy
import time
from multiprocessing import Process, Manager, Queue

# import cv2
import serial

from workers.csp import *
from workers.processing import *

_logger = logging.getLogger(__name__)

def acquire_frames(output_queue, shared_status):
    camera = camera()
    while True:
        ret, frame = camera.capture_frame()
        if ret:
            output_queue.put(frame)
            # Update shared_status
            shared_status['frame_acquisition'] = 'success'
        time.sleep(0.1)  # Maintain 10 Hz cadence


def frame_distributor(input_queue, process_queue, save_queue):
    while True:
        frame = input_queue.get()
        process_queue.put(copy.deepcopy(frame))
        save_queue.put(frame)


def process_frames(input_queue, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status):
    while True:
        frame = input_queue.get()
        centroid_data = find_centroid(frame)
        centroid_process_queue.put(copy.deepcopy(centroid_data))
        centroid_save_queue.put(copy.deepcopy(centroid_data))
        piezo_actuation_queue.put(centroid_data)


def save_to_disk(input_queue,shared_status):
    while True:
        frame = input_queue.get()
        cv2.imwrite('frame.png', frame)


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
    Process(target=csp_sender, args=(shared_status,)).start()

    # Start the frame acquisition and processing processes
    Process(target=acquire_frames, args=(frame_queue, shared_status)).start()
    Process(target=frame_distributor, args=(frame_queue, process_queue, save_queue)).start()
    Process(target=process_frames,
            args=(process_queue, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status)).start()
    Process(target=save_to_disk, args=(save_queue, shared_status)).start()
    Process(target=serial_comm, args=(centroid_process_queue, shared_status)).start()
    Process(target=save_centroid, args=(centroid_save_queue, shared_status)).start()
    Process(target=actuate_piezo, args=(piezo_actuation_queue, shared_status)).start()
