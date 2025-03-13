import copy
import time
from multiprocessing import Process, Manager, Queue, Pool
import logging
import psutil

import cv2
import serial

# from workers.csp import *
from workers.images import create_fits, compress_image
from workers.processing import *
from workers.camera import CameraInterface
from workers.compression import crop_centre
from dummy_csp import csp_listener

import PySpin

_logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename = "process_log.txt", level = logging.DEBUG, format=FORMAT)

DEDICATED_ACQUIRE_FRAMES_CORE = 2
OTHER_CORES = [c for c in range(psutil.cpu_count()) if c != DEDICATED_ACQUIRE_FRAMES_CORE]

def acquire_frames(output_queue, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity([DEDICATED_ACQUIRE_FRAMES_CORE])
    pid = p.pid
    cpu_num = p.cpu_num()

    while True:
        with CameraInterface() as camera:
            _logger.debug(f"process {pid} on cpu-{cpu_num}: start acquire_frames_setup")

            camera.apply_settings(shared_status["camera_settings"])
            if shared_status["testing"] == True:
                camera.apply_pattern()
            camera.start()

            _logger.debug(f"cpu-{cpu_num}: end acquire_frames_setup")

            while shared_status["begin_imaging"] == True:
                _logger.debug(f"process {pid} cpu-{cpu_num}: start acquire_frames")

                data = camera.capture_frame()
                output_queue.put(data)

                shared_status['frame_acquisition'] = 'success'

                _logger.debug(f"process {pid} cpu-{cpu_num}: end acquire_frames")

            camera.stop()

        if "close_app" in shared_status and shared_status["close_app"] == True:
            output_queue.put(None)
            # _logger.debug(f"cpu-{cpu_num}: stopping acquire_frames worker")
            break

def frame_distributor(input_queue, process_queue, save_queue):

     # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        frame = input_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start frame_distributor")

        if frame is None:
            process_queue.put(None)
            save_queue.put(None)
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping frame_distributor worker")
            break

        # for frame in data:
        process_queue.put(copy.deepcopy(frame))
        save_queue.put(frame)

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end frame_distributor")


def process_frames(input_queue, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        frame = input_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start process_frames")

        if frame is None:
            centroid_process_queue.put(None)
            centroid_save_queue.put(None)
            piezo_actuation_queue.put(None)
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping process_frames worker")
            break

        centroid_data = find_centroid(frame["frame"])
        centroid_process_queue.put(copy.deepcopy(centroid_data))
        centroid_save_queue.put(copy.deepcopy(centroid_data))
        piezo_actuation_queue.put(centroid_data)

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end process_frames")


def save_to_disk(input_queue, compress_queue, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        frame = input_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start save_to_disk")

        if frame is None:
            compress_queue.put(None)
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping save_to_disk worker")
            break

        result = create_fits(frame)
        compress_queue.put(result)
        # cv2.imwrite(f'images/raw/frame_{frame["camtime"]}.png', frame["frame"])
        if not result:
            _logger.error(f"FITS file not created")

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end save_to_disk")

def compress(compress_queue,shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    # p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        while shared_status["begin_compression"] == True:
            image_filename = compress_queue.get()

            cpu_num = p.cpu_num()
            _logger.debug(f"process {pid} on cpu-{cpu_num}: start compress")

            if image_filename is None:
                _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping compress worker")
                shared_status["end_compression"] = True
                shared_status["begin_compression"] = False
            result = compress_image(image_filename)

            if not result:
                _logger.error(f"Error compressing frame ")

            _logger.debug(f"process {pid} on cpu-{cpu_num}: end compress")

        if "end_compression" in shared_status and shared_status["end_compression"] == True:
            break


def compress_manager(compress_queue, shared_status, max_workers=3):
    while True:
        processes = []
        while shared_status["begin_compression"] == True:
            if len(processes) < max_workers:
                p = Process(target = compress, args=(compress_queue, shared_status))
                p.start()
                processes.append(p)
                _logger.info(f"Spawned compress worker {len(processes)}")

            time.sleep(1)

        if "end_compression" in shared_status and shared_status["end_compression"] == True:
            for _ in processes[1:]:
                compress_queue.put(None)

            for p in processes:
                p.join()


def serial_comm(centroid_queue,shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    # ser = serial.Serial('/dev/ttyUSB0', 9600)
    while True:
        centroid = centroid_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start serial_comm")

        if centroid is None:
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping serial_comm worker")
            break

        # ser.write(str(centroid).encode())

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end serial_comm")


def save_centroid(centroid_queue,shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        centroid = centroid_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start save_centroid")

        if centroid is None:
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping save_centroid worker")
            break

        with open('centroids.txt', 'a') as file:
            file.write(f"{centroid}\n")

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end save_centroid")


def actuate_piezo(centroid_queue,shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        centroid = centroid_queue.get()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start actuate_piezo")

        if centroid is None:
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping actuate_piezo worker")
            break

        # Actuate piezo system using centroid data

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end actuate_piezo")


if __name__ == '__main__':
    manager = Manager()
    shared_status = manager.dict()

    frame_queue = Queue()
    process_queue = Queue()
    save_queue = Queue()
    centroid_process_queue = Queue()
    centroid_save_queue = Queue()
    piezo_actuation_queue = Queue()
    compress_queue = Queue()

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
    save_frame_process = Process(target=save_to_disk, args=(save_queue, compress_queue, shared_status))
    save_frame_process.start()
    serial_comm_process = Process(target=serial_comm, args=(centroid_process_queue, shared_status))
    serial_comm_process.start()
    save_centroid_process = Process(target=save_centroid, args=(centroid_save_queue, shared_status))
    save_centroid_process.start()
    piezo_process = Process(target=actuate_piezo, args=(piezo_actuation_queue, shared_status))
    piezo_process.start()
    compress_manager_process = Process(target = compress_manager, args=(compress_queue, shared_status))
    compress_manager_process.start()
 

    # # Start the compression processes
    # with Pool(processes=3) as compress_pool:
    #     while not compress_queue.empty():
    #         compress_pool.apply_async(compress, args=(compress_queue, shared_status))

    #     compress_pool.close()
    #     compress_pool.join()


    acquire_frame_process.join()
    frame_distributor_process.join()
    process_frames_process.join()
    save_frame_process.join()
    serial_comm_process.join()
    save_centroid_process.join()
    piezo_process.join()
    compress_manager_process.join()


    _logger.debug(f"all workers finished")