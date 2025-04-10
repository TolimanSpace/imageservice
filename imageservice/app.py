import copy
import time
from multiprocessing import Process, Manager, Queue, Pool, Pipe
import multiprocessing.shared_memory as shm
import logging
import psutil

import serial

# from workers.csp import *
from workers.images import dump_data, compress_netcdf, compress_netcdf_bulk
from workers.processing import *
from workers.camera import CameraInterface, SimulatedCameraInterface, BaseCameraInterface
from workers.compression import crop_centre
from dummy_csp import csp_listener


_logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename = "process_log.txt", level = logging.DEBUG, format=FORMAT)

DEDICATED_ACQUIRE_FRAMES_CORE = 2
# DEDICATED_SAVE_FRAMES_CORE = 3
# DEDICATED_SAVE_FRAMES_CORE_2 = 4
DEDICATED_FRAME_DISTRIBUTOR_CORE = 4
OTHER_CORES = [c for c in range(psutil.cpu_count())]
# OTHER_CORES = [c for c in range(psutil.cpu_count()) if c not in [DEDICATED_ACQUIRE_FRAMES_CORE]]
# OTHER_CORES = [c for c in range(psutil.cpu_count()) if c not in [DEDICATED_ACQUIRE_FRAMES_CORE, DEDICATED_SAVE_FRAMES_CORE, DEDICATED_FRAME_DISTRIBUTOR_CORE]]
OTHER_CORES = [c for c in range(psutil.cpu_count()) if c not in [DEDICATED_ACQUIRE_FRAMES_CORE,DEDICATED_FRAME_DISTRIBUTOR_CORE]]

FFI_SHAPE = (3648, 3648)
FFI_DTYPE = np.uint16

def run_camera(camera: BaseCameraInterface, output_pipe, shared_array, image_id, pid, cpu_num):
    _logger.debug(f"process {pid} on cpu-{cpu_num}: start acquire_frames_setup")

    camera.apply_settings(shared_status["camera_settings"])
    if shared_status["testing"] == True:
        camera.apply_pattern()
    camera.start()

    _logger.debug(f"cpu-{cpu_num}: end acquire_frames_setup")

    previous_image = -1

    while shared_status["begin_imaging"] == True:
        _logger.debug(f"process {pid} cpu-{cpu_num}: start acquire_frames")

        data, frame = camera.capture_frame()

        if data["i"] - previous_image > 1:
            _logger.warning(f"{data['i'] - previous_image - 1} frame(s) skipped")
        previous_image = data["i"]

        shared_array[:] = frame
        image_id[...] = data["camtime"]

        output_pipe.send(data)

        shared_status['frame_acquisition'] = 'success'

        _logger.debug(f"process {pid} cpu-{cpu_num}: end acquire_frames")

    camera.stop()

def acquire_frames(output_pipe, shared_mem_name, shared_id_name, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity([DEDICATED_ACQUIRE_FRAMES_CORE])
    # p.cpu_affinity(OTHER_CORES)
    pid = p.pid
    cpu_num = p.cpu_num()

    shared_mem = shm.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(FFI_SHAPE, dtype=FFI_DTYPE, buffer=shared_mem.buf)

    id_mem = shm.SharedMemory(name=shared_id_name)
    image_id = np.ndarray((), dtype=np.int64, buffer=id_mem.buf)

    while True:
        if "simulate" in shared_status and shared_status["simulate"] == True:
            _logger.info(f"Using simulated images")
            with SimulatedCameraInterface() as camera:
                run_camera(camera, output_pipe, shared_array, image_id, pid, cpu_num)

        else:
            with CameraInterface() as camera:
                run_camera(camera, output_pipe, shared_array, image_id, pid, cpu_num)

        if "close_app" in shared_status and shared_status["close_app"] == True:
            output_pipe.send(None)
            # _logger.debug(f"cpu-{cpu_num}: stopping acquire_frames worker")
            break

def frame_distributor(input_pipe, shared_mem_name, shared_id_name, process_pipe, save_pipe):

     # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity([DEDICATED_FRAME_DISTRIBUTOR_CORE])
    # p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    shared_mem = shm.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(FFI_SHAPE, dtype=FFI_DTYPE, buffer=shared_mem.buf)

    id_mem = shm.SharedMemory(name=shared_id_name)
    image_id = np.ndarray((), dtype=np.int64, buffer=id_mem.buf)

    while True:
        metadata = input_pipe.recv()
        frame = shared_array.copy()
        frame_id = image_id.copy()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start frame_distributor")

        if metadata is None:
            process_pipe.send(None)
            save_pipe.send(None)
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping frame_distributor worker")
            break

        # for frame in data:
        # frame_centre = crop_centre(frame['frame'], frame['frame'].shape[1]/2, frame['frame'].shape[0]/2)
        frame_centre = crop_centre(frame, frame.shape[1]/2, frame.shape[0]/2)
        process_pipe.send(frame_centre)

        if frame_id != metadata["camtime"]:
            _logger.error(f"Shared ID {frame_id} does not match camera time {metadata['camtime']}")

        filename = metadata["rawfile"]
        np.save(filename, frame)
        save_pipe.send(metadata)

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end frame_distributor")


def process_frames(input_pipe, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        image_centre = input_pipe.recv()

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start process_frames")

        if image_centre is None:
            centroid_process_queue.put(None)
            centroid_save_queue.put(None)
            piezo_actuation_queue.put(None)
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping process_frames worker")
            break

        # image = np.load(frame["rawfile"])
        # image_centre = crop_centre(image, image.shape[1]/2, image.shape[0]/2)

        centroid_data = find_centroid(image_centre)
        centroid_process_queue.put(copy.deepcopy(centroid_data))
        centroid_save_queue.put(copy.deepcopy(centroid_data))
        piezo_actuation_queue.put(centroid_data)

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end process_frames")


def save_to_disk(input_pipe, shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    # p.cpu_affinity([DEDICATED_SAVE_FRAMES_CORE])
    p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    shared_status["saving"] = True

    while True:
        data = input_pipe.recv()
        # filename = data["rawfile"]

        cpu_num = p.cpu_num()
        _logger.debug(f"process {pid} on cpu-{cpu_num}: start save_to_disk")

        if data is None:
            shared_status["saving"] = False
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping save_to_disk worker")
            break

        # result = create_fits(frame)
        result = dump_data(data)

        # cv2.imwrite(f'images/raw/frame_{frame["camtime"]}.png', frame["frame"])
        if not result:
            _logger.error(f"FITS file not created")

        _logger.debug(f"process {pid} on cpu-{cpu_num}: end save_to_disk")


def save_manager(input_queue, compress_queue, shared_status, max_workers=2):
    while True:
        processes = []
        time.sleep(1)
        while shared_status["begin_imaging"] == True:
            if len(processes) < max_workers:
                p = Process(target = save_to_disk, args=(input_queue, compress_queue, shared_status))
                p.start()
                processes.append(p)
                _logger.info(f"Spawned save_to_disk worker {len(processes)}")

            time.sleep(1)

        if input_queue.empty():
            for _ in processes[1:]:
                input_queue.put(None)

            for p in processes:
                p.join()

            _logger.info(f"Stopping save_manager")
            break

def compress(compress_queue,shared_status):

    # Set CPU affinity
    p = psutil.Process(os.getpid())
    # p.cpu_affinity(OTHER_CORES)
    pid = p.pid

    while True:
        # Get netcdf compression encoding to use
        netcdf_encoding = shared_status["compression_settings"].setdefault("netcdf_encoding", None)

        while shared_status["begin_compression"] == True:
            image_filenames = compress_queue.get()

            cpu_num = p.cpu_num()
            _logger.debug(f"process {pid} on cpu-{cpu_num}: start compress")

            if image_filenames is None:
                _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping compress werker")
                shared_status["end_compression"] = True
                shared_status["begin_compression"] = False
                break

            # result = compress_dump(image_filename)
            if len(image_filenames) == 1:
                result = compress_netcdf(image_filenames[0], encoding=netcdf_encoding)
            else:
                result = compress_netcdf_bulk(image_filenames, encoding=netcdf_encoding)

            if not result:
                _logger.error(f"Error compressing frame ")

            _logger.debug(f"process {pid} on cpu-{cpu_num}: end compress")

        if "end_compression" in shared_status and shared_status["end_compression"] == True:
            _logger.debug(f"process {pid} on cpu-{cpu_num}: stopping compress worker")
            break


def compress_manager(compress_queue, shared_status):
    while True:
        settings = shared_status.setdefault("compression_settings", {
            'max_workers': 4,
            'chunk_size': 1,
            'netcdf_encoding': {"zlib": True, "complevel": 9}
        })
        max_workers = shared_status["compression_settings"].setdefault("max_workers", 4)
        chunk_size = settings["chunk_size"] if "chunk_size" in settings else 10

        chunk_size = shared_status["compression_settings"]["chunk_size"]

        processes = []

        if "saving" in shared_status and shared_status["saving"] == False:
            # Populate queue
            _logger.info(f"Populating compression queue")
            image_dir = 'images/raw/'
            image_paths = sorted([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith('.npy')
            ])

            for i in range(0, len(image_paths), chunk_size):
                compress_queue.put(image_paths[i:i+chunk_size])

            #Add end signal
            compress_queue.put(None)

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

            _logger.info(f"Stopping compress_manager")
            break


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
    shared_mem = shm.SharedMemory(create=True, size=np.prod(FFI_SHAPE) * np.dtype(FFI_DTYPE).itemsize)
    shared_array = np.ndarray(FFI_SHAPE, dtype=FFI_DTYPE, buffer=shared_mem.buf)

    shared_id = shm.SharedMemory(create=True, size=np.dtype(np.int64).itemsize)
    image_id = np.ndarray((), dtype=np.int64, buffer=shared_id.buf)

    frame_parent_conn, frame_child_conn = Pipe()
    process_parent_conn, process_child_conn = Pipe()
    save_parent_conn, save_child_conn = Pipe()

    # frame_queue = Queue()
    # process_queue = Queue()
    # save_queue = Queue()
    centroid_process_queue = Queue()
    centroid_save_queue = Queue()
    piezo_actuation_queue = Queue()
    compress_queue = Queue()


    # Start the CSP processes
    Process(target=csp_listener, args=(shared_status,)).start()
    # Process(target=csp_sender, args=(shared_status,)).start()

    # Start the frame acquisition and processing processes
    acquire_frame_process = Process(target=acquire_frames, args=(frame_parent_conn, shared_mem.name, shared_id.name, shared_status))
    acquire_frame_process.start()

    frame_distributor_process = Process(target=frame_distributor, args=(frame_child_conn, shared_mem.name, shared_id.name, process_parent_conn, save_parent_conn))
    frame_distributor_process.start()

    process_frames_process = Process(target=process_frames,
            args=(process_child_conn, centroid_process_queue, centroid_save_queue, piezo_actuation_queue, shared_status))
    process_frames_process.start()
    
    save_frame_process = Process(target=save_to_disk, args=(save_child_conn, shared_status))
    save_frame_process.start()
    
    # save_manager_process = Process(target=save_manager, args=(save_queue, compress_queue, shared_status))
    # save_manager_process.start()

    serial_comm_process = Process(target=serial_comm, args=(centroid_process_queue, shared_status))
    serial_comm_process.start()
    save_centroid_process = Process(target=save_centroid, args=(centroid_save_queue, shared_status))
    save_centroid_process.start()
    piezo_process = Process(target=actuate_piezo, args=(piezo_actuation_queue, shared_status))
    piezo_process.start()

    compress_manager_process = Process(target = compress_manager, args=(compress_queue, shared_status))
    compress_manager_process.start()
 
    # Join processes
    acquire_frame_process.join()
    frame_distributor_process.join()
    process_frames_process.join()
    save_frame_process.join()
    # save_manager_process.join()
    serial_comm_process.join()
    save_centroid_process.join()
    piezo_process.join()
    compress_manager_process.join()

    shared_mem.close()
    shared_mem.unlink()
    shared_id.close()
    shared_id.unlink()


    _logger.debug(f"all workers finished")