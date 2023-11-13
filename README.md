# imageservice


```mermaid
graph TD;
    Camera(Camera) -->|captures frame| A[acquire_frames Process]
    A -->|frame| FQ[frame_queue]
    FQ -->|frame| FD[frame_distributor Process]
    FD -->|frame copy| PQ[process_queue]
    FD -->|frame copy| SQ[save_queue]
    PQ -->|frame| PC[process_frames Process]
    SQ -->|frame| SD[save_to_disk Process]
    PC -->|centroid| CPQ[centroid_process_queue]
    PC -->|centroid| CSDQ[centroid_save_queue]
    PC -->|centroid| PAQ[piezo_actuation_queue]
    CPQ -->|centroid| SC[serial_comm Process]
    CSDQ -->|centroid| CS[centroid_save Process]
    PAQ -->|centroid| PA[piezo_actuator Process]
    SC -->|sends centroid| SP[Serial Port]
    CS -->|saves centroid| CF[Centroid File]
    SD -->|saves frame| Disk
    PA -->|actuates| PS[Piezo System]
