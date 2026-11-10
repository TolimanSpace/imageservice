"""
pointing_link.py - Centroid datagrams from the Camera service to the pointing service.

The SEC's pointing loop (SEC repository, docs/pointing.md) runs in its own
service (SID 5). After each frame, the acquisition loop sends it one Centroid
datagram over a Unix datagram socket: TOLIMAN_POINTING_SOCKET, default
/tmp/toliman-pointing.sock. This replaces the serial publisher stub
(open question C9).

Sending never blocks and never fails the acquisition: if the pointing service
is not running, or its socket buffer is full, the datagram is dropped (a slow
reader loses old samples instead of queueing them).

Centroid layout, 32 bytes little endian, struct '<BBBBIQffff' (the SEC
repository's libs/pointing-link, whose Python twin pointing_link.py has
Centroid.encode(); keep the two in step):

    u8   kind             0x01
    u8   version          1
    u8   flags            bit 0 no signal, bit 1 saturated
    u8   reserved         0
    u32  frame_id         camera acq_nframe (wraps)
    u64  unix_time_ns     SEC Unix time when the frame arrived
    f32  dx, dy           pixels; same sign as PointingError.dx/dy
    f32  total_intensity  centroid weight sum (ADU)
    f32  peak             maximum pixel value in the centre ROI (ADU)

dx and dy are PointingError.dx/dy: from the sensor centre, which is the ROI
centre when the centre ROI is centred on the sensor (the default geometry).
"""

from __future__ import annotations

import logging
import os
import socket
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

POINTING_SOCKET = "/tmp/toliman-pointing.sock"
SOCKET_ENV = "TOLIMAN_POINTING_SOCKET"

CENTROID_KIND = 0x01
VERSION = 1
FLAG_NO_SIGNAL = 0x01
FLAG_SATURATED = 0x02

CENTROID = struct.Struct("<BBBBIQffff")

# Peak value at which a frame counts as saturated (12-bit full scale)
SATURATION_ADU = 4095


def encode_centroid(
    frame_id: int,
    unix_time_ns: int,
    dx: float,
    dy: float,
    total_intensity: float,
    peak: float,
    no_signal: bool = False,
    saturated: bool = False,
) -> bytes:
    """The 32-byte Centroid datagram."""
    flags = (FLAG_NO_SIGNAL if no_signal else 0) | (FLAG_SATURATED if saturated else 0)
    return CENTROID.pack(
        CENTROID_KIND, VERSION, flags, 0,
        frame_id & 0xFFFFFFFF, unix_time_ns & 0xFFFFFFFFFFFFFFFF,
        dx, dy, total_intensity, peak,
    )


class CentroidSender:
    """
    Sends one Centroid datagram per frame to the pointing service.

    Use send_pointing() for a frame with a centroid and send_no_signal()
    for one without. Errors are counted, logged once per kind, and dropped.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path if path is not None else os.environ.get(SOCKET_ENV, POINTING_SOCKET)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = 0
        self.dropped = 0
        self._logged: set = set()

    def send_pointing(self, frame, pointing) -> None:
        """A centroid for `frame` (AcquiredFrame) from `pointing` (PointingError)."""
        peak = pointing.centroid.peak_value
        self._send(encode_centroid(
            frame.frame_id, frame.unix_time_ns,
            pointing.dx, pointing.dy,
            pointing.centroid.total_intensity, peak,
            saturated=peak >= SATURATION_ADU,
        ))

    def send_no_signal(self, frame, centre_roi: Optional[np.ndarray]) -> None:
        """No centroid for `frame`; the peak still tells ground how dark it was."""
        peak = int(centre_roi.max()) if centre_roi is not None and centre_roi.size else 0
        self._send(encode_centroid(
            frame.frame_id, frame.unix_time_ns, 0.0, 0.0, 0.0, peak,
            no_signal=True, saturated=peak >= SATURATION_ADU,
        ))

    def _send(self, datagram: bytes) -> None:
        try:
            self.sock.sendto(datagram, self.path)
            self.sent += 1
        except OSError as e:
            # FileNotFoundError / ConnectionRefusedError: no pointing service;
            # BlockingIOError: its queue is full. Either way the sample is stale
            # by the next frame.
            self.dropped += 1
            kind = type(e).__name__
            if kind not in self._logged:
                self._logged.add(kind)
                logger.info("Centroid datagrams to %s dropped (%s); further drops not logged", self.path, e)

    def close(self) -> None:
        self.sock.close()
        if self.sent or self.dropped:
            logger.info("Centroid datagrams: %d sent, %d dropped", self.sent, self.dropped)
