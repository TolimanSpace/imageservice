# Vendored from the SEC repository (TolimanSpace/toliman-sec), libs/toliman-service/python/toliman_service.py
# at commit 43e893e. Don't edit it here: change it there and copy it again.
"""Python version of the `toliman-service` crate: everything a SEC payload service needs to
talk to the executive over its Unix socket. Standard library only (Python 3.8+).

Subclass `Service` and call `run(sid, service)`. `run` connects to the executive, registers
the SID, answers port 23 commands with `Service.command`, calls `Service.tick` periodically,
sends the telemetry and events queued on the `Context`, and reconnects whenever the executive
restarts. `example_service.py` is a complete service to copy from.

What the executive expects (docs/icd.md):

- SID: 1 TipTilt, 2 Camera or 4 LVDS. The first message on a connection registers it.
- Commands arrive as (command, params) from port 23. Return 0x00 for success, 0x01 for a
  generic failure, or a service-defined code 0x02-0xEF; anything above becomes 0x01. The OBC
  gets 0xF3 if no reply comes within 5 s, so start longer jobs in the background. Commands are
  handled one at a time.
- Telemetry is one fixed-length record per SID: the first record's length (1-503 bytes) holds
  until the executive restarts. Push at least once per HK interval (default 10 s). Start the
  record with a layout version byte. The executive stamps each record with the SEC's Unix time
  when it arrives and archives every record on the SEC, so push at the rate ground should see.
- Events are short free-text lines for the event log files the OBC downloads.

Wire format (toliman-ipc): a 5-byte header `>BBBH` (source SID, destination SID, message
type, payload length) and the payload. CmdReq/CmdRep payloads start with a 2-byte request id
that the reply echoes.
"""

import logging
import os
import queue
import socket
import struct
import time

HEADER = struct.Struct(">BBBH")
CMD_REQ, CMD_REP, TLM_PUSH, EVT_LOG = 0x01, 0x02, 0x03, 0x04
SID_EXECUTIVE = 0
DEFAULT_SOCKET = "/tmp/toliman-executive.sock"
MAX_TELEMETRY_LEN = 503  # 512-byte CSP buffer minus the SID and Unix ms header
QUEUE_DEPTH = 64

OK = 0x00
ERROR = 0x01
SERVICE_MAX = 0xEF

log = logging.getLogger("toliman_service")


class Context:
    """Queues telemetry and events for the executive. Thread-safe."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=QUEUE_DEPTH)

    def telemetry(self, record: bytes) -> None:
        if not 1 <= len(record) <= MAX_TELEMETRY_LEN:
            log.warning("telemetry record of %d bytes not sent (1-%d)", len(record), MAX_TELEMETRY_LEN)
            return
        self._put(TLM_PUSH, bytes(record))

    def event(self, message: str) -> None:
        log.info("event: %s", message)
        self._put(EVT_LOG, message.encode("utf-8"))

    def _put(self, msg_type, payload):
        try:
            self._queue.put_nowait((msg_type, payload))
        except queue.Full:
            log.warning("message dropped: queue full")

    def _drain(self):
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return


class Service:
    def command(self, ctx: Context, command: int, params: bytes) -> int:
        """Handles one port 23 command; returns the reply byte."""
        raise NotImplementedError

    def tick(self, ctx: Context) -> None:
        """Called every `tick` seconds, e.g. to push telemetry."""

    def connected(self, ctx: Context) -> None:
        """Called after each (re)connection, once the SID is registered."""


def _frame(sid, msg_type, payload):
    return HEADER.pack(sid, SID_EXECUTIVE, msg_type, len(payload)) + payload


def run(sid, service, socket_path=None, tick=1.0, reconnect=1.0, stop=None, ctx=None):
    """Runs `service` until `stop` (a threading.Event) is set, or for ever."""
    socket_path = socket_path or os.environ.get("TOLIMAN_IPC_SOCKET", DEFAULT_SOCKET)
    ctx = ctx or Context()
    next_tick = time.monotonic()
    warned = False
    while not (stop and stop.is_set()):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(socket_path)
            # An event registers the SID without fixing the telemetry record length.
            sock.sendall(_frame(sid, EVT_LOG, f"SID {sid} service connected".encode()))
        except OSError as e:
            sock.close()
            if not warned:
                log.warning("cannot connect to %s: %s; retrying", socket_path, e)
                warned = True
            time.sleep(reconnect)
            continue
        warned = False
        log.info("connected to %s as SID %d", socket_path, sid)
        service.connected(ctx)
        try:
            next_tick = _serve(sid, service, ctx, sock, tick, next_tick, stop)
        except OSError as e:
            log.warning("IPC error: %s", e)
        finally:
            sock.close()
        if not (stop and stop.is_set()):
            log.warning("disconnected from the executive; reconnecting")
            time.sleep(reconnect)


def _serve(sid, service, ctx, sock, tick, next_tick, stop):
    buf = b""
    while not (stop and stop.is_set()):
        for msg_type, payload in ctx._drain():
            sock.sendall(_frame(sid, msg_type, payload))
        now = time.monotonic()
        if now >= next_tick:
            service.tick(ctx)
            next_tick = max(next_tick + tick, now)
            continue
        sock.settimeout(min(next_tick - now, 0.1))
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        if not chunk:
            return next_tick
        buf += chunk
        while len(buf) >= HEADER.size:
            _src, _dst, msg_type, length = HEADER.unpack_from(buf)
            if len(buf) < HEADER.size + length:
                break
            payload, buf = buf[HEADER.size:HEADER.size + length], buf[HEADER.size + length:]
            reply = _handle(service, ctx, msg_type, payload)
            if reply is not None:
                sock.sendall(_frame(sid, CMD_REP, reply))
    return next_tick


def _handle(service, ctx, msg_type, payload):
    """Returns the CmdRep payload for a CmdReq, else None."""
    if msg_type != CMD_REQ:
        log.warning("ignoring message type %d from the executive", msg_type)
        return None
    if len(payload) < 2:
        log.warning("CmdReq without a req_id; ignored")
        return None
    req_id, body = payload[:2], payload[2:]
    if not body:
        return req_id + bytes([ERROR])
    try:
        code = service.command(ctx, body[0], bytes(body[1:]))
    except Exception:
        log.exception("command 0x%02X failed", body[0])
        code = ERROR
    if not isinstance(code, int) or not 0 <= code <= SERVICE_MAX:
        log.warning("command 0x%02X returned %r; sending 0x01", body[0], code)
        code = ERROR
    return req_id + bytes([code])
