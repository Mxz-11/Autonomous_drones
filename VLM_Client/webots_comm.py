import socket
import struct
import time

import numpy as np

from mission_config import WEBOTS_IP, WEBOTS_PORT, SOCKET_TIMEOUT_S, DIST_SENSOR_ENABLED, DOWN_CAM_ENABLED
from advanced_logger import get_logger


def connect_to_webots(ip: str = WEBOTS_IP, port: int = WEBOTS_PORT) -> socket.socket:
    logger = get_logger("socket")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(SOCKET_TIMEOUT_S)
            sock.connect((ip, port))
            logger.info("Conectado a Webots en %s:%d", ip, port)
            return sock
        
        except ConnectionRefusedError:
            logger.warning("Conexión rechazada, reintentando en 2s…")
            time.sleep(2)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))

        except socket.timeout as exc:
            raise TimeoutError(f"Timeout recibiendo {n} bytes (recibidos {len(data)})") from exc
        
        if not chunk:
            raise ConnectionError("Socket cerrado por Webots")
        
        data += chunk
    return data


def decode_frame_rgb(frame_bytes: bytes, width: int, height: int) -> np.ndarray:
    frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 4))
    return frame[:, :, 2::-1]


_DOWN_CAM_FALLBACK = np.zeros((160, 160, 3), dtype=np.uint8)


def receive_frame(sock: socket.socket, position_floats: int = 3, position_enabled: bool = True) -> tuple[np.ndarray, int, int, float, float, float, float, np.ndarray]:
    sock.send(b"FRAME\n")

    header = recv_exact(sock, 8)
    w, h = struct.unpack("ii", header)
    if not (0 < w <= 2000 and 0 < h <= 2000):
        raise ConnectionError(f"Cabecera inválida: {w}×{h}")

    img_bytes = recv_exact(sock, w * h * 4)
    frame_rgb = decode_frame_rgb(img_bytes, w, h)

    pos_x = pos_y = pos_z = 0.0
    if position_enabled:
        pos_bytes = recv_exact(sock, 4 * position_floats)
        if position_floats == 2:
            pos_x, pos_y = struct.unpack("ff", pos_bytes)
            
        else:
            pos_x, pos_y, pos_z = struct.unpack("fff", pos_bytes)

    dist_m = 0.0
    if DIST_SENSOR_ENABLED:
        dist_bytes = recv_exact(sock, 4)
        dist_m = struct.unpack("f", dist_bytes)[0]

    down_frame_rgb = _DOWN_CAM_FALLBACK
    if DOWN_CAM_ENABLED:
        down_header = recv_exact(sock, 8)
        dw, dh = struct.unpack("ii", down_header)
        if 0 < dw <= 1000 and 0 < dh <= 1000:
            down_bytes = recv_exact(sock, dw * dh * 4)
            down_frame_rgb = decode_frame_rgb(down_bytes, dw, dh)

    return frame_rgb, w, h, pos_x, pos_y, pos_z, dist_m, down_frame_rgb
