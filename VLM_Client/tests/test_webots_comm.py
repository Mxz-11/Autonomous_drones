import socket
import struct
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from webots_comm import recv_exact, decode_frame_rgb


class TestRecvExact:
    def _make_sock(self, *chunks: bytes) -> MagicMock:
        sock = MagicMock(spec=socket.socket)
        sock.recv.side_effect = list(chunks)
        return sock

    def test_single_chunk_full_data(self):
        sock = self._make_sock(b"hello")
        result = recv_exact(sock, 5)
        assert result == b"hello"

    def test_multiple_chunks_reassembled(self):
        sock = self._make_sock(b"hel", b"lo")
        result = recv_exact(sock, 5)
        assert result == b"hello"

    def test_many_small_chunks(self):
        sock = self._make_sock(b"a", b"b", b"c", b"d", b"e")
        result = recv_exact(sock, 5)
        assert result == b"abcde"

    def test_empty_chunk_raises_connection_error(self):
        sock = self._make_sock(b"hi", b"")
        with pytest.raises(ConnectionError):
            recv_exact(sock, 5)

    def test_timeout_raises_timeout_error(self):
        sock = MagicMock(spec=socket.socket)
        sock.recv.side_effect = socket.timeout("timed out")
        with pytest.raises(TimeoutError):
            recv_exact(sock, 10)

    def test_returns_correct_length(self):
        data = bytes(range(20))
        sock = self._make_sock(data)
        result = recv_exact(sock, 20)
        assert len(result) == 20


class TestDecodeFrameRgb:
    def _make_bgra_bytes(self, h: int, w: int,
                         b: int, g: int, r: int, a: int = 255) -> bytes:
        pixel = bytes([b, g, r, a])
        return pixel * (h * w)

    def test_output_shape_is_hwc_rgb(self):
        data = self._make_bgra_bytes(10, 20, 100, 150, 200)
        frame = decode_frame_rgb(data, width=20, height=10)
        assert frame.shape == (10, 20, 3)

    def test_bgra_to_rgb_channel_order(self):
        data = self._make_bgra_bytes(1, 1, b=10, g=20, r=30)
        frame = decode_frame_rgb(data, width=1, height=1)
        assert frame[0, 0, 0] == 30  # R
        assert frame[0, 0, 1] == 20  # G
        assert frame[0, 0, 2] == 10  # B

    def test_output_dtype_is_uint8(self):
        data = self._make_bgra_bytes(5, 5, 128, 128, 128)
        frame = decode_frame_rgb(data, width=5, height=5)
        assert frame.dtype == np.uint8

    def test_uniform_color_frame(self):
        data = self._make_bgra_bytes(4, 6, b=50, g=100, r=200)
        frame = decode_frame_rgb(data, width=6, height=4)
        assert np.all(frame[:, :, 0] == 200)
        assert np.all(frame[:, :, 1] == 100)
        assert np.all(frame[:, :, 2] == 50)

    def test_no_alpha_channel_in_output(self):
        data = self._make_bgra_bytes(8, 8, 0, 0, 0)
        frame = decode_frame_rgb(data, width=8, height=8)
        assert frame.shape[2] == 3
