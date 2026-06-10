"""Tests for vision_pipeline.py — frame encoding and error detection."""
import base64

import numpy as np
import pytest

from vision_pipeline import (
    frame_rgb_to_jpeg_bytes,
    jpeg_bytes_to_b64,
    jpeg_b64_to_data_url,
    encode_frame_for_vlm,
    encode_both_profiles,
    encode_single,
    is_context_window_error,
    DEFAULT_MAX_EDGE,
    TINY_MAX_EDGE,
)


@pytest.fixture
def rgb_frame():
    """100×80 RGB frame with some color variation."""
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:40, :, 0] = 200
    frame[40:, :, 2] = 180
    return frame


@pytest.fixture
def large_frame():
    """600×800 frame that will be resized."""
    return np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)


class TestFrameRgbToJpegBytes:
    def test_returns_bytes(self, rgb_frame):
        result = frame_rgb_to_jpeg_bytes(rgb_frame)
        assert isinstance(result, bytes)

    def test_output_is_nonempty(self, rgb_frame):
        result = frame_rgb_to_jpeg_bytes(rgb_frame)
        assert len(result) > 0

    def test_starts_with_jpeg_magic(self, rgb_frame):
        result = frame_rgb_to_jpeg_bytes(rgb_frame)
        assert result[:2] == b"\xff\xd8"

    def test_large_frame_is_resized(self, large_frame):
        result = frame_rgb_to_jpeg_bytes(large_frame, max_edge=100)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 100

    def test_smaller_quality_gives_smaller_file(self, large_frame):
        high = frame_rgb_to_jpeg_bytes(large_frame, quality=90)
        low = frame_rgb_to_jpeg_bytes(large_frame, quality=20)
        assert len(low) < len(high)

    def test_wrong_ndim_raises_value_error(self):
        bad = np.zeros((80, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            frame_rgb_to_jpeg_bytes(bad)

    def test_fewer_than_3_channels_raises_value_error(self):
        bad = np.zeros((80, 100, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            frame_rgb_to_jpeg_bytes(bad)

    def test_rgba_frame_accepted(self, rgb_frame):
        rgba = np.dstack([rgb_frame, np.full((*rgb_frame.shape[:2], 1), 255, dtype=np.uint8)])
        result = frame_rgb_to_jpeg_bytes(rgba)
        assert result[:2] == b"\xff\xd8"

    def test_already_small_frame_not_upscaled(self, rgb_frame):
        result = frame_rgb_to_jpeg_bytes(rgb_frame, max_edge=384)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 80)


class TestJpegBytesToB64:
    def test_returns_string(self, rgb_frame):
        raw = frame_rgb_to_jpeg_bytes(rgb_frame)
        b64 = jpeg_bytes_to_b64(raw)
        assert isinstance(b64, str)

    def test_valid_base64(self, rgb_frame):
        raw = frame_rgb_to_jpeg_bytes(rgb_frame)
        b64 = jpeg_bytes_to_b64(raw)
        decoded = base64.b64decode(b64)
        assert decoded == raw

    def test_ascii_only_characters(self, rgb_frame):
        raw = frame_rgb_to_jpeg_bytes(rgb_frame)
        b64 = jpeg_bytes_to_b64(raw)
        b64.encode("ascii")


class TestJpegB64ToDataUrl:
    def test_prefix(self):
        url = jpeg_b64_to_data_url("abc123")
        assert url.startswith("data:image/jpeg;base64,")

    def test_b64_appended(self):
        url = jpeg_b64_to_data_url("MYDATA")
        assert url.endswith("MYDATA")


class TestEncodeFrameForVlm:
    def test_default_profile_returns_tuple(self, rgb_frame):
        b64, data_url = encode_frame_for_vlm(rgb_frame)
        assert isinstance(b64, str)
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_tiny_profile_smaller_than_default(self, large_frame):
        b64_def, _ = encode_frame_for_vlm(large_frame, profile="default")
        b64_tiny, _ = encode_frame_for_vlm(large_frame, profile="tiny")
        assert len(b64_tiny) < len(b64_def)

    def test_data_url_contains_b64(self, rgb_frame):
        b64, data_url = encode_frame_for_vlm(rgb_frame)
        assert b64 in data_url


class TestEncodeBothProfiles:
    def test_returns_two_strings(self, rgb_frame):
        b64_main, b64_tiny = encode_both_profiles(rgb_frame)
        assert isinstance(b64_main, str)
        assert isinstance(b64_tiny, str)

    def test_profiles_differ(self, large_frame):
        b64_main, b64_tiny = encode_both_profiles(large_frame)
        assert b64_main != b64_tiny


class TestEncodeSingle:
    def test_returns_base64_string(self, rgb_frame):
        b64 = encode_single(rgb_frame)
        assert isinstance(b64, str)
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8"

    def test_custom_max_edge(self, large_frame):
        b64_small = encode_single(large_frame, max_edge=64)
        b64_large = encode_single(large_frame, max_edge=256)
        assert len(b64_small) < len(b64_large)


class TestIsContextWindowError:
    def test_detects_context_keyword_in_message(self):
        exc = ValueError("context length exceeded")
        assert is_context_window_error(exc) is True

    def test_detects_token_keyword(self):
        exc = RuntimeError("maximum token length")
        assert is_context_window_error(exc) is True

    def test_detects_exceed_keyword(self):
        exc = Exception("prompt exceed window size")
        assert is_context_window_error(exc) is True

    def test_unrelated_exception_returns_false(self):
        exc = ValueError("some unrelated error")
        assert is_context_window_error(exc) is False

    def test_dict_body_with_error_message(self):
        exc = Exception("API error")
        exc.body = {"error": {"message": "context window exceeded"}}
        assert is_context_window_error(exc) is True

    def test_response_text_checked(self):
        exc = Exception("API error")
        resp = type("Resp", (), {"text": "token limit exceeded"})()
        exc.response = resp
        assert is_context_window_error(exc) is True
