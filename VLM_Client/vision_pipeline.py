"""
Codificación ligera de frames para VLM: redimensionado + JPEG con compresión.
"""

from __future__ import annotations

import base64
import re
import io
from typing import Literal

import numpy as np
from PIL import Image

Profile = Literal["default", "tiny"]

# Producción: ~384px, calidad 78 → más detalle para el VLM (accuracy > speed)
DEFAULT_MAX_EDGE = 384
DEFAULT_JPEG_QUALITY = 78

TINY_MAX_EDGE = 256
TINY_JPEG_QUALITY = 65


def _resize_rgb(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale = min(max_edge / float(w), max_edge / float(h), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def frame_rgb_to_jpeg_bytes(
    frame_rgb: np.ndarray,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        raise ValueError("Se espera frame RGB (H,W,3+)")
    rgb = frame_rgb[:, :, :3]
    im = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    im = _resize_rgb(im, max_edge)
    buf = io.BytesIO()
    im.save(
        buf,
        format="JPEG",
        quality=int(max(1, min(quality, 95))),
        optimize=True,
        subsampling=2,
    )
    return buf.getvalue()


def jpeg_bytes_to_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def jpeg_b64_to_data_url(b64: str) -> str:
    return f"data:image/jpeg;base64,{b64}"


def encode_frame_for_vlm(
    frame_rgb: np.ndarray,
    *,
    profile: Profile = "default",
) -> tuple[str, str]:
    """
    Returns:
        (base64_sin_prefijo, data_url) para usar con LangChain / OpenAI-style.
    """
    if profile == "tiny":
        raw = frame_rgb_to_jpeg_bytes(
            frame_rgb, max_edge=TINY_MAX_EDGE, quality=TINY_JPEG_QUALITY
        )
    else:
        raw = frame_rgb_to_jpeg_bytes(
            frame_rgb, max_edge=DEFAULT_MAX_EDGE, quality=DEFAULT_JPEG_QUALITY
        )
    b64 = jpeg_bytes_to_b64(raw)
    return b64, jpeg_b64_to_data_url(b64)


def encode_both_profiles(frame_rgb: np.ndarray) -> tuple[str, str]:
    """Principal + fallback compacto (reintento ante contexto lleno)."""
    b64_main, _ = encode_frame_for_vlm(frame_rgb, profile="default")
    b64_tiny, _ = encode_frame_for_vlm(frame_rgb, profile="tiny")
    return b64_main, b64_tiny


_CTX_ERR_RE = re.compile(
    r"context|token|length|maximum|too long|window|exceed",
    re.IGNORECASE,
)


def is_context_window_error(exc: BaseException) -> bool:
    """Heurística OpenAI/LM Studio / límites de contexto."""
    msg = f"{type(exc).__name__} {exc}"
    if _CTX_ERR_RE.search(msg):
        return True
    err = getattr(exc, "response", None)
    if err is not None and getattr(err, "text", None):
        if _CTX_ERR_RE.search(err.text):
            return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        try:
            em = str(body.get("error", {}).get("message", ""))
            if _CTX_ERR_RE.search(em):
                return True
        except (TypeError, AttributeError):
            pass
    return False
