"""Engine klasik berbasis OpenCV/NumPy: denoise, unsharp, CLAHE, JPEG-fix,
koreksi warna, dan upscale cadangan (fallback) ketika model AI belum dipasang.

Semua fungsi menerima/mengembalikan float32 0..1 HxWx3 (kecuali disebut lain),
sehingga mudah diuji tanpa GUI dan tanpa model.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.core.device import DeviceInfo
from app.core.engines.base_engine import BaseEngine, register_engine
from app.core.types import CancellationToken


# --------------------------------------------------------------------------- #
# Fungsi murni (dipakai juga oleh pipeline untuk pra/pasca-proses)
# --------------------------------------------------------------------------- #
def to_u8(arr: np.ndarray) -> np.ndarray:
    """float32 0..1 -> uint8 dengan clipping aman."""
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def from_u8(arr: np.ndarray) -> np.ndarray:
    """uint8 -> float32 0..1."""
    return arr.astype(np.float32) / 255.0


def denoise(img: np.ndarray, strength: int) -> np.ndarray:
    """Denoise Non-Local Means; ``strength`` 0-100 dipetakan ke h 1..25.

    Kecepatan diturunkan pada gambar besar (threshold cepat NLMeans).
    """
    if strength <= 0:
        return img
    u8 = to_u8(img)
    h = 1.0 + (strength / 100.0) * 24.0
    out = cv2.fastNlMeansDenoisingColored(u8, None, h=h, hForColorComponents=h,
                                          templateWindowSize=7, searchWindowSize=21)
    return from_u8(out)


def remove_jpeg_artifacts(img: np.ndarray, strength: int = 40) -> np.ndarray:
    """Kurangi blocking artefak JPEG dengan bilateral filter ringan + deblocking.

    Dipakai sebelum upscale agar model tidak memperkuat kotak-kotak kompresi.
    """
    u8 = to_u8(img)
    try:
        u8 = cv2.detailEnhance(u8, sigma_s=8, sigma_r=0.05)   # haluskan blok, jaga tepi
    except cv2.error:  # pragma: no cover - fallback bila build tanpa contrib
        pass
    d = max(1, int(strength / 100 * 6))
    blur = cv2.bilateralFilter(u8, d=9, sigmaColor=d * 8, sigmaSpace=d)
    # blend konservatif agar tekstur asli tidak hilang
    mixed = cv2.addWeighted(u8, 0.65, blur, 0.35, 0)
    return from_u8(mixed)


def unsharp_mask(img: np.ndarray, amount: float, radius: float = 1.2,
                 threshold: float = 0.01) -> np.ndarray:
    """Unsharp mask adaptif. ``amount`` 0..~3 (0 = tanpa efek)."""
    if amount <= 0.001:
        return img
    base = to_u8(img).astype(np.float32)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=radius)
    detail = base - blurred
    if threshold > 0:
        t = threshold * 255.0
        detail = np.where(np.abs(detail) < t, 0.0, detail)
    sharpened = base + detail * amount
    return from_u8(np.clip(sharpened, 0, 255).astype(np.uint8))


def clahe(img: np.ndarray, strength: int) -> np.ndarray:
    """CLAHE pada kanal L (LAB) supaya terang kontras lokal naik tanpa geser warna."""
    if strength <= 0:
        return img
    clip = 1.0 + (strength / 100.0) * 3.0
    lab = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2LAB)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[..., 0] = eq.apply(lab[..., 0])
    return from_u8(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def auto_levels(img: np.ndarray, cutoff: float = 0.5) -> np.ndarray:
    """Auto-levels per kanal dengan pemangkasan persentil ``cutoff``%."""
    u8 = to_u8(img)
    out = np.empty_like(u8)
    channels = u8.shape[2] if u8.ndim == 3 else 1
    view = u8[..., None] if u8.ndim == 2 else u8
    for c in range(channels):
        ch = view[..., c]
        lo, hi = np.percentile(ch, cutoff), np.percentile(ch, 100 - cutoff)
        if hi - lo < 2:
            out[..., c] = ch
            continue
        scaled = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
        out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return from_u8(out)


def adjust_color(img: np.ndarray, brightness: int = 0, contrast: int = 0,
                 saturation: int = 0) -> np.ndarray:
    """Koreksi warna sederhana. Masing-masing -100..100, 0 = netral.

    - Brightness: geser aditif (0..1 space).
    - Contrast: skala sekitar titik tengah 0.5.
    - Saturation: interpolasi menuju/menjauhi versi grayscale.
    """
    out = img.copy()
    if brightness:
        out = out + (brightness / 100.0) * 0.5
    if contrast:
        factor = 1.0 + (contrast / 100.0)
        out = (out - 0.5) * factor + 0.5
    out = np.clip(out, 0.0, 1.0)
    if saturation:
        gray = out.mean(axis=2, keepdims=True)
        s = 1.0 + (saturation / 100.0)
        out = np.clip(gray + (out - gray) * s, 0.0, 1.0)
    return out.astype(np.float32)


def resize_lanczos(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize presisi tinggi (Lanczos-4 untuk turun, Cubic untuk naik)."""
    interp = cv2.INTER_LANCZOS4 if height < img.shape[0] or width < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=interp).astype(np.float32)


def guided_blend(base: np.ndarray, enhanced: np.ndarray, balance: int) -> np.ndarray:
    """Campur hasil model (enhanced) dengan basis halus sesuai slider detail.

    balance 0 => lebih smooth (perbanyak enhanced yang sudah dilembutkan),
    balance 100 => preserve detail maksimal (enhanced penuh + unsharp ekstra).
    Implementasi praktis: bobot pencampuran [halus..tajam].
    """
    b = float(balance) / 100.0
    smooth = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.6)
    return (smooth * (1.0 - b) + enhanced * b).astype(np.float32)


# --------------------------------------------------------------------------- #
# Engine fallback "classic" — jalan tanpa model apa pun
# --------------------------------------------------------------------------- #
@register_engine("classic_x2")
@register_engine("classic_x4")
class ClassicUpscaleEngine(BaseEngine):
    """Upscale non-AI (Lanczos + edge-aware sharpening) sebagai fallback offline.

    Mutu di bawah Real-ESRGAN, tetapi membuat aplikasi tetap dapat digunakan
    dan diuji end-to-end walaupun folder ``models/`` masih kosong.
    """

    def __init__(self, device: DeviceInfo, scale: int = 4) -> None:
        super().__init__(device)
        self.native_scale = scale
        self.key = f"classic_x{scale}"
        self.display_name = f"Klasik (Lanczos+Sharpen) x{scale}"

    def load(self) -> None:
        self._loaded = True

    def process_tile(self, tile: np.ndarray, token: Optional[CancellationToken] = None) -> np.ndarray:
        if token is not None:
            token.raise_if_canceled()
        h, w = tile.shape[:2]
        up = resize_lanczos(tile, w * self.native_scale, h * self.native_scale)
        # edge-aware sharpen sederhana agar hasil tidak lembek
        return unsharp_mask(up, amount=0.7, radius=1.6)


@register_engine("classic_identity")
class ClassicIdentityEngine(BaseEngine):
    """Engine 1:1 (tanpa upscale) untuk mode 'hanya perbaikan kualitas'."""

    def __init__(self, device: DeviceInfo) -> None:
        super().__init__(device)
        self.native_scale = 1
        self.key = "classic_identity"
        self.display_name = "Klasik 1:1 (denoise+sharpen)"

    def load(self) -> None:
        self._loaded = True

    def process_tile(self, tile: np.ndarray, token: Optional[CancellationToken] = None) -> np.ndarray:
        if token is not None:
            token.raise_if_canceled()
        sharp = unsharp_mask(tile, amount=0.5, radius=1.2)
        return guided_blend(tile, sharp, balance=60)
