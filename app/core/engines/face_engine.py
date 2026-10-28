"""Engine restorasi wajah (GFPGAN / CodeFormer) + fallback deteksi wajah OpenCV.

Strategi:
  1. Bila ``gfpgan-v1_4.onnx`` tersedia -> gunakan inferensi ONNX penuh.
  2. Bila tidak -> gunakan *fallback klasik*: deteksi wajah Haar/DNN, lalu
     perbaiki area wajah dengan denoise + unsharp ringan. Ini membuat fitur
     "Face Enhance" tetap berfungsi offline tanpa model besar (kualitas tentu
     di bawah GFPGAN; README mencatat hal ini).

Hasil :class:`FaceResult` berisi patch wajah yang sudah diselaraskan warna
(menggunakan histogram matching sederhana) agar penyatuan kembali halus.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.device import DeviceInfo
from app.core.engines.base_engine import BaseEngine
from app.core.engines.classic_engine import denoise, from_u8, to_u8, unsharp_mask
from app.core.types import CancellationToken, EngineError

log = logging.getLogger("iep.face")

_FACE_CASCADE_NAME = "haarcascade_frontalface_default.xml"


@dataclass
class FaceBox:
    """Lokasi satu wajah pada gambar (koordinat piksel)."""

    x: int
    y: int
    w: int
    h: int

    def expand(self, frac: float, img_w: int, img_h: int) -> "FaceBox":
        """Perbesar kotak sebesar ``frac`` (dipakai utk margin dahi/dagu)."""
        dx, dy = int(self.w * frac), int(self.h * frac)
        return FaceBox(max(0, self.x - dx), max(0, self.y - dy),
                       min(img_w, self.w + 2 * dx) - max(0, self.x - dx),
                       min(img_h, self.h + 2 * dy) - max(0, self.y - dy))


@dataclass
class FaceResult:
    box: FaceBox
    patch: np.ndarray          # float32 HxWx3 hasil restorasi (ukuran == box)
    mask: np.ndarray           # float32 HxW bobot blending elips (halus)


def detect_faces(img: np.ndarray) -> list[FaceBox]:
    """Deteksi wajah pada float32 0..1 memakai cascade Haar bawaan OpenCV."""
    gray = cv2.cvtColor(to_u8(np.clip(img, 0, 1)), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    try:
        path = cv2.data.haarcascades + _FACE_CASCADE_NAME  # type: ignore[attr-defined]
        cascade = _get_cascade(path)
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(int(gray.shape[0] * 0.05),) * 2)
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
    except Exception as exc:  # pragma: no cover
        log.warning("Deteksi wajah gagal: %s", exc)
        return []


_CASCADE_CACHE: dict[str, cv2.CascadeClassifier] = {}


def _get_cascade(path: str) -> cv2.CascadeClassifier:
    if path not in _CASCADE_CACHE:
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise EngineError("Berkas cascade wajah OpenCV tidak ditemukan")
        _CASCADE_CACHE[path] = c
    return _CASCADE_CACHE[path]


def face_blending_mask(h: int, w: int, feather: float = 0.25) -> np.ndarray:
    """Mask elips dengan tepi ter-feather untuk penempelan wajah yang halus."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    nx, ny = (xx - cx) / max(1.0, cx), (yy - cy) / max(1.0, cy)
    d = np.sqrt(nx * nx + ny * ny)
    inner = 1.0 - 2.0 * feather
    m = np.clip((1.0 - d) / max(1e-6, 1.0 - inner), 0.0, 1.0)
    m = m ** 0.75   # kurva lembut
    return m.astype(np.float32)[..., None]


def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Selaraskan distribusi warna ``source`` ke ``reference`` (per kanal, uint8)."""
    s = to_u8(source).astype(np.float32)
    r = to_u8(reference)
    out = np.empty_like(s)
    for c in range(3):
        sc, rc = s[..., c].ravel(), r[..., c].ravel()
        s_vals, s_idx = np.unique(sc, return_inverse=True)
        r_counts = np.bincount(to_u8(reference)[..., c].ravel(), minlength=256).astype(np.float32)
        s_counts = np.bincount(sc.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
        s_cdf = np.cumsum(s_counts) / max(1.0, s_counts.sum())
        r_cdf = np.cumsum(r_counts) / max(1.0, r_counts.sum())
        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        out[..., c] = lut[sc.astype(np.uint8)].reshape(s[..., c].shape)
    return from_u8(np.clip(out, 0, 255).astype(np.uint8))


class FaceEnhancer:
    """Orkestrasi restorasi wajah: coba GFPGAN ONNX, fallback ke perbaikan lokal."""

    def __init__(self, device: DeviceInfo, model_path: Optional[Path] = None) -> None:
        self.device = device
        self.model_path = model_path
        self._session = None
        self._tried = False

    # ------------------------------------------------------------------ #
    @property
    def has_model(self) -> bool:
        """Apakah model GFPGAN onnx tersedia?"""
        return bool(self.model_path and Path(self.model_path).exists())

    def _load_session(self):
        if self._session is not None or self._tried:
            return self._session
        self._tried = True
        if not self.has_model:
            return None
        try:
            import onnxruntime as ort

            from app.core.device import build_session_options

            so, providers = build_session_options(self.device)
            self._session = ort.InferenceSession(str(self.model_path), sess_options=so,
                                                 providers=providers)
        except Exception as exc:
            log.warning("GFPGAN tidak dapat dimuat (%s); pakai fallback klasik", exc)
            self._session = None
        return self._session

    # ------------------------------------------------------------------ #
    def enhance(self, img: np.ndarray, strength: int,
                token: Optional[CancellationToken] = None) -> tuple[np.ndarray, int]:
        """Restorasi semua wajah pada gambar.

        Args:
            img: float32 HxWx3 0..1 (gambar HASIL upscale).
            strength: 0-100 -> kekuatan blending wajah hasil vs asli.
            token: pembatalan.

        Returns:
            (gambar baru, jumlah wajah yang diproses). Gambar tak berubah bila
            tidak ada wajah.
        """
        faces = detect_faces(img)
        if not faces:
            return img, 0
        h, w = img.shape[:2]
        out = img.copy()
        alpha = float(strength) / 100.0
        session = self._load_session()
        count = 0
        for box in faces:
            if token is not None:
                token.raise_if_canceled()
            region = out[box.y:box.y + box.h, box.x:box.x + box.w]
            if session is not None:
                patched = self._run_gfpgan(session, region)
            else:
                patched = self._classic_face_fix(region)
            patched = match_histogram(patched, region)      # jaga tone asli
            mask = face_blending_mask(box.h, box.w)
            blended = region * (1 - mask * alpha) + patched * (mask * alpha)
            out[box.y:box.y + box.h, box.x:box.x + box.w] = blended
            count += 1
        return out, count

    # ------------------------------------------------------------------ #
    def _classic_face_fix(self, region: np.ndarray) -> np.ndarray:
        """Fallback: denoise ringan + sharpen sedang pada area wajah.

        NLMeans dibatasi pada wilayah kecil saja (hemat CPU); untuk crop wajah
        besar dipakai bilateral filter sebagai gantinya.
        """
        h, w = region.shape[:2]
        if h * w <= 512 * 512:
            fixed = denoise(region, strength=35)
        else:
            u8 = to_u8(np.clip(region, 0, 1))
            fixed = from_u8(cv2.bilateralFilter(u8, d=7, sigmaColor=24, sigmaSpace=7))
        fixed = unsharp_mask(fixed, amount=0.6, radius=1.0)
        return fixed

    def _run_gfpgan(self, session, region: np.ndarray) -> np.ndarray:
        """Inferensi GFPGAN ONNX (input NCHW BGR 0..1, sisi kelipatan 8)."""
        u8 = to_u8(np.clip(region, 0, 1))
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        size = 512
        face = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        inp = face[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        res = session.run([out_name], {in_name: inp})[0][0]
        res = np.transpose(res, (1, 2, 0))[:, :, ::-1]       # -> RGB
        res = cv2.resize(to_u8(np.clip(res, 0, 1)), (region.shape[1], region.shape[0]))
        return from_u8(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


# Integrasi sebagai engine 1:1 agar bisa dipakai lewat registry yang sama.
class FaceEngine(BaseEngine):
    """Adapter :class:`FaceEnhancer` ke antarmuka BaseEngine (skala 1:1)."""

    def __init__(self, device: DeviceInfo, model_path: Optional[Path] = None) -> None:
        super().__init__(device)
        self.key = "face"
        self.display_name = "Restorasi Wajah (GFPGAN)"
        self.native_scale = 1
        self.enhancer = FaceEnhancer(device, model_path)
        self.strength = 60

    def load(self) -> None:
        self._loaded = True

    def process_tile(self, tile: np.ndarray, token: Optional[CancellationToken] = None) -> np.ndarray:
        out, _ = self.enhancer.enhance(tile, self.strength, token)
        return out
