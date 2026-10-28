"""Engine Real-ESRGAN berbasis ONNX Runtime.

Mendukung berkas ``.onnx`` hasil ekspor arsitektur RRDBNet (Real-ESRGAN x2/x4,
x4-anime) dengan deteksi skala otomatis dari metadata/ujung graf model.
Bila file model tidak ada, pipeline akan jatuh ke engine klasik (fallback).

Kebijakan memori:
  * Model di-cache per proses (lazy load, satu instance per kunci).
  * FP16 diaktifkan bila perangkat GPU mendukung dan diminta.
  * Kesalahan "allocasi gagal" dari ORT diterjemahkan menjadi
    :class:`OutOfMemoryError` agar pipeline dapat mengecilkan tile & mencoba lagi.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.device import DeviceInfo, build_session_options
from app.core.engines.base_engine import BaseEngine
from app.core.types import CancellationToken, EngineError, OutOfMemoryError

log = logging.getLogger("iep.esrgan")

# Cache session global: key -> (session, native_scale). Model mahal dimuat sekali.
_SESSION_CACHE: dict[str, tuple[object, int]] = {}
_CACHE_LOCK = threading.Lock()


def _find_onnx(path: Path) -> Optional[Path]:
    """Cari berkas ONNX yang cocok; kembalikan None bila belum terpasang."""
    if path.exists():
        return path
    # beri kelonggaran penamaan (mis. user menaruh realesrgan_x4plus.onnx)
    stem_guesses = [
        path.parent.glob("*x4*.onnx"),
        path.parent.glob("*anime*.onnx"),
    ]
    if "x2" in path.name.lower():
        stem_guesses.insert(0, path.parent.glob("*x2*.onnx"))
    for g in stem_guesses:
        for p in g:
            return p
    return None


def detect_native_scale(session, filename: str) -> int:
    """Tentukan skala native model dari bentuk output vs input."""
    m = re.search(r"x(\d)", filename.lower())
    try:
        in_shape = session.get_inputs()[0].shape
        out_shape = session.get_outputs()[0].shape
        if all(isinstance(v, int) for v in (*in_shape, *out_shape)):
            ih, iw = in_shape[2], in_shape[3]
            oh, ow = out_shape[2], out_shape[3]
            if iw and ow and ow % iw == 0:
                return ow // iw
            if ih and oh and oh % ih == 0:
                return oh // ih
    except Exception:  # shape dinamis (-1 / string) — pakai tebakan nama file
        pass
    return int(m.group(1)) if m else 4


class EsrganOnnxEngine(BaseEngine):
    """Wrapper satu model RRDBNet/ONNX menjadi engine tile-per-tile."""

    def __init__(self, device: DeviceInfo, model_path: Path, key: str = "esrgan",
                 display_name: str = "Real-ESRGAN (ONNX)", fp16: bool = False) -> None:
        super().__init__(device)
        self.model_path = Path(model_path)
        self.key = key
        self.display_name = display_name
        self.fp16 = fp16 and device.fp16_supported
        self._session = None
        self.native_scale = 4

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Muat (atau ambil dari cache) session ONNX Runtime."""
        if self._loaded and self._session is not None:
            return
        path = _find_onnx(self.model_path)
        if path is None:
            raise EngineError(
                f"Model '{self.display_name}' belum terpasang.\n"
                f"Jalankan skrip: python scripts/download_models.py\n"
                f"atau impor berkas .onnx lewat menu Model."
            )
        cache_key = f"{path}|{self.device.kind}|{int(self.fp16)}"
        with _CACHE_LOCK:
            cached = _SESSION_CACHE.get(cache_key)
            if cached is None:
                import onnxruntime as ort

                so, providers = build_session_options(self.device)
                try:
                    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
                except Exception as exc:
                    raise EngineError(f"Gagal memuat model ONNX: {exc}") from exc
                scale = detect_native_scale(sess, path.name)
                cached = (sess, scale)
                _SESSION_CACHE[cache_key] = cached
                log.info("Model dimuat: %s (skala x%d, provider=%s)", path.name, scale,
                         sess.get_providers()[0])
            else:
                log.debug("Model diambil dari cache: %s", path.name)
        self._session, self.native_scale = cached
        self._loaded = True

    def unload(self) -> None:
        # Session sengaja tetap di cache global agar antar-job tidak reload.
        self._session = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    def process_tile(self, tile: np.ndarray, token: Optional[CancellationToken] = None) -> np.ndarray:
        """Jalankan inferensi pada satu tile float32 HxWx3 (0..1)."""
        if token is not None:
            token.raise_if_canceled()
        if self._session is None:
            self.load()
        assert self._session is not None

        # RRDBNet menerima NCHW RGB 0..1.
        inp = np.transpose(tile, (2, 0, 1))[None].astype(np.float32)
        # Genapkan agar dimensi habis dibagi minimal blok (aman utk semua varian)
        h, w = inp.shape[2], inp.shape[3]
        pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
        if pad_h or pad_w:
            inp = np.pad(inp, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        in_name = self._session.get_inputs()[0].name
        out_name = self._session.get_outputs()[0].name
        try:
            out = self._session.run([out_name], {in_name: inp})[0]
        except Exception as exc:  # tangkap pesan OOM khas ORT/CUDA/DML
            msg = str(exc).lower()
            if any(s in msg for s in ("out of memory", "failed to allocate", "cuda failure",
                                      "insufficient", "alloc")):
                raise OutOfMemoryError(
                    "VRAM/RAM tidak cukup untuk ukuran tile ini."
                ) from exc
            raise EngineError(f"Inferensi gagal: {exc}") from exc

        if token is not None:
            token.raise_if_canceled()
        result = np.transpose(out[0], (1, 2, 0)).astype(np.float32)
        if pad_h or pad_w:
            result = result[: result.shape[0] - pad_h * self.native_scale,
                            : result.shape[1] - pad_w * self.native_scale]
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def clear_cache() -> None:
        with _CACHE_LOCK:
            _SESSION_CACHE.clear()
