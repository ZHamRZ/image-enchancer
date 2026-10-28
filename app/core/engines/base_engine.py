"""Kontrak engine pemrosesan gambar (pola Strategy / Template Method).

Semua engine (Real-ESRGAN ONNX, GFPGAN, klasik OpenCV) mengimplementasikan
:class:`BaseEngine` sehingga pipeline tidak perlu tahu detail model. Menambah
model baru = membuat subclass baru + mendaftarkannya di ``registry``.
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.core.device import DeviceInfo
from app.core.types import CancellationToken


class BaseEngine(ABC):
    """Antarmuka seragam untuk semua engine upscale/restorasi.

    Metode inti: :meth:`process_tile` yang menerima dan mengembalikan array
    float32 rentang 0..1 berbentuk HxWx3. Pipeline bertanggung jawab atas
    tiling; engine hanya mengurus satu tile pada satu waktu.
    """

    #: skala native output terhadap input (mis. 4 untuk x4)
    native_scale: int = 1
    #: nama tampilan untuk UI
    display_name: str = "Engine"
    #: kunci unik engine/model
    key: str = "base"

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device
        self._loaded = False

    # -- siklus hidup -------------------------------------------------------- #
    @abstractmethod
    def load(self) -> None:
        """Muat bobot ke memori (lazy). Idempoten."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Bebaskan memori engine; aman dipanggil berulang kali."""
        self._loaded = False
        gc.collect()

    # -- inferensi ----------------------------------------------------------- #
    @abstractmethod
    def process_tile(self, tile: np.ndarray, token: Optional[CancellationToken] = None) -> np.ndarray:
        """Proses satu tile.

        Args:
            tile: float32 HxWx3, rentang 0..1.
            token: pembatalan opsional (diperiksa per tile besar).

        Returns:
            float32 H'xW'x3 dengan H'=H*native_scale (atau sama utk engine 1:1).
        """

    def supports_scale(self, scale: float) -> bool:
        return True

    def estimate_tile_cost_ms(self, tile_px: int) -> float:
        """Perkiraan kasar durasi proses satu tile (dipakai benchmark/ETA)."""
        base = 1500.0 if self.device.is_gpu else 6000.0
        return base * (tile_px / (512 * 512))


# --------------------------------------------------------------------------- #
# Registry sederhana (dependency injection ringan)
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[BaseEngine]] = {}


def register_engine(key: str):
    """Dekorator pendaftar kelas engine berdasarkan kunci model/engine."""

    def deco(cls: type[BaseEngine]) -> type[BaseEngine]:
        _REGISTRY[key] = cls
        return cls

    return deco


def create_engine(key: str, **kwargs) -> BaseEngine:
    """Factory: buat instance engine dari kunci.

    Raises:
        KeyError: bila kunci tidak terdaftar.
    """
    if key not in _REGISTRY:
        raise KeyError(f"Engine '{key}' tidak terdaftar. Tersedia: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def registered_keys() -> list[str]:
    return sorted(_REGISTRY)
