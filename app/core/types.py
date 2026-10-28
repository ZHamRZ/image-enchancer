"""Tipe data bersama untuk Image Enhancer Pro.

Modul ini hanya berisi struktur data (dataclass / enum) dan tidak mengimpor
GUI maupun engine AI, sehingga aman dipakai dari semua lapisan (core, services, ui).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Enumerasi
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    """Status satu item dalam antrean."""

    WAITING = "waiting"        # Menunggu
    RUNNING = "running"        # Berjalan
    PAUSED = "paused"          # Dijeda
    DONE = "done"              # Selesai
    FAILED = "failed"          # Gagal
    CANCELED = "canceled"      # Dibatalkan

    @property
    def label_id(self) -> str:
        """Label ramah berbahasa Indonesia untuk UI."""
        return {
            JobStatus.WAITING: "Menunggu",
            JobStatus.RUNNING: "Berjalan",
            JobStatus.PAUSED: "Dijeda",
            JobStatus.DONE: "Selesai",
            JobStatus.FAILED: "Gagal",
            JobStatus.CANCELED: "Dibatalkan",
        }[self]


class DeviceKind(str, Enum):
    """Jenis perangkat yang dipakai inferensi."""

    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"


class OutputFormat(str, Enum):
    """Format file hasil ekspor."""

    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"
    TIFF = "tiff"


class ColorMode(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class Language(str, Enum):
    ID = "id"
    EN = "en"


# --------------------------------------------------------------------------- #
# Pembatalan (cancellation token)
# --------------------------------------------------------------------------- #
class CancellationToken:
    """Token sederhana agar pipeline bisa dibatalkan di antara tile.

    Dipakai lintas-thread tanpa lock berat: hanya flag ``threading.Event``.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def canceled(self) -> bool:
        return self._event.is_set()

    def raise_if_canceled(self) -> None:
        if self._event.is_set():
            raise JobCanceled("Proses dibatalkan pengguna")


class JobCanceled(Exception):
    """Diangkat bila pekerjaan dibatalkan lewat CancellationToken."""


class EngineError(Exception):
    """Kesalahan level engine (model hilang, OOM, dsb.) dengan pesan ramah."""


class OutOfMemoryError(EngineError):
    """VRAM/RAM habis saat inferensi; pemicu pengecilan tile otomatis."""

    def __init__(self, message: str = "Memori habis (OOM)") -> None:
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Parameter pemrosesan
# --------------------------------------------------------------------------- #
@dataclass
class EnhanceParams:
    """Seluruh parameter pipeline enhancement dalam satu objek."""

    # Upscale
    model_key: str = "general_x4"          # kunci model ESRGAN (lihat model_manager)
    scale: float = 2.0                     # skala akhir (2/3/4/custom)
    custom_target_w: int = 0               # jika >0, resize akhir ke lebar ini
    custom_target_h: int = 0               # jika >0, resize akhir ke tinggi ini

    # Pra-proses klasik
    denoise: int = 15                      # 0-100
    deblur: int = 20                       # 0-100 (unsharp/sharpen pasca)
    jpeg_fix: bool = False                 # hapus artefak JPEG ringan
    detail_balance: int = 50               # 0=Smooth .. 100=Preserve detail

    # Wajah
    face_enhance: bool = False
    face_strength: int = 60                # 0-100 blending hasil restorasi wajah

    # Warna
    brightness: int = 0                    # -100..100
    contrast: int = 0                      # -100..100
    saturation: int = 0                    # -100..100
    auto_levels: bool = False
    clahe: bool = False
    clahe_strength: int = 40               # 0-100 -> clip limit

    # Tiling & memori
    tile_size: int = 512
    tile_overlap: int = 32
    fp16: bool = True                      # half precision bila didukung

    def clamp(self) -> "EnhanceParams":
        """Batasi semua nilai ke rentang sah."""
        self.denoise = _clamp(self.denoise, 0, 100)
        self.deblur = _clamp(self.deblur, 0, 100)
        self.face_strength = _clamp(self.face_strength, 0, 100)
        self.detail_balance = _clamp(self.detail_balance, 0, 100)
        self.brightness = _clamp(self.brightness, -100, 100)
        self.contrast = _clamp(self.contrast, -100, 100)
        self.saturation = _clamp(self.saturation, -100, 100)
        self.clahe_strength = _clamp(self.clahe_strength, 0, 100)
        self.scale = _clamp(self.scale, 1.0, 8.0)
        self.tile_size = max(64, (int(self.tile_size) // 8) * 8)
        self.tile_overlap = _clamp(self.tile_overlap, 0, self.tile_size // 2)
        return self

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhanceParams":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known}).clamp()


def _clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


# --------------------------------------------------------------------------- #
# Gambar dalam memori
# --------------------------------------------------------------------------- #
@dataclass
class ImageData:
    """Muatan gambar selama pipeline berjalan.

    Atribut:
        rgb: float32 array HxWx3 rentang 0..1 (saluran kerja utama).
        alpha: float32 array HxW atau None bila tidak transparan.
        icc: bytes profil warna ICC atau None.
        exif: bytes EXIF mentah (untuk dipertahankan saat ekspor).
        orientation: orientasi EXIF asli (sudah dikoreksi saat load).
        grayscale_input: True bila sumbernya grayscale.
    """

    rgb: Any                                # np.ndarray HxWx3 float32 0..1
    alpha: Optional[Any] = None             # np.ndarray HxW float32 0..1 | None
    icc: Optional[bytes] = None
    exif: Optional[bytes] = None
    orientation: int = 1
    grayscale_input: bool = False
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.width == 0 and self.rgb is not None:
            self.height, self.width = self.rgb.shape[:2]


# --------------------------------------------------------------------------- #
# Item antrean
# --------------------------------------------------------------------------- #
_progress_cb = Callable[[float, str], None]


@dataclass
class JobItem:
    """Satu pekerjaan enhancement pada satu file."""

    id: str
    input_path: Path
    params: EnhanceParams
    output_dir: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.PNG
    jpg_quality: int = 95
    filename_pattern: str = "{name}_enhanced_{scale}x"
    overwrite: bool = False                 # False => tambah suffix otomatis
    status: JobStatus = JobStatus.WAITING
    progress: float = 0.0                   # 0..1 per gambar
    message: str = ""
    error: str = ""
    result_path: Optional[Path] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    token: CancellationToken = field(default_factory=CancellationToken)

    @property
    def elapsed(self) -> float:
        if self.started_at == 0.0:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at


@dataclass
class ProgressReport:
    """Laporan progres dari worker ke UI."""

    job_id: str
    status: JobStatus
    job_progress: float
    stage: str
    tiles_done: int = 0
    tiles_total: int = 0
    overall: float = 0.0
    eta_seconds: float = 0.0
