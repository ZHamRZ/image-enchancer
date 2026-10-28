"""IO gambar: load (dengan koreksi EXIF/ICC/alpha) dan validasi.

Fungsi :func:`load_image` adalah satu-satunya tempat aplikasi menyentuh disk
untuk membaca piksel; hasilnya :class:`ImageData` siap dipakai pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.types import EngineError, ImageData

log = logging.getLogger("iep.imageio")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_FILE_BYTES = 200 * 1024 * 1024          # 200 MB
MAX_DIMENSION = 30_000                       # batas lebar/tinggi input


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def validate_input(path: Path) -> None:
    """Validasi awal; mengangkat :class:`EngineError` dgn pesan Indonesia."""
    if not path.exists():
        raise EngineError(f"File tidak ditemukan: {path.name}")
    if not is_supported(path):
        raise EngineError(
            f"Format '{path.suffix}' tidak didukung. "
            f"Gunakan: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    size = path.stat().st_size
    if size == 0:
        raise EngineError(f"File kosong/rusak: {path.name}")
    if size > MAX_FILE_BYTES:
        raise EngineError(
            f"Ukuran file {size / 1e6:.0f} MB melebihi batas {MAX_FILE_BYTES // 1e6:.0f} MB."
        )


def load_image(path: Path, max_side_hint: Optional[int] = None) -> ImageData:
    """Baca gambar menjadi :class:`ImageData`.

    Langkah:
      1. Validasi dasar.
      2. Buka dengan Pillow; ``exif_transpose`` mengoreksi orientasi EXIF.
      3. Pisahkan alpha (bila ada) -> disalin ke kanal tersendiri.
      4. RGB -> float32 0..1. Simpan ICC & EXIF mentah untuk ekspor nanti.
    """
    validate_input(path)
    try:
        img = Image.open(path)
        img.load()
    except UnidentifiedImageError as exc:
        raise EngineError(f"File bukan gambar yang valid atau rusak: {path.name}") from exc
    except MemoryError as exc:
        raise EngineError("Gambar terlalu besar untuk dimuat ke memori.") from exc

    with Image.open(path) as src:
        icc = src.info.get("icc_profile")
        exif = src.info.get("exif")
    orientation = 1
    try:
        exif_dict = img.getexif() or {}
        orientation = int(exif_dict.get(274, 1))
    except Exception:  # pragma: no cover - tidak semua format punya EXIF
        pass

    # Koreksi orientasi EXIF secara fisik agar pipeline bekerja pada citra "benar".
    img = ImageOps.exif_transpose(img)

    if max(img.size) > MAX_DIMENSION:
        raise EngineError(
            f"Dimensi gambar {img.width}x{img.height} melebihi batas {MAX_DIMENSION} px."
        )
    if max_side_hint and max(img.size) > max_side_hint:
        ratio = max_side_hint / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
                         Image.LANCZOS)
        log.info("Gambar dikecilkan saat load sesuai batas: %sx%s", img.width, img.height)

    alpha: Optional[np.ndarray] = None
    grayscale = False
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        a = np.asarray(img.split()[-1], dtype=np.float32) / 255.0
        if float(a.min()) < 1.0:                 # hanya dianggap alpha bila benar-benar transparan
            alpha = a
        rgb_img = img.convert("RGB")
    elif img.mode == "L":
        grayscale = True
        rgb_img = img.convert("RGB")
    else:
        rgb_img = img.convert("RGB")

    rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
    return ImageData(
        rgb=rgb,
        alpha=alpha,
        icc=icc,
        exif=exif,
        orientation=orientation,
        grayscale_input=grayscale,
        width=rgb.shape[1],
        height=rgb.shape[0],
    )


def to_pil(arr_u8: np.ndarray, alpha: Optional[np.ndarray] = None) -> Image.Image:
    """Array uint8 HxWx3 (+ opsional alpha HxW 0..1) -> PIL RGBA/RGB."""
    if alpha is not None:
        a8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(np.dstack([arr_u8, a8]), mode="RGBA")
    return Image.fromarray(arr_u8, mode="RGB")
