"""Ekspor hasil: penamaan file, format, kualitas, metadata (EXIF/ICC), alpha.

Aturan keselamatan data:
  * Tidak pernah menimpa file asli kecuali ``overwrite=True`` DAN pengguna sudah
    mengonfirmasi lewat dialog destruktif di UI.
  * Bila target sudah ada dan overwrite mati -> tambahkan suffix ``_1``, ``_2``...
  * Disk penuh ditangani dengan pesan ramah.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.engines.classic_engine import to_u8
from app.core.image_io import to_pil
from app.core.types import EngineError, EnhanceParams, ImageData, OutputFormat

log = logging.getLogger("iep.exporter")

_EXT_FOR_FORMAT = {
    OutputFormat.PNG: ".png",
    OutputFormat.JPG: ".jpg",
    OutputFormat.WEBP: ".webp",
    OutputFormat.TIFF: ".tiff",
}


def build_output_name(stem: str, params: EnhanceParams, pattern: str) -> str:
    """Terapkan pola nama file.

    Token tersedia: ``{name}``, ``{scale}``, ``{model}``, ``{format}``.
    Contoh pola: ``{name}_enhanced_{scale}x``.
    """
    scale_txt = f"{params.scale:g}"
    model = params.model_key.split(":")[-1].replace("_", "-")
    out = pattern or "{name}_enhanced"
    try:
        return out.format(name=stem, scale=scale_txt, model=model, format="")
    except (KeyError, IndexError):
        # pola tidak dikenal dari preset lama — fallback aman
        return f"{stem}_enhanced_{scale_txt}x"


def unique_path(directory: Path, filename: str, overwrite: bool) -> Path:
    """Kembalikan path tulis-aman; tambah suffix _1.. bila perlu."""
    target = directory / filename
    if overwrite or not target.exists():
        return target
    stem, ext = target.stem, target.suffix
    n = 1
    while True:
        candidate = directory / f"{stem}_{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 1 << 40  # anggap cukup bila tak terdeteksi


class Exporter:
    """Menulis :class:`ImageData` hasil akhir ke disk sesuai preferensi."""

    def save(self, image: ImageData, out_dir: Path, base_stem: str,
             params: EnhanceParams, fmt: OutputFormat, jpg_quality: int = 95,
             pattern: str = "{name}_enhanced_{scale}x", overwrite: bool = False) -> Path:
        """Simpan gambar; kembalikan path final.

        Raises:
            EngineError: disk penuh / gagal tulis (pesan berbahasa Indonesia).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = _EXT_FOR_FORMAT[fmt]
        name = build_output_name(base_stem, params, pattern) + ext
        target = unique_path(out_dir, name, overwrite)

        arr = to_u8(np.clip(image.rgb, 0.0, 1.0))
        pil = to_pil(arr, alpha=image.alpha)

        # JPG/WEBP-TIFF tanpa alpha: konversi eksplisit agar tidak error
        if fmt in (OutputFormat.JPG,) and pil.mode == "RGBA":
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[-1])
            pil = bg

        save_kwargs: dict = {}
        if image.icc:
            save_kwargs["icc_profile"] = image.icc
        if fmt == OutputFormat.JPG:
            save_kwargs["quality"] = int(max(1, min(100, jpg_quality)))
            save_kwargs["subsampling"] = 1
            if image.exif:
                save_kwargs["exif"] = image.exif
        elif fmt == OutputFormat.PNG:
            save_kwargs["optimize"] = False   # cepat; kompresi ringan saja
            save_kwargs["compress_level"] = 6
        elif fmt == OutputFormat.WEBP:
            save_kwargs["quality"] = int(max(1, min(100, jpg_quality)))
            save_kwargs["method"] = 4
        elif fmt == OutputFormat.TIFF:
            save_kwargs["compression"] = "tiff_lzw"
            if image.exif:
                save_kwargs["exif"] = image.exif

        need = max(target.stat().st_size if target.exists() else 0,
                   pil.size[0] * pil.size[1] * len(pil.getbands()) // 2)
        if free_bytes(out_dir) < need:
            raise EngineError(
                f"Penyimpanan '{out_dir}' hampir penuh. Kosongkan disk lalu coba lagi."
            )

        try:
            pil.save(target, **save_kwargs)
        except (OSError, ValueError) as exc:
            raise EngineError(f"Gagal menyimpan file: {exc}") from exc
        log.info("Tersimpan: %s", target)
        return target


def read_back_for_preview(path: Path, max_side: Optional[int] = None) -> Image.Image:
    """Baca ulang hasil untuk panel After (dibatasi ukurannya utk hemat memori)."""
    try:
        img = Image.open(path)
        img.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise EngineError(f"Hasil tidak dapat dibaca: {path.name}") from exc
    if max_side and max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img
