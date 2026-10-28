"""Preset satu-klik + penyimpanan preset buatan pengguna (JSON).

Preset bawaan (7) dan preset kustom disimpan terpisah: preset bawaan tidak
bisa dihapus, preset kustom tersimpan di ``presets.json`` pada folder data.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from app.core.types import EnhanceParams

log = logging.getLogger("iep.presets")


@dataclass
class Preset:
    """Kumpulan parameter enhancement dengan nama & ikon."""

    key: str
    name: str
    description: str = ""
    builtin: bool = False
    params: EnhanceParams = field(default_factory=EnhanceParams)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "builtin": self.builtin,
            "params": self.params.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Preset":
        return cls(
            key=str(data.get("key", "custom")),
            name=str(data.get("name", "Preset")),
            description=str(data.get("description", "")),
            builtin=bool(data.get("builtin", False)),
            params=EnhanceParams.from_dict(data.get("params", {})),
        )


# --------------------------------------------------------------------------- #
# Preset bawaan — nilai dipilih sebagai titik awal praktis (lihat README).
# --------------------------------------------------------------------------- #
def _p(**kw) -> EnhanceParams:
    base = EnhanceParams()
    for k, v in kw.items():
        setattr(base, k, v)
    return base.clamp()


BUILTIN_PRESETS: list[Preset] = [
    Preset(
        key="clear_photo", name="Foto Jernih", builtin=True,
        description="Penajaman & denoise seimbang untuk foto sehari-hari.",
        params=_p(model_key="general_x4", scale=2.0, denoise=15, deblur=35,
                  detail_balance=65),
    ),
    Preset(
        key="old_photo", name="Foto Lama / Restorasi", builtin=True,
        description="Untuk foto lama: noise tinggi, artefak, warna pudar.",
        params=_p(model_key="oldphoto_x4", scale=3.0, denoise=55, jpeg_fix=True,
                  deblur=30, detail_balance=45, auto_levels=True, saturation=15,
                  clahe=True, clahe_strength=35, face_enhance=True, face_strength=55),
    ),
    Preset(
        key="anime", name="Anime / Ilustrasi", builtin=True,
        description="Garis bersih tanpa artefak untuk gambar 2D.",
        params=_p(model_key="anime_x4", scale=2.0, denoise=8, deblur=25,
                  detail_balance=80),
    ),
    Preset(
        key="document", name="Dokumen / Scan", builtin=True,
        description="Teks & garis tajam, latar putih rata.",
        params=_p(model_key="document_x2", scale=2.0, denoise=25, deblur=55,
                  detail_balance=90, auto_levels=True, clahe=True, clahe_strength=25),
    ),
    Preset(
        key="portrait", name="Potret Wajah", builtin=True,
        description="Kulit halus namun wajah tetap natural.",
        params=_p(model_key="general_x4", scale=2.0, denoise=20, deblur=25,
                  face_enhance=True, face_strength=65, detail_balance=55,
                  saturation=5),
    ),
    Preset(
        key="max_quality", name="Maksimal Kualitas (lambat)", builtin=True,
        description="Kualitas tertinggi; waktu proses paling lama.",
        params=_p(model_key="general_x4", scale=4.0, denoise=25, deblur=40,
                  jpeg_fix=True, detail_balance=75, face_enhance=True,
                  face_strength=60, tile_size=384, tile_overlap=48, fp16=True),
    ),
    Preset(
        key="draft_fast", name="Cepat (draft)", builtin=True,
        description="Pratinjau cepat dengan engine klasik tanpa AI.",
        params=_p(model_key="classic_x2", scale=2.0, denoise=10, deblur=20,
                  detail_balance=50, tile_size=768, face_enhance=False),
    ),
]


class PresetManager:
    """Gabungan preset bawaan + kustom; muft/simpan ke JSON."""

    def __init__(self, store_path: Optional[Path] = None) -> None:
        self.store_path = store_path
        self._custom: dict[str, Preset] = {}
        if store_path and store_path.exists():
            self.load()

    # -- kueri -------------------------------------------------------------- #
    def all(self) -> list[Preset]:
        return BUILTIN_PRESETS + list(self._custom.values())

    def get(self, key: str) -> Optional[Preset]:
        for p in BUILTIN_PRESETS:
            if p.key == key:
                return p
        return self._custom.get(key)

    def names(self) -> list[str]:
        return [p.name for p in self.all()]

    # -- mutasi ------------------------------------------------------------- #
    def save_custom(self, preset: Preset) -> None:
        """Simpan/perbarui preset kustom (nama harus unik terhadap bawaan)."""
        taken = {p.name for p in BUILTIN_PRESETS} | {
            p.name for k, p in self._custom.items() if k != preset.key
        }
        if preset.name in taken:
            raise ValueError(f"Nama preset '{preset.name}' sudah dipakai.")
        preset.builtin = False
        if not preset.key or preset.key.startswith("builtin"):
            preset.key = f"custom_{_slug(preset.name)}"
        self._custom[preset.key] = preset
        self._flush()

    def delete_custom(self, key: str) -> bool:
        if key in self._custom:
            del self._custom[key]
            self._flush()
            return True
        return False

    def duplicate_as_custom(self, key: str, new_name: str) -> Preset:
        src = self.get(key)
        if src is None:
            raise KeyError(f"Preset '{key}' tidak ditemukan")
        clone = Preset(key=f"custom_{_slug(new_name)}", name=new_name,
                       description=src.description, builtin=False,
                       params=replace(src.params))
        self.save_custom(clone)
        return clone

    # -- persistensi --------------------------------------------------------- #
    def load(self) -> None:
        assert self.store_path is not None
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._custom = {
                d["key"]: Preset.from_dict(d) for d in data.get("presets", [])
            }
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Preset kustom gagal dimuat (%s); mulai kosong", exc)
            self._custom = {}

    def _flush(self) -> None:
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"presets": [p.to_dict() for p in self._custom.values()]}
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.store_path)


def _slug(text: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "preset"
