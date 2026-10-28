"""Pengaturan aplikasi (JSON di folder data).

Menyimpan preferensi: perangkat, ukuran tile, batas VRAM/RAM, jumlah worker,
folder output default, bahasa, tema, pola nama file, format default, dsb.
Menggunakan JSON agar mudah dibaca/diedit pengguna & ikut ter-bundle rapi
dengan PyInstaller (QSettings registry Windows tidak portable untuk laporan bug).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.types import ColorMode, Language, OutputFormat

log = logging.getLogger("iep.settings")

SETTINGS_VERSION = 1


@dataclass
class Settings:
    """Semua preferensi pengguna dengan nilai default sehat."""

    version: int = SETTINGS_VERSION
    # Perangkat & memori
    device_preference: str = "auto"          # auto | gpu | cpu
    tile_size: int = 512
    tile_overlap: int = 32
    vram_limit_mb: int = 0                   # 0 = otomatis
    ram_limit_mb: int = 0
    fp16_enabled: bool = True
    worker_count: int = 1                    # >1 hanya utk engine CPU ringan

    # Ekspor
    default_output_dir: str = ""             # kosong => Pictures/Enhanced
    output_format: str = OutputFormat.PNG.value
    jpg_quality: int = 95
    filename_pattern: str = "{name}_enhanced_{scale}x"
    overwrite_existing: bool = False

    # Tampilan
    theme: str = ColorMode.DARK.value
    language: str = Language.ID.value
    ui_scale: float = 1.0

    # Perilaku
    show_first_run_wizard: bool = True
    last_preset: str = "clear_photo"
    face_enhance_default: bool = False
    models_dir: str = ""                     # override lokasi model

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in known}
        s = cls(**clean)
        s.tile_size = max(128, min(2048, int(s.tile_size)))
        s.tile_overlap = max(0, min(256, int(s.tile_overlap)))
        s.worker_count = max(1, min(4, int(s.worker_count)))
        s.jpg_quality = max(1, min(100, int(s.jpg_quality)))
        return s


class SettingsStore:
    """Baca/tulis :class:`Settings` dari/ke satu berkas JSON atomik."""

    FILENAME = "settings.json"

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else None
        self._settings = Settings()

    @property
    def path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / self.FILENAME

    # ------------------------------------------------------------------ #
    def load(self) -> Settings:
        p = self.path
        if p and p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                self._settings = Settings.from_dict(raw)
                log.info("Pengaturan dimuat dari %s", p)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                log.warning("Pengaturan rusak (%s); memakai default", exc)
                self._settings = Settings()
        return self._settings

    def save(self, settings: Optional[Settings] = None) -> None:
        if settings is not None:
            self._settings = settings
        p = self.path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self._settings.to_json(), encoding="utf-8")
        tmp.replace(p)

    @property
    def current(self) -> Settings:
        return self._settings

    def update(self, **changes: Any) -> Settings:
        """Ubah beberapa bidang lalu simpan (dipanggil dialog Pengaturan)."""
        d = asdict(self._settings)
        d.update(changes)
        self._settings = Settings.from_dict(d)
        self.save()
        return self._settings
