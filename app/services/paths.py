"""Lokasi folder data aplikasi (pengaturan, riwayat, log, preset kustom).

Menggunakan folder ``%APPDATA%/ImageEnhancerPro`` di Windows dan
``~/.config/ImageEnhancerPro`` di Linux/macOS agar hasil PyInstallerOneFolder
tidak mengotori direktori instalasi.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "ImageEnhancerPro"


def is_frozen() -> bool:
    """True bila berjalan sebagai exe hasil PyInstaller."""
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """Direktori sumber aplikasi (atau isi bundle exe)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Folder data pengguna; dibuat bila belum ada."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", str(Path.home())))
        d = base / APP_NAME
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Folder hasil sementara (resume setelah crash) + thumbnail."""
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_output_dir() -> Path:
    d = Path.home() / "Pictures" / "Enhanced"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.home()
    return d
