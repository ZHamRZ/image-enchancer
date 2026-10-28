"""Logging terpusat: file berputar (rotating) + konsol + buffer memori.

Buffer memori 500 baris terakhir dipakai tombol "Salin log" di UI untuk
laporan bug, tanpa pengguna harus membuka folder.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILENAME = "image_enhancer.log"
MAX_BYTES = 2 * 1024 * 1024   # 2 MB per berkas
BACKUP_COUNT = 3
FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configured = False


class MemoryLogHandler(logging.Handler):
    """Simpan N pesan terakhir di deque agar bisa disalin ke clipboard."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        from collections import deque

        self.buffer: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
        except Exception:  # pragma: no cover - handler tidak boleh mengangkat error
            self.handleError(record)

    def text(self) -> str:
        return "\n".join(self.buffer)


_memory_handler: MemoryLogHandler | None = None


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> None:
    """Konfigurasi root logger sekali saja (idempoten)."""
    global _configured, _memory_handler
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        fh = RotatingFileHandler(log_dir / LOG_FILENAME, maxBytes=MAX_BYTES,
                                 backupCount=BACKUP_COUNT, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _memory_handler = MemoryLogHandler()
    _memory_handler.setFormatter(formatter)
    root.addHandler(_memory_handler)

    # Redam kebisingan library pihak ketiga
    for noisy in ("onnxruntime", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_log_file(log_dir: Path) -> Path:
    return log_dir / LOG_FILENAME


def recent_log_text() -> str:
    """Isi buffer log terbaru untuk 'Salin log'."""
    if _memory_handler is None:
        setup_logging()
    assert _memory_handler is not None
    return _memory_handler.text()
