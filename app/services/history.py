"""Riwayat proses tersimpan di SQLite (folder data aplikasi).

Setiap gambar yang selesai/gagal dicatat: file asal, hasil, parameter JSON,
durasi, timestamp. UI memakai ini untuk daftar Riwayat + "proses ulang dengan
pengaturan sama" + "buka folder".
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("iep.history")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    input_path TEXT NOT NULL,
    output_path TEXT,
    status TEXT NOT NULL,
    duration_s REAL,
    params_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC);
"""


@dataclass
class HistoryEntry:
    id: int
    created_at: str
    input_path: str
    output_path: Optional[str]
    status: str            # done | failed | canceled
    duration_s: float
    params: dict
    error: str = ""

    def params_dict(self) -> dict:
        return self.params or {}


class HistoryStore:
    """Akses aman- thread ke tabel riwayat."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def add(self, input_path: Path, output_path: Optional[Path], status: str,
            duration_s: float, params: dict, error: str = "") -> int:
        """Catat satu kejadian; kembalikan id baris."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO history (created_at, input_path, output_path, status,"
                " duration_s, params_json, error) VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), str(input_path),
                 str(output_path) if output_path else None, status,
                 float(duration_s), json.dumps(params, ensure_ascii=False), error),
            )
            return int(cur.lastrowid or 0)

    def recent(self, limit: int = 200) -> list[HistoryEntry]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out: list[HistoryEntry] = []
        for r in rows:
            try:
                params = json.loads(r["params_json"] or "{}")
            except json.JSONDecodeError:
                params = {}
            out.append(HistoryEntry(
                id=int(r["id"]), created_at=r["created_at"],
                input_path=r["input_path"], output_path=r["output_path"],
                status=r["status"], duration_s=float(r["duration_s"] or 0),
                params=params, error=r["error"] or "",
            ))
        return out

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM history")

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) c FROM history").fetchone()
        return int(row["c"]) if row else 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # util untuk dialog "proses ulang" ------------------------------------- #
    @staticmethod
    def entry_to_params(entry: HistoryEntry) -> dict:
        """Salin parameter riwayat agar bisa dipakai ulang."""
        return dict(asdict(entry.params) if hasattr(entry.params, "__dataclass_fields__")
                    else entry.params)
