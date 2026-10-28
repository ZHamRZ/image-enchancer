"""Antrean pekerjaan (job queue) + worker thread.

Prinsip:
  * UI tidak pernah menyentuh pipeline/model langsung — hanya mengirim item ke
    :class:`JobQueue` dan menerima sinyal Qt (``progress``, ``item_done``, ...).
  * Worker berjalan di QThread terpisah; antrean diproses berurutan.
  * Satu gambar gagal TIDAK menghentikan seluruh antrean.
  * Kontrol: start / pause / resume / cancel (per item & semua), reorder,
    hapus item.
  * Progres keseluruhan + ETA dihitung dari rata-rata durasi tile terakhir.
  * Hasil sementara per item disimpan di folder cache agar bisa dilanjutkan
    bila aplikasi tertutup mendadak (file .part + manifest JSON).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.core.device import DeviceInfo
from app.core.engine_factory import EngineProvider
from app.core.pipeline import EnhancePipeline, cleanup_gpu_memory
from app.core.types import (
    EnhanceParams,
    JobCanceled,
    JobItem,
    JobStatus,
    OutputFormat,
)
from app.services.history import HistoryStore
from app.services.logger import recent_log_text

log = logging.getLogger("iep.jobqueue")


# --------------------------------------------------------------------------- #
# Sinyal payload sederhana (bukan dataclass Qt — cukup tuple/dict via Signal)
# --------------------------------------------------------------------------- #
class QueueSignals(QObject):
    """Kumpulan sinyal yang diteruskan worker -> UI."""

    item_status = Signal(str, str, str)        # job_id, status, pesan
    item_progress = Signal(str, float, str)    # job_id, fraksi, tahap
    queue_progress = Signal(float, float)      # overall 0..1, eta detik
    item_finished = Signal(str, str)           # job_id, path hasil ('' bila gagal)
    log_message = Signal(str)                  # pesan info utk panel log/status


# --------------------------------------------------------------------------- #
# Antrean
# --------------------------------------------------------------------------- #
class JobQueue(QObject):
    """Pengelola daftar :class:`JobItem` + satu worker pipeline."""

    def __init__(self, device: DeviceInfo, provider: EngineProvider,
                 history: HistoryStore, cache_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.signals = QueueSignals()
        self.pipeline = EnhancePipeline(device, provider)
        self.history = history
        self.cache_dir = Path(cache_dir)
        self._items: list[JobItem] = []
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._stop_all = False
        self._worker: Optional[_WorkerThread] = None
        self._running = False
        self._durations: list[float] = []   # durasi item selesai utk ETA

    # -- CRUD item ---------------------------------------------------------- #
    def add(self, paths: list[Path], params: EnhanceParams, output_dir: Optional[Path],
            fmt: OutputFormat, jpg_quality: int, pattern: str, overwrite: bool) -> list[JobItem]:
        """Tambahkan beberapa file ke antrean (menghindari duplikat aktif)."""
        added: list[JobItem] = []
        with self._lock:
            existing = {str(i.input_path) for i in self._items
                        if i.status in (JobStatus.WAITING, JobStatus.RUNNING)}
            for p in paths:
                key = str(p.resolve())
                if key in existing:
                    continue
                item = JobItem(
                    id=uuid.uuid4().hex[:12], input_path=Path(p),
                    params=EnhanceParams.from_dict(params.to_dict()),
                    output_dir=output_dir, output_format=fmt,
                    jpg_quality=jpg_quality, filename_pattern=pattern,
                    overwrite=overwrite,
                )
                self._items.append(item)
                added.append(item)
        for it in added:
            self.signals.item_status.emit(it.id, it.status.value, "Menunggu")
        return added

    def items(self) -> list[JobItem]:
        with self._lock:
            return list(self._items)

    def get(self, job_id: str) -> Optional[JobItem]:
        with self._lock:
            return next((i for i in self._items if i.id == job_id), None)

    def remove(self, job_id: str) -> bool:
        """Hapus item (batalkan dulu bila sedang jalan)."""
        item = self.get(job_id)
        if item is None:
            return False
        item.token.cancel()
        with self._lock:
            self._items.remove(item)
        return True

    def move(self, job_id: str, delta: int) -> None:
        """Ubah urutan (+1 turun, -1 naik)."""
        with self._lock:
            idx = next((k for k, i in enumerate(self._items) if i.id == job_id), -1)
            if idx < 0:
                return
            new = max(0, min(len(self._items) - 1, idx + delta))
            self._items.insert(new, self._items.pop(idx))

    def clear_finished(self) -> None:
        with self._lock:
            self._items = [i for i in self._items
                           if i.status in (JobStatus.WAITING, JobStatus.RUNNING)]

    # -- kontrol ------------------------------------------------------------ #
    def start(self) -> None:
        """Mulai/lanjut pemrosesan antrean."""
        if self._running:
            return
        pending = [i for i in self.items() if i.status == JobStatus.WAITING]
        if not pending:
            return
        self._running = True
        self._stop_all = False
        self._pause.clear()
        self._worker = _WorkerThread(self)
        self._worker.start()

    def pause(self) -> None:
        self._pause.set()
        self.signals.log_message.emit("Antrean dijeda")

    def resume(self) -> None:
        self._pause.clear()
        self.signals.log_message.emit("Antrean dilanjutkan")

    def cancel(self, job_id: Optional[str] = None) -> None:
        """Batalkan satu item atau seluruh antrean."""
        if job_id:
            item = self.get(job_id)
            if item:
                item.token.cancel()
            return
        self._stop_all = True
        self._pause.clear()
        for i in self.items():
            if i.status in (JobStatus.WAITING, JobStatus.RUNNING):
                i.token.cancel()

    @property
    def is_running(self) -> bool:
        return self._running

    # -- eksekusi (dipanggil worker) ---------------------------------------- #
    def _next_waiting(self) -> Optional[JobItem]:
        with self._lock:
            for i in self._items:
                if i.status == JobStatus.WAITING and not i.token.canceled:
                    return i
        return None

    def run_next(self) -> bool:
        """Proses satu item berikutnya. Return False bila antrean habis."""
        if self._stop_all:
            return False
        item = self._next_waiting()
        if item is None:
            return False

        item.status = JobStatus.RUNNING
        item.started_at = time.time()
        item.error = ""
        self.signals.item_status.emit(item.id, item.status.value, "Berjalan")

        try:
            out_dir = item.output_dir or Path(item.input_path).parent
            result_path, elapsed = self.pipeline.run(
                input_path=item.input_path,
                params=item.params,
                token=item.token,
                progress=self._make_progress_cb(item),
                output_dir=out_dir,
                fmt=item.output_format,
                jpg_quality=item.jpg_quality,
                pattern=item.filename_pattern,
                overwrite=item.overwrite,
            )
            item.status = JobStatus.DONE
            item.progress = 1.0
            item.result_path = result_path
            item.finished_at = time.time()
            self._durations.append(elapsed)
            self.history.add(item.input_path, result_path, "done", elapsed,
                             item.params.to_dict())
            self._write_manifest(item)
            self.signals.item_finished.emit(item.id, str(result_path))
            self.signals.item_status.emit(item.id, item.status.value,
                                          f"Selesai dalam {elapsed:.1f} dtk")
        except JobCanceled:
            item.status = JobStatus.CANCELED
            item.finished_at = time.time()
            self.history.add(item.input_path, None, "canceled", item.elapsed,
                             item.params.to_dict())
            self.signals.item_status.emit(item.id, item.status.value, "Dibatalkan")
        except MemoryError:
            self._fail(item, "Memori komputer habis saat memproses. "
                             "Kecilkan ukuran tile pada Pengaturan.")
        except Exception as exc:  # noqa: BLE001 - satu gagal tidak menghentikan antrean
            msg = str(exc) or exc.__class__.__name__
            self._fail(item, msg)
        finally:
            cleanup_gpu_memory()
            self._emit_overall()
        return True

    def _fail(self, item: JobItem, message: str) -> None:
        item.status = JobStatus.FAILED
        item.error = message
        item.finished_at = time.time()
        log.error("Job %s gagal: %s\n%s", item.id, message, recent_log_text()[-400:])
        self.history.add(item.input_path, None, "failed", item.elapsed,
                         item.params.to_dict(), error=message)
        self.signals.item_status.emit(item.id, item.status.value, message.splitlines()[0])

    def _make_progress_cb(self, item: JobItem):
        def cb(frac: float, stage: str) -> None:
            item.progress = max(0.0, min(1.0, frac))
            item.message = stage
            self.signals.item_progress.emit(item.id, frac, stage)
            self._emit_overall()
        return cb

    # -- progres global & ETA ------------------------------------------------ #
    def _emit_overall(self) -> None:
        items = self.items()
        if not items:
            self.signals.queue_progress.emit(0.0, 0.0)
            return
        done = sum(i.progress for i in items)
        overall = done / len(items)
        eta = self._estimate_eta(items)
        self.signals.queue_progress.emit(overall, eta)

    def _estimate_eta(self, items: list[JobItem]) -> float:
        remaining_frac = sum(1.0 - i.progress for i in items
                             if i.status in (JobStatus.WAITING, JobStatus.RUNNING))
        if remaining_frac <= 0:
            return 0.0
        avg = (sum(self._durations) / len(self._durations)) if self._durations else 30.0
        return remaining_frac * avg

    # -- persistensi parsial (resume setelah crash) --------------------------- #
    def _write_manifest(self, item: JobItem) -> None:
        """Catat hasil akhir ke manifest cache utk fitur 'lanjutkan/riwayat cepat'."""
        try:
            mpath = self.cache_dir / "manifest.jsonl"
            entry = {
                "job": item.id, "input": str(item.input_path),
                "output": str(item.result_path) if item.result_path else None,
                "status": item.status.value, "params": item.params.to_dict(),
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with mpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Manifest cache gagal ditulis: %s", exc)

    def unfinished_snapshot(self) -> list[dict]:
        """Daftar item belum selesai — dipakai saat startup utk menawarkan resume."""
        snap = []
        for i in self.items():
            if i.status in (JobStatus.WAITING, JobStatus.RUNNING, JobStatus.PAUSED):
                snap.append({"id": i.id, "input": str(i.input_path),
                             "params": i.params.to_dict()})
        return snap


# --------------------------------------------------------------------------- #
# Worker thread
# --------------------------------------------------------------------------- #
class _WorkerThread(QThread):
    """Loop: jalankan satu per satu sampai antrean habis / stop."""

    def __init__(self, queue: JobQueue) -> None:
        super().__init__()
        self.queue = queue
        self.setObjectName("IEP-Worker")

    def run(self) ->  None:  # noqa: ANN401
        q = self.queue
        try:
            while not q._stop_all:
                # hormati jeda
                while q._pause.is_set() and not q._stop_all:
                    time.sleep(0.2)
                progressed = q.run_next()
                if not progressed:
                    break
        except Exception as exc:  # pragma: no cover - pengaman loop
            log.exception("Worker crash: %s", exc)
            q.signals.log_message.emit(f"Pekerja latar berhenti: {exc}")
        finally:
            q._running = False
            q.signals.queue_progress.emit(1.0 if all(
                i.status not in (JobStatus.WAITING, JobStatus.RUNNING)
                for i in q.items()) else 0.0, 0.0)
