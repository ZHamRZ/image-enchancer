"""Deteksi perangkat (CPU / NVIDIA CUDA / DirectML) untuk ONNX Runtime.

Hasil deteksi bersifat *cached* per proses: cukup sekali dijalankan,
lalu dipakai seluruh engine. Tidak ada dependensi GUI di sini.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.core.types import DeviceKind

log = logging.getLogger("iep.device")


@dataclass
class DeviceInfo:
    """Ringkasan kemampuan perangkat keras/akselerasi."""

    kind: DeviceKind = DeviceKind.CPU
    name: str = "CPU"
    providers_available: list[str] = field(default_factory=list)
    vram_bytes: int = 0                 # 0 = tidak terdeteksi
    ram_bytes: int = 0
    cpu_count: int = 1
    fp16_supported: bool = False
    details: dict[str, str] = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        return self.kind != DeviceKind.CPU

    @property
    def label(self) -> str:
        if self.kind == DeviceKind.CPU:
            return f"CPU ({self.cpu_count} thread)"
        return f"{self.kind.value.upper()}: {self.name}"


def _total_ram_bytes() -> int:
    try:
        import os

        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(pages)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return int(stat.ullTotalPhys)
    except Exception:  # pragma: no cover - sangat tergantung platform
        return 0


def _nvidia_vram_bytes() -> int:
    """Coba membaca VRAM lewat NVML (opsional)."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        pynvml.nvmlShutdown()
        return int(info.total)
    except Exception:
        return 0


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover
        log.warning("onnxruntime tidak tersedia: %s", exc)
        return []


@lru_cache(maxsize=1)
def detect_device(prefer: str = "auto") -> DeviceInfo:
    """Deteksi perangkat akselerasi terbaik.

    Args:
        prefer: ``"auto"`` | ``"gpu"`` | ``"cpu"``.

    Urutan prioritas: CUDA (NVIDIA) -> DirectML (AMD/Intel Windows) -> CPU.
    """
    import multiprocessing

    info = DeviceInfo(
        providers_available=_onnx_providers(),
        ram_bytes=_total_ram_bytes(),
        cpu_count=multiprocessing.cpu_count(),
    )
    info.details = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "providers": ", ".join(info.providers_available) or "-",
    }

    if prefer == "cpu":
        info.kind = DeviceKind.CPU
        info.name = "CPU (dipaksa)"
        return info

    if "CUDAExecutionProvider" in info.providers_available:
        vram = _nvidia_vram_bytes()
        gpu_name = "NVIDIA GPU"
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            n = pynvml.nvmlDeviceGetName(h)
            gpu_name = n.decode() if isinstance(n, bytes) else str(n)
            pynvml.nvmlShutdown()
        except Exception:
            pass
        info.kind = DeviceKind.CUDA
        info.name = gpu_name
        info.vram_bytes = vram
        info.fp16_supported = True
        return info

    if prefer != "gpu" and "DmlExecutionProvider" in info.providers_available:
        info.kind = DeviceKind.DIRECTML
        info.name = platform.processor() or "DirectML GPU"
        info.fp16_supported = True
        return info

    info.kind = DeviceKind.CPU
    info.name = f"CPU {info.cpu_count}-core"
    return info


def build_session_options(info: DeviceInfo, intra_threads: int = 0) -> Any:
    """Buat ``onnxruntime.SessionOptions`` + daftar provider sesuai perangkat."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads <= 0:
        intra_threads = max(1, info.cpu_count // 2)
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = 1

    if info.kind == DeviceKind.CUDA:
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    elif info.kind == DeviceKind.DIRECTML:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return so, providers


def estimate_memory_bytes(width: int, height: int, scale: float, tile: int) -> int:
    """Estimasi kasar kebutuhan RAM+VRAM (byte) untuk satu proses upscale.

    Perhitungan sederhana: buffer input uint8 + output float32 + cache tile.
    """
    out_w, out_h = width * scale, height * scale
    inp = width * height * 3 * 4          # RGB float32 sumber
    out = int(out_w * out_h * 3 * 4)       # hasil float32
    tiles = int(tile * tile * 3 * 4 * 8)   # ~8 buffer tile (in/out/blend)
    return inp + out + tiles


def memory_risk(width: int, height: int, scale: float, tile: int, info: DeviceInfo) -> tuple[bool, str]:
    """Kembalikan (berisiko, pesan peringatan berbahasa Indonesia)."""
    need = estimate_memory_bytes(width, height, scale, tile)
    avail = info.ram_bytes
    if avail and need > avail * 0.75:
        return True, (
            f"Estimasi pemakaian memori {need / 1e9:.1f} GB mendekati batas RAM "
            f"{avail / 1e9:.1f} GB. Saran: kecilkan ukuran tile atau turunkan skala."
        )
    return False, ""
