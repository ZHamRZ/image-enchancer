"""Factory engine: memilih dan membuat engine konkret untuk satu kunci model.

Memisahkan keputusan "model X memakai engine Y" dari pipeline, sehingga
pipeline hanya perlu memanggil :func:`get_engine`. Engine yang dibuat di-cache
per proses (dependency injection sederhana lewat :class:`EngineProvider`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.device import DeviceInfo
from app.core.engines.base_engine import BaseEngine
from app.core.engines.classic_engine import ClassicIdentityEngine, ClassicUpscaleEngine
from app.core.engines.esrgan_engine import EsrganOnnxEngine
from app.core.model_manager import MODEL_REGISTRY, ModelManager, ModelSpec
from app.core.types import EngineError

log = logging.getLogger("iep.engine_factory")


class EngineProvider:
    """Pencetak & pengcache engine berdasarkan kunci model + perangkat."""

    def __init__(self, device: DeviceInfo, models_dir: Optional[Path] = None) -> None:
        self.device = device
        self.manager = ModelManager(models_dir)
        self._cache: dict[str, BaseEngine] = {}

    # ------------------------------------------------------------------ #
    def get(self, model_key: str, fp16: bool = True) -> BaseEngine:
        """Kembalikan engine siap pakai untuk ``model_key``.

        Bila berkas model ONNX tidak ditemukan, otomatis memakai fallback
        klasik dengan pesan peringatan pada log (bukan error keras).
        """
        cache_key = f"{model_key}|{fp16}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        engine = self._build(model_key, fp16)
        self._cache[cache_key] = engine
        return engine

    def _spec_for_key(self, model_key: str) -> Optional[ModelSpec]:
        """Cari spec registri ATAU model impor pengguna utk sebuah kunci."""
        for s in MODEL_REGISTRY:
            if s.key == model_key:
                return s
        if model_key.startswith("custom:"):
            stem = model_key.split(":", 1)[1]
            for c in self.manager.custom_models():
                if c.key == f"custom:{stem}":
                    return c
        return None

    def _build(self, model_key: str, fp16: bool) -> BaseEngine:
        # kunci klasik eksplisit dari UI
        if model_key == "classic_x2":
            return ClassicUpscaleEngine(self.device, scale=2)
        if model_key == "classic_x4":
            return ClassicUpscaleEngine(self.device, scale=4)
        if model_key == "classic_identity":
            return ClassicIdentityEngine(self.device)

        spec = self._spec_for_key(model_key)
        if spec is None:
            raise EngineError(f"Kunci model tidak dikenal: {model_key}")

        path = spec.path(self.manager.models_dir)
        if path.exists() and path.suffix.lower() == ".onnx":
            eng = EsrganOnnxEngine(self.device, path, key=model_key,
                                   display_name=spec.name, fp16=fp16)
            eng.load()
            return eng

        # Fallback klasik — aplikasi tetap jalan tanpa model AI.
        log.warning("Model '%s' belum terpasang; memakai engine klasik x%d", spec.name, spec.scale)
        if spec.scale <= 1:
            return ClassicIdentityEngine(self.device)
        return ClassicUpscaleEngine(self.device, scale=min(4, max(2, spec.scale)))

    # ------------------------------------------------------------------ #
    def face_model_path(self) -> Optional[Path]:
        spec = next(s for s in MODEL_REGISTRY if s.key == "gfpgan_v14")
        p = spec.path(self.manager.models_dir)
        return p if p.exists() else None

    def list_choices(self) -> list[tuple[str, str, bool]]:
        """(key, nama_tampil, tersedia) utk combobox UI."""
        choices: list[tuple[str, str, bool]] = []
        for s in MODEL_REGISTRY:
            if s.category == "face":
                continue
            ok = self.manager.status(s.key).usable
            choices.append((s.key, s.name, ok))
        for c in self.manager.custom_models():
            choices.append((c.key, c.name, True))
        choices.append(("classic_x2", "Klasik Cepat x2 (tanpa AI)", True))
        choices.append(("classic_x4", "Klasik Cepat x4 (tanpa AI)", True))
        return choices

    def clear(self) -> None:
        self._cache.clear()
        EsrganOnnxEngine.clear_cache()
