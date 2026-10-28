"""Pipeline enhancement lengkap: load -> validasi -> pra-proses -> tiling AI ->
face enhance -> pasca-proses -> resize target -> ekspor.

Kelas ini adalah "logika murni" (tanpa Qt). Worker thread di
:mod:`app.services.job_queue` memanggil :meth:`EnhancePipeline.run` dan
mendapat laporan progres lewat callback.

Alur data (ringkas)::

    file --load_image--> ImageData(rgb float32, alpha, icc, exif)
         --pra-proses--> denoise / jpeg-fix (opsional)
         --tiling-----> engine.process_tile per tile + feather stitch
         --wajah------> FaceEnhancer (opsional, blending)
         --pasca------> unsharp adaptif, koreksi warna, resize target
         --ekspor-----> Exporter.save (format+metadata+alpha)

Ketahanan:
  * OOM -> tile otomatis dikecilkan separuh lalu percobaan ulang (maks 3x).
  * Pembatalan dicek sebelum/setiap tile via CancellationToken.
  * Memori dibersihkan (gc + empty_cache) setelah tiap gambar.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.core.device import DeviceInfo, memory_risk
from app.core.engine_factory import EngineProvider
from app.core.engines.classic_engine import (
    adjust_color,
    auto_levels,
    clahe,
    denoise,
    guided_blend,
    remove_jpeg_artifacts,
    resize_lanczos,
    unsharp_mask,
)
from app.core.engines.face_engine import FaceEnhancer
from app.core.exporter import Exporter
from app.core.image_io import load_image
from app.core.tiling import TileStitcher, crop_tile, plan_tiles
from app.core.types import (
    CancellationToken,
    EnhanceParams,
    EngineError,
    ImageData,
    JobCanceled,
    OutOfMemoryError,
    OutputFormat,
)

log = logging.getLogger("iep.pipeline")

#: callback progres: (fraksi 0..1, teks tahap berbahasa Indonesia)
ProgressFn = Callable[[float, str], None]


class EnhancePipeline:
    """Orkestrator proses; bergantung pada DI: device, engine provider, exporter."""

    MAX_TILE_RETRY = 3

    def __init__(self, device: DeviceInfo, provider: EngineProvider,
                 exporter: Optional[Exporter] = None) -> None:
        self.device = device
        self.provider = provider
        self.exporter = exporter or Exporter()
        self._face: Optional[FaceEnhancer] = None

    # ------------------------------------------------------------------ #
    # Tahap individual (dipisah agar mudah diuji)
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess(img: np.ndarray, params: EnhanceParams) -> np.ndarray:
        """Pra-proses klasik SEBELUM model: denoise & perbaikan artefak JPEG."""
        out = img
        if params.jpeg_fix:
            out = remove_jpeg_artifacts(out, strength=45)
        if params.denoise > 0:
            # Denoise ringan pr-AI: kekuatan dipotong 60% agar model tidak kehilangan tekstur
            out = denoise(out, strength=int(params.denoise * 0.6))
        return out

    def upscale(self, img: np.ndarray, params: EnhanceParams,
                token: CancellationToken, progress: ProgressFn,
                base_frac: float, span_frac: float) -> tuple[np.ndarray, float]:
        """Jalankan engine dengan tiling+feather; tangani OOM dgn retry tile kecil.

        Returns:
            (gambar hasil, skala efektif yang benar-benar dipakai engine).
        """
        engine = self.provider.get(params.model_key, fp16=params.fp16)
        h, w = img.shape[:2]
        tile, overlap = params.tile_size, params.tile_overlap
        attempt = 0
        while True:
            try:
                result = self._run_tiled(engine, img, tile, overlap, token,
                                         progress, base_frac, span_frac)
                return result, float(engine.native_scale)
            except OutOfMemoryError as exc:
                new_tile = max(128, (tile // 2) // 8 * 8)
                attempt += 1
                log.warning("OOM (%s); tile %d -> %d (percobaan %d)",
                            exc, tile, new_tile, attempt)
                if attempt > self.MAX_TILE_RETRY or new_tile == tile:
                    raise EngineError(
                        "Memori tidak cukup bahkan dengan tile terkecil. "
                        "Coba format model lebih ringan atau tutup aplikasi lain."
                    ) from exc
                tile = new_tile
                overlap = min(overlap, tile // 4)
                gc.collect()

    def _run_tiled(self, engine, img: np.ndarray, tile: int, overlap: int,
                   token: CancellationToken, progress: ProgressFn,
                   base: float, span: float) -> np.ndarray:
        """Inti tiling: rencana -> inferensi per tile -> feather stitch."""
        scale = engine.native_scale
        h, w = img.shape[:2]
        rects = plan_tiles(w, h, tile, overlap)
        stitcher = TileStitcher(int(round(w * scale)), int(round(h * scale)), channels=3)
        total = len(rects)
        for i, rect in enumerate(rects):
            token.raise_if_canceled()
            piece = crop_tile(img, rect)
            out_tile = engine.process_tile(piece, token)
            edges = (
                rect.x == 0,
                rect.x + rect.w >= w,
                rect.y == 0,
                rect.y + rect.h >= h,
            )
            stitcher.add(out_tile, rect, float(scale), overlap, edges)
            frac = base + span * ((i + 1) / max(1, total))
            progress(frac, f"Mengolah tile {i + 1}/{total}")
            del piece, out_tile
        return stitcher.finalize()

    # ------------------------------------------------------------------ #
    @staticmethod
    def postprocess(img: np.ndarray, original_small: np.ndarray,
                    params: EnhanceParams, effective_scale: float) -> np.ndarray:
        """Pasca-proses: blending detail, unsharp adaptif, koreksi warna."""
        out = guided_blend(img, img, params.detail_balance)

        # Unsharp mask ADAPTIF: makin besar upscale, makin halus radius & amount
        # agar tidak terjadi halo/oversharp pada hasil AI.
        if params.deblur > 0:
            amount = (params.deblur / 100.0) * 1.2
            radius = 0.8 + 0.7 * min(effective_scale, 4.0) / 4.0
            out = unsharp_mask(out, amount=amount, radius=radius)

        # Koreksi warna (setelah penajaman supaya kontras tidak menggandakan noise)
        if params.auto_levels:
            out = auto_levels(out)
        if params.clahe:
            out = clahe(out, params.clahe_strength)
        if params.brightness or params.contrast or params.saturation:
            out = adjust_color(out, params.brightness, params.contrast, params.saturation)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def resize_to_target(img: np.ndarray, params: EnhanceParams,
                         src_w: int, src_h: int) -> np.ndarray:
        """Resize akhir ke skala target (x2/x3/custom dimensi)."""
        if params.custom_target_w > 0 and params.custom_target_h > 0:
            tw, th = params.custom_target_w, params.custom_target_h
        else:
            tw = int(round(src_w * params.scale))
            th = int(round(src_h * params.scale))
        if (tw, th) == (img.shape[1], img.shape[0]):
            return img
        return resize_lanczos(img, tw, th)

    @staticmethod
    def upscale_alpha(alpha: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Alpha di-upscale terpisah dengan Lanczos (menjaga tepi transparan)."""
        import cv2

        return cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)\
            .astype(np.float32)

    # ------------------------------------------------------------------ #
    # Titik masuk utama
    # ------------------------------------------------------------------ #
    def run(self, input_path: Path, params: EnhanceParams, token: CancellationToken,
            progress: Optional[ProgressFn] = None,
            output_dir: Optional[Path] = None,
            fmt: OutputFormat = OutputFormat.PNG, jpg_quality: int = 95,
            pattern: str = "{name}_enhanced_{scale}x", overwrite: bool = False,
            warn_memory: bool = True) -> tuple[Path, float]:
        """Proses satu gambar dari awal sampai file hasil tertulis.

        Args: lihat deskripsi modul. ``progress`` default no-op.

        Returns:
            (path hasil, durasi detik).

        Raises:
            EngineError / JobCanceled.
        """
        params = params.clamp()
        progress = progress or (lambda f, s: None)
        t0 = time.time()

        # 1-2) Load + validasi + estimasi memori
        progress(0.02, "Memuat gambar…")
        image = load_image(input_path)
        if warn_memory:
            risky, msg = memory_risk(image.width, image.height, params.scale,
                                     params.tile_size, self.device)
            if risky:
                log.warning(msg)
                progress(0.03, msg)   # diteruskan ke UI sbg peringatan, bukan error

        # 3) Pra-proses
        progress(0.06, "Pra-proses (denoise/artefak)…")
        token.raise_if_canceled()
        work = self.preprocess(image.rgb, params)

        # 4-5) Tiling + inferensi
        progress(0.10, "Memuat model…")
        upscaled, eff_scale = self.upscale(work, params, token, progress,
                                           base_frac=0.12, span_frac=0.70)

        # 6) Face enhance (di ruang hasil, sebelum resize target)
        faces_done = 0
        if params.face_enhance:
            token.raise_if_canceled()
            progress(0.85, "Restorasi wajah…")
            face = self._get_face_enhancer()
            upscaled, faces_done = face.enhance(upscaled, params.face_strength, token)
            if faces_done:
                log.info("%d wajah ditingkatkan pada %s", faces_done, input_path.name)

        # 7) Pasca-proses + resize target + alpha
        progress(0.88, "Pasca-proses & penajaman…")
        token.raise_if_canceled()
        final_rgb = self.postprocess(upscaled, image.rgb, params, eff_scale)
        final_rgb = self.resize_to_target(final_rgb, params, image.width, image.height)

        alpha_out = None
        if image.alpha is not None:
            progress(0.92, "Menyelaraskan channel transparansi…")
            alpha_out = self.upscale_alpha(image.alpha, final_rgb.shape[1], final_rgb.shape[0])

        result = ImageData(rgb=final_rgb, alpha=alpha_out, icc=image.icc,
                           exif=image.exif, orientation=1,
                           grayscale_input=image.grayscale_input)

        # Bersihkan memori besar sebelum menulis file
        del work, upscaled, final_rgb
        gc.collect()

        # 8) Ekspor
        progress(0.94, "Menyimpan hasil…")
        out_dir = output_dir or input_path.parent
        saved = self.exporter.save(result, out_dir, input_path.stem, params, fmt,
                                   jpg_quality, pattern, overwrite)
        elapsed = time.time() - t0
        progress(1.0, "Selesai")
        return saved, elapsed

    # ------------------------------------------------------------------ #
    def preview_region(self, input_path: Path, params: EnhanceParams,
                       box: tuple[int, int, int, int],
                       token: Optional[CancellationToken] = None) -> ImageData:
        """Preview cepat: hanya proses crop ``box`` (x,y,w,h) koordinat input.

        Dipakai panel "Preview Region" agar pengguna bisa menilai hasil tanpa
        menunggu seluruh gambar. Hasil dikembalikan dalam memori (tidak ditulis).
        """
        token = token or CancellationToken()
        image = load_image(input_path)
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        w = max(16, min(w, image.width - x))
        h = max(16, min(h, image.height - y))
        crop = image.rgb[y:y + h, x:x + w]
        work = self.preprocess(crop, params)
        out, eff = self.upscale(work, params, token, lambda f, s: None, 0.0, 1.0)
        if params.face_enhance:
            out, _ = self._get_face_enhancer().enhance(out, params.face_strength, token)
        out = self.postprocess(out, crop, params, eff)
        alpha = None
        if image.alpha is not None:
            alpha = self.upscale_alpha(image.alpha[y:y + h, x:x + w],
                                       out.shape[1], out.shape[0])
        return ImageData(rgb=out, alpha=alpha)

    def benchmark(self, model_key: str, tile_candidates: list[int],
                  params: EnhanceParams) -> dict[int, float]:
        """Ukur ms/tile utk beberapa ukuran tile -> saran tile terbaik ( tercepat
        yg masih hemat memori ). Dipakai 'cek sistem' saat pertama jalan."""
        results: dict[int, float] = {}
        engine = self.provider.get(model_key, fp16=params.fp16)
        dummy = np.random.rand(1, 1, 3).astype(np.float32)
        for t in tile_candidates:
            dummy = np.random.rand(t, t, 3).astype(np.float32) * 0.4 + 0.3
            try:
                tic = time.perf_counter()
                engine.process_tile(dummy)
                ms = (time.perf_counter() - tic) * 1000.0
            except Exception as exc:
                log.warning("Benchmark tile %d gagal: %s", t, exc)
                ms = float("inf")
            results[t] = ms
            del dummy
            gc.collect()
        return results

    # ------------------------------------------------------------------ #
    def _get_face_enhancer(self) -> FaceEnhancer:
        if self._face is None:
            self._face = FaceEnhancer(self.device, self.provider.face_model_path())
        return self._face


def cleanup_gpu_memory() -> None:
    """Bersihkan cache GPU bila tersedia (panggil antar-gambar)."""
    gc.collect()
    try:
        import torch  # opsional; tidak wajib terpasang

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Ekspor nama agar lint tidak complain tentang impor tak terpakai.
_ = JobCanceled
