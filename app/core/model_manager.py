"""Manajer model: daftar model yang dikenal, cek keberadaan file, verifikasi hash.

Model disimpan di folder ``models/`` (dapat di-override). Aplikasi berjalan
100% offline setelah model diunduh sekali lewat ``scripts/download_models.py``.

Catatan lisensi model tercantum pada :data:`MODEL_REGISTRY` dan README.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("iep.models")


@dataclass
class ModelSpec:
    """Deskripsi satu berkas model AI."""

    key: str                       # id internal, mis. "general_x4"
    name: str                      # nama tampilan
    category: str                  # photo | anime | document | oldphoto | face
    filename: str                  # nama file di folder models/
    scale: int                     # skala native model (2 atau 4)
    url: str                       # sumber unduhan (hanya dipakai skrip download)
    sha256: str = ""               # hash verifikasi; kosong = lewati
    size_mb: float = 0.0           # ukuran perkiraan untuk UI
    license: str = "BSD-3-Clause (Real-ESRGAN)"
    optional: bool = False         # opsional -> aplikasi tetap jalan tanpa ini
    engine: str = "esrgan_onnx"    # esrgan_onnx | gfpgan_onnx | classic

    def path(self, root: Path) -> Path:
        return root / self.filename


@dataclass
class ModelStatus:
    spec: ModelSpec
    installed: bool = False
    hash_ok: Optional[bool] = None     # None bila tidak ada hash untuk dicek
    error: str = ""

    @property
    def usable(self) -> bool:
        return self.installed and self.hash_ok is not False


def default_models_dir() -> Path:
    """Folder model standar: <repo>/models (bisa dioverride settings)."""
    return Path(__file__).resolve().parents[2] / "models"


# --------------------------------------------------------------------------- #
# Registri model bawaan.
# URL memakai mirror resmi repo OpenCV/ai-model zoo & xinntao Real-ESRGAN
# (hanya diakses oleh skrip download, bukan saat memproses gambar).
# --------------------------------------------------------------------------- #
_REAL_ESRGAN_BASE = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/"
)
_MODEL_ZOO = "https://github.com/Saafke/EDSR_Tensorflow_trt_int8/raw/master/"

MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="general_x4",
        name="General Photo x4 (Real-ESRGAN)",
        category="photo",
        filename="realesrgan-general-x4v3.onnx",
        scale=4,
        url=_REAL_ESRGAN_BASE + "RealESRGAN_x4plus.pth",
        size_mb=67.0,
        license="BSD-3-Clause (Real-ESRGAN, xinntao)",
    ),
    ModelSpec(
        key="general_x2",
        name="General Photo x2 (Real-ESRGAN)",
        category="photo",
        filename="realesrgan-general-x2.onnx",
        scale=2,
        url=_REAL_ESRGAN_BASE + "RealESRGAN_x2plus.pth",
        size_mb=64.0,
        license="BSD-3-Clause (Real-ESRGAN, xinntao)",
    ),
    ModelSpec(
        key="anime_x4",
        name="Anime / Ilustrasi x4",
        category="anime",
        filename="realesrgan-anime-x4.onnx",
        scale=4,
        url=_REAL_ESRGAN_BASE + "RealESRGAN_x4plus_anime_6B.pth",
        size_mb=17.0,
        license="BSD-3-Clause (Real-ESRGAN anime, xinntao)",
    ),
    ModelSpec(
        key="document_x2",
        name="Dokumen / Teks x2 (EDSR-like)",
        category="document",
        filename="edsr-document-x2.onnx",
        scale=2,
        url=_MODEL_ZOO + "model_EDSR_RGB_x2.pb",
        size_mb=36.0,
        license="Apache-2.0 (EDSR re-implementation)",
        optional=True,
    ),
    ModelSpec(
        key="oldphoto_x4",
        name="Foto Lama / Restorasi x4",
        category="oldphoto",
        filename="realesrgan-general-x4v3.onnx",   # memakai bobot umum + pra-proses khusus
        scale=4,
        url=_REAL_ESRGAN_BASE + "RealESRGAN_x4plus.pth",
        size_mb=67.0,
        license="BSD-3-Clause (Real-ESRGAN, xinntao)",
    ),
    ModelSpec(
        key="gfpgan_v14",
        name="Restorasi Wajah GFPGAN v1.4",
        category="face",
        filename="gfpgan-v1_4.onnx",
        scale=1,
        url="https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/GFPGANv1.4.pth",
        size_mb=340.0,
        license="Apache-2.0 (GFPGAN, Tencent)",
        optional=True,
        engine="gfpgan_onnx",
    ),
]


class ModelManager:
    """Pelacak ketersediaan model di folder lokal + verifikasi SHA-256."""

    REGISTRY_FILE = "model_hashes.json"

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self._expected_hashes = self._load_expected_hashes()

    # -- util hash ---------------------------------------------------------- #
    def _load_expected_hashes(self) -> dict[str, str]:
        p = self.models_dir / self.REGISTRY_FILE
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Gagal membaca %s: %s", p, exc)
        return {}

    @staticmethod
    def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def verify(self, spec: ModelSpec) -> tuple[Optional[bool], str]:
        """Verifikasi integritas file. Return (ok|None, pesan_error)."""
        path = spec.path(self.models_dir)
        if not path.exists():
            return None, "File belum terpasang"
        expected = spec.sha256 or self._expected_hashes.get(spec.filename, "")
        if not expected:
            return None, "Hash referensi tidak tersedia - cek dilewati"
        actual = self.sha256_of(path)
        if actual != expected:
            return False, f"Hash tidak cocok ({actual[:12]}… != {expected[:12]}…)"
        return True, "OK"

    # -- kueri -------------------------------------------------------------- #
    def status(self, key: str) -> ModelStatus:
        spec = self.spec(key)
        st = ModelStatus(spec=spec)
        path = spec.path(self.models_dir)
        st.installed = path.exists()
        if st.installed:
            ok, msg = self.verify(spec)
            st.hash_ok = ok
            st.error = "" if ok is not False else msg
        return st

    def all_status(self) -> list[ModelStatus]:
        return [self.status(s.key) for s in MODEL_REGISTRY]

    def specs_for_category(self, category: str) -> list[ModelSpec]:
        return [s for s in MODEL_REGISTRY if s.category == category]

    def spec(self, key: str) -> ModelSpec:
        for s in MODEL_REGISTRY:
            if s.key == key:
                return s
        raise KeyError(f"Model tidak dikenal: {key}")

    def available_keys(self) -> list[str]:
        """Kunci model yang filenya sudah ada (termasuk hasil impor pengguna)."""
        keys = [s.key for s in MODEL_REGISTRY if self.status(s.key).usable]
        for extra in self.custom_models():
            keys.append(extra.key)
        return keys

    # -- impor model pengguna ----------------------------------------------- #
    def custom_models(self) -> list[ModelSpec]:
        """Daftar .onnx milik pengguna yang diimpor ke folder models/."""
        out: list[ModelSpec] = []
        known_files = {s.filename for s in MODEL_REGISTRY}
        if not self.models_dir.exists():
            return out
        for p in sorted(self.models_dir.glob("*.onnx")):
            if p.name in known_files:
                continue
            scale = _guess_scale_from_name(p.name)
            out.append(
                ModelSpec(
                    key=f"custom:{p.stem}",
                    name=f"Model impor: {p.stem}",
                    category="photo",
                    filename=p.name,
                    scale=scale,
                    url="",
                    size_mb=round(p.stat().st_size / 1e6, 1),
                    license="Lisensi mengikuti berkas yang diimpor pengguna",
                    optional=False,
                )
            )
        return out

    def register_custom(self, src: Path) -> ModelSpec:
        """Salin berkas .onnx/.pth pengguna ke folder models dan daftarkan."""
        src = Path(src)
        if src.suffix.lower() not in (".onnx", ".pth"):
            raise ValueError("Format harus .onnx atau .pth")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        dst = self.models_dir / src.name
        n = 1
        while dst.exists():
            dst = self.models_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        import shutil

        shutil.copy2(src, dst)
        return ModelSpec(
            key=f"custom:{dst.stem}",
            name=f"Model impor: {dst.stem}",
            category="photo",
            filename=dst.name,
            scale=_guess_scale_from_name(dst.name),
            url="",
            size_mb=round(dst.stat().st_size / 1e6, 1),
            license="Lisensi mengikuti berkas yang diimpor pengguna",
        )


def _guess_scale_from_name(name: str) -> int:
    import re

    m = re.search(r"x(\d)", name.lower())
    return int(m.group(1)) if m else 4
