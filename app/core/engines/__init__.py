"""Paket engine Image Enhancer Pro.

Impor modul di sini agar dekorator ``@register_engine`` dijalankan dan
registry :mod:`app.core.engines.base_engine` terisi.
"""
from app.core.engines.base_engine import (  # noqa: F401
    BaseEngine,
    create_engine,
    register_engine,
    registered_keys,
)
from app.core.engines.classic_engine import (  # noqa: F401
    ClassicIdentityEngine,
    ClassicUpscaleEngine,
)
from app.core.engines.esrgan_engine import EsrganOnnxEngine  # noqa: F401
from app.core.engines.face_engine import FaceEnhancer, FaceEngine  # noqa: F401

__all__ = [
    "BaseEngine",
    "ClassicIdentityEngine",
    "ClassicUpscaleEngine",
    "EsrganOnnxEngine",
    "FaceEngine",
    "FaceEnhancer",
    "create_engine",
    "register_engine",
    "registered_keys",
]
