"""rasmcore — Image processing SDK powered by WebAssembly.

Two usage modes:

Dynamic (works with any module, no codegen needed):
    from rasmcore import RcImage
    img = RcImage.load("rasmcore_image.wasm", open("photo.png", "rb").read())
    jpeg = img.apply("blur", radius=3.0).encode("jpeg", quality=85)

Typed (generated per-module, full autocomplete):
    python -m rasmcore.typegen manifest.json --output rasmcore_typed.py
    from rasmcore_typed import RcImage
    jpeg = RcImage.load("module.wasm", data).blur(radius=3.0).to_jpeg(quality=85)
"""

__version__ = "0.1.0"

from .runtime import RcImage
from .types import FilterManifest, OperationMeta, ParamMeta

__all__ = [
    "RcImage",
    "FilterManifest",
    "OperationMeta",
    "ParamMeta",
]
