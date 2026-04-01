"""WASM Component Model pipeline — loads rasmcore-image.wasm and exposes Python API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import wasmtime
from wasmtime import Config, Engine, Linker, Module, Store


def _find_wasm() -> Path:
    """Locate the rasmcore-image WASM component binary."""
    # Check common locations
    candidates = [
        Path(__file__).parent / "rasmcore_image.wasm",
        Path.cwd() / "target" / "wasm32-wasip1" / "release" / "rasmcore_image.wasm",
        Path.cwd() / "target" / "wasm32-wasip1" / "debug" / "rasmcore_image.wasm",
    ]
    env_path = os.environ.get("RASMCORE_WASM_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "rasmcore_image.wasm not found. Set RASMCORE_WASM_PATH or build with: "
        "cargo component build -p rasmcore-image --release"
    )


class Pipeline:
    """Image processing pipeline backed by WASM Component Model.

    Usage:
        from rasmcore import Pipeline

        pipe = Pipeline()
        with open("input.jpg", "rb") as f:
            img = pipe.read(f.read())
        blurred = pipe.blur(img, radius=5.0)
        result = pipe.write_png(blurred)
        with open("output.png", "wb") as f:
            f.write(result)
    """

    def __init__(
        self,
        wasm_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        self._use_gpu = use_gpu
        self._gpu_executor = None

        # Initialize GPU if requested
        if use_gpu:
            try:
                from .gpu_host import WgpuExecutor
                self._gpu_executor = WgpuExecutor()
            except Exception:
                pass  # GPU unavailable — CPU fallback

        # Load WASM component
        path = Path(wasm_path) if wasm_path else _find_wasm()

        config = Config()
        config.wasm_component_model = True
        self._engine = Engine(config)
        self._linker = Linker(self._engine)
        self._linker.define_wasi()

        # If GPU available, provide the gpu-execute import
        if self._gpu_executor:
            self._register_gpu_import()

        self._module = Module.from_file(self._engine, str(path))
        self._store = Store(self._engine)
        self._instance = self._linker.instantiate(self._store, self._module)

        # Get pipeline resource constructor
        self._pipeline_new = self._instance.exports(self._store).get("pipeline", "new")
        self._pipeline = None

    def _register_gpu_import(self):
        """Register the gpu-execute host function for GPU offload."""
        executor = self._gpu_executor

        def gpu_execute(ops, input_bytes, width, height):
            """Host-side GPU execution via wgpu."""
            from .gpu_host import translate_ops
            native_ops = translate_ops(ops)
            return executor.execute(native_ops, input_bytes, width, height)

        # Register as WIT import
        # Note: exact registration depends on wasmtime-py component model API
        # This is a placeholder — the actual binding depends on the generated types
        self._gpu_execute_fn = gpu_execute

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._pipeline_new(self._store)

    def read(self, image_bytes: bytes) -> int:
        """Read encoded image bytes (PNG, JPEG, etc.) and return a node ID."""
        self._ensure_pipeline()
        read_fn = self._instance.exports(self._store).get("pipeline", "read")
        return read_fn(self._store, self._pipeline, image_bytes)

    def blur(self, source: int, radius: float = 5.0) -> int:
        """Apply gaussian blur."""
        self._ensure_pipeline()
        blur_fn = self._instance.exports(self._store).get("pipeline", "blur")
        return blur_fn(self._store, self._pipeline, source, {"radius": radius})

    def spherize(self, source: int, strength: float = 0.5) -> int:
        """Apply spherize distortion."""
        self._ensure_pipeline()
        fn = self._instance.exports(self._store).get("pipeline", "spherize")
        return fn(self._store, self._pipeline, source, {"strength": strength})

    def sepia(self, source: int, intensity: float = 1.0) -> int:
        """Apply sepia tone."""
        self._ensure_pipeline()
        fn = self._instance.exports(self._store).get("pipeline", "sepia")
        return fn(self._store, self._pipeline, source, {"intensity": intensity})

    def hue_rotate(self, source: int, degrees: float = 90.0) -> int:
        """Rotate hue."""
        self._ensure_pipeline()
        fn = self._instance.exports(self._store).get("pipeline", "hue-rotate")
        return fn(self._store, self._pipeline, source, {"degrees": degrees})

    def write_png(self, source: int, gpu: Optional[bool] = None) -> bytes:
        """Encode as PNG and return bytes."""
        self._ensure_pipeline()
        write_fn = self._instance.exports(self._store).get("pipeline", "write-png")
        config = {}
        return write_fn(self._store, self._pipeline, source, config, None)

    def write_jpeg(self, source: int, quality: int = 85, gpu: Optional[bool] = None) -> bytes:
        """Encode as JPEG and return bytes."""
        self._ensure_pipeline()
        write_fn = self._instance.exports(self._store).get("pipeline", "write-jpeg")
        config = {"quality": quality}
        return write_fn(self._store, self._pipeline, source, config, None)

    def write(self, source: int, format: str = "png", quality: int = 85, gpu: Optional[bool] = None) -> bytes:
        """Generic write — dispatches to format-specific encoder."""
        self._ensure_pipeline()
        write_fn = self._instance.exports(self._store).get("pipeline", "write")
        return write_fn(
            self._store, self._pipeline, source,
            format, quality if quality > 0 else None, None,
        )
