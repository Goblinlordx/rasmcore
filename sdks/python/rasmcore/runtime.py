"""rasmcore dynamic runtime — loads any rasmcore-compatible WASM module.

Usage:
    from rasmcore import RcImage

    img = RcImage.load("rasmcore_image.wasm", open("photo.png", "rb").read())
    jpeg = img.apply("blur", radius=3.0).encode("jpeg", quality=85)

The runtime is module-agnostic — it works with any WASM component
that exposes get_filter_manifest() and get_manifest_hash().
"""

from __future__ import annotations

import json
import re
from typing import Any

from .types import FilterManifest, OperationMeta, ParamMeta


def _to_camel_case(name: str) -> str:
    """Convert snake_case or kebab-case to camelCase (for WASM method dispatch)."""
    return re.sub(r"[-_](\w)", lambda m: m.group(1).upper(), name)


def _validate_param(meta: ParamMeta, name: str, value: Any) -> None:
    """Validate a parameter value against its manifest metadata."""
    if value is None:
        return

    t = meta.type
    if t in ("f32", "f64", "u32", "u16", "u8", "i32") and not isinstance(value, (int, float)):
        raise TypeError(f"{name}: expected number, got {type(value).__name__}")
    if t == "bool" and not isinstance(value, bool):
        raise TypeError(f"{name}: expected bool, got {type(value).__name__}")
    if t in ("string", "String") and not isinstance(value, str):
        raise TypeError(f"{name}: expected str, got {type(value).__name__}")

    if isinstance(value, (int, float)):
        if meta.min is not None and value < meta.min:
            raise ValueError(f"{name}: {value} is below minimum {meta.min}")
        if meta.max is not None and value > meta.max:
            raise ValueError(f"{name}: {value} is above maximum {meta.max}")


class RcImage:
    """Dynamic image processing chain.

    Works with any rasmcore-compatible WASM module. Discovers available
    operations at load time via the module's embedded manifest.

    All operations are lazy — pixels are only computed when an encode
    method is called.
    """

    def __init__(
        self,
        pipeline: Any,
        node_id: int,
        manifest: FilterManifest,
        manifest_hash: str,
        write_formats: list[str],
    ) -> None:
        self._pipeline = pipeline
        self._node_id = node_id
        self._manifest = manifest
        self._manifest_hash = manifest_hash
        self._write_formats = write_formats
        self._operation_index: dict[str, OperationMeta] = {
            op.name: op for op in manifest.filters
        }

    @classmethod
    def load(cls, wasm_path: str, data: bytes) -> RcImage:
        """Load image data from a WASM module.

        Args:
            wasm_path: Path to the rasmcore WASM component file.
            data: Raw image file bytes (PNG, JPEG, WebP, etc.)
        """
        try:
            from wasmtime import Config, Engine, Linker, Module, Store
        except ImportError:
            raise ImportError(
                "wasmtime is required: pip install wasmtime"
            )

        # Load and instantiate the WASM component
        config = Config()
        engine = Engine(config)
        store = Store(engine)
        module = Module.from_file(engine, wasm_path)
        linker = Linker(engine)
        linker.define_wasi()
        instance = linker.instantiate(store, module)

        # Discover the pipeline interface
        # This is a simplified approach — full component model support
        # may require wasmtime-py component bindings
        pipeline = instance

        # Read manifest
        get_manifest = instance.exports(store).get("get-filter-manifest") or \
                       instance.exports(store).get("getFilterManifest")
        if not get_manifest:
            raise RuntimeError(
                "WASM module does not expose get_filter_manifest(). "
                "Ensure it was built with rasmcore pipeline support."
            )

        manifest_json = get_manifest(store)
        manifest = FilterManifest.from_json(json.loads(manifest_json))

        get_hash = instance.exports(store).get("get-manifest-hash") or \
                   instance.exports(store).get("getManifestHash")
        manifest_hash = get_hash(store) if get_hash else ""

        get_formats = instance.exports(store).get("supported-write-formats") or \
                      instance.exports(store).get("supportedWriteFormats")
        write_formats = get_formats(store) if get_formats else []

        # Read the image
        read_fn = instance.exports(store).get("read")
        if not read_fn:
            raise RuntimeError("WASM module does not expose read().")
        node_id = read_fn(store, data)

        return cls(pipeline, node_id, manifest, manifest_hash, write_formats)

    @property
    def manifest(self) -> FilterManifest:
        """The full manifest describing all operations this module supports."""
        return self._manifest

    @property
    def manifest_hash(self) -> str:
        """The manifest content hash — for SDK version validation."""
        return self._manifest_hash

    @property
    def write_formats(self) -> list[str]:
        """Available output formats (from registered encoders)."""
        return self._write_formats

    @property
    def available_operations(self) -> list[str]:
        """List all available filter/operation names."""
        return list(self._operation_index.keys())

    def operation_meta(self, name: str) -> OperationMeta | None:
        """Get metadata for a specific operation."""
        return self._operation_index.get(name)

    def fork(self) -> RcImage:
        """Branch the pipeline — returns a new RcImage sharing the same graph."""
        return RcImage(
            self._pipeline,
            self._node_id,
            self._manifest,
            self._manifest_hash,
            self._write_formats,
        )

    def apply(self, name: str, **params: Any) -> RcImage:
        """Apply a named operation with keyword parameters.

        Args:
            name: Operation name (snake_case, as in the manifest).
            **params: Parameter values keyed by parameter name.

        Returns:
            self (for chaining)

        Example:
            img.apply("blur", radius=3.0)
            img.apply("brightness", amount=0.1)
        """
        meta = self._operation_index.get(name)
        if meta is None:
            available = ", ".join(self.available_operations[:10])
            raise ValueError(
                f'Unknown operation "{name}". Available: {available}...'
            )

        # Validate params
        for pmeta in meta.params:
            value = params.get(pmeta.name)
            if value is not None:
                _validate_param(pmeta, f"{name}.{pmeta.name}", value)

        # Build ordered args: node_id, then params in manifest order
        args = [self._node_id]
        for pmeta in meta.params:
            value = params.get(pmeta.name, pmeta.default)
            args.append(value)

        # Dispatch to the pipeline method (camelCase for WASM convention)
        method_name = _to_camel_case(name)
        method = getattr(self._pipeline, method_name, None)
        if method is None:
            raise RuntimeError(
                f'Pipeline method "{method_name}" not found. '
                f"The WASM module may not implement this operation."
            )

        self._node_id = method(*args)
        return self

    def encode(self, format: str, **config: Any) -> bytes:
        """Encode the current image to a format.

        Args:
            format: Output format name (e.g., "jpeg", "png", "webp").
            **config: Format-specific configuration (e.g., quality=85).

        Returns:
            Encoded bytes.

        Example:
            jpeg = img.encode("jpeg", quality=85)
            png = img.encode("png", compression_level=6)
        """
        # Try format-specific write method first (e.g., writeJpeg)
        write_method = _to_camel_case(f"write_{format}")
        method = getattr(self._pipeline, write_method, None)

        if method is not None:
            return bytes(method(self._node_id, config, None))

        # Fall back to generic write(node_id, format, quality, metadata)
        generic = getattr(self._pipeline, "write", None)
        if generic is not None:
            quality = config.get("quality")
            return bytes(generic(self._node_id, format, quality, None))

        raise RuntimeError(f'No encoder found for format "{format}".')
