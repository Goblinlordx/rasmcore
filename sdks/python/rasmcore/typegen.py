"""rasmcore-typegen — Generate typed RcImage wrapper from a WASM module's manifest.

Usage:
    python -m rasmcore.typegen manifest.json > rasmcore_typed.py
    python -m rasmcore.typegen manifest.json --output rasmcore_typed.py

Reads param-manifest.json (from alongside the WASM module or build output)
and emits a typed Python class with methods for all operations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _rust_type_to_python(rust_type: str) -> str:
    """Map a Rust/WIT type to a Python type annotation."""
    mapping = {
        "f32": "float",
        "f64": "float",
        "u8": "int",
        "u16": "int",
        "u32": "int",
        "u64": "int",
        "i32": "int",
        "i64": "int",
        "bool": "bool",
        "string": "str",
        "String": "str",
    }
    if rust_type in mapping:
        return mapping[rust_type]
    if rust_type.startswith("list<") or rust_type.startswith("&["):
        return "list"
    return "Any"


def _snake_case(name: str) -> str:
    """Ensure a name is valid Python snake_case."""
    return re.sub(r"-", "_", name)


def generate_typed_sdk(manifest: dict, manifest_hash: str) -> str:
    """Generate a typed Python SDK from a manifest dictionary."""
    lines: list[str] = []

    filters = manifest.get("filters", [])

    lines.append('"""Auto-generated typed rasmcore SDK.')
    lines.append("")
    lines.append(f"Module manifest hash: {manifest_hash}")
    lines.append(f"Operations: {len(filters)}")
    lines.append("")
    lines.append("Do not edit — regenerate with:")
    lines.append("    python -m rasmcore.typegen <manifest.json>")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import warnings")
    lines.append("from typing import Any")
    lines.append("")
    lines.append("from rasmcore.runtime import RcImage as RcImageDynamic")
    lines.append("")
    lines.append(f'MANIFEST_HASH = "{manifest_hash}"')
    lines.append("")
    lines.append("")
    lines.append("class RcImage(RcImageDynamic):")
    lines.append('    """Typed image processing chain.')
    lines.append("")
    lines.append("    Generated from a specific WASM module — provides autocomplete,")
    lines.append("    type checking, and parameter documentation for all operations.")
    lines.append("    Extends RcImageDynamic — all typed methods delegate to apply()/encode().")
    lines.append('    """')
    lines.append("")
    lines.append("    @classmethod")
    lines.append('    def load(cls, wasm_path: str, data: bytes) -> "RcImage":')
    lines.append('        """Load image data and validate manifest hash."""')
    lines.append("        base = RcImageDynamic.load(wasm_path, data)")
    lines.append("        if base.manifest_hash and base.manifest_hash != MANIFEST_HASH:")
    lines.append("            warnings.warn(")
    lines.append('                f"rasmcore: typed SDK was generated for module {MANIFEST_HASH[:8]}, "')
    lines.append('                f"but loaded module is {base.manifest_hash[:8]}. "')
    lines.append('                f"Regenerate with: python -m rasmcore.typegen <manifest.json>",')
    lines.append("                UserWarning,")
    lines.append("                stacklevel=2,")
    lines.append("            )")
    lines.append("        base.__class__ = cls")
    lines.append("        return base  # type: ignore[return-value]")
    lines.append("")

    # Generate filter methods
    for f in filters:
        method_name = _snake_case(f["name"])
        params = f.get("params", [])
        reference = f.get("reference", "")

        # Build signature
        py_params = []
        for p in params:
            py_name = _snake_case(p["name"])
            py_type = _rust_type_to_python(p.get("type", ""))
            default = p.get("default")
            if default is not None and default != "":
                py_params.append(f"{py_name}: {py_type} = {repr(default)}")
            else:
                py_params.append(f"{py_name}: {py_type}")

        sig = ", ".join(["self"] + py_params)

        # Docstring
        doc = reference or f"{f['name']} filter"
        param_docs = []
        for p in params:
            py_name = _snake_case(p["name"])
            desc = p.get("label", "") or py_name
            if p.get("min") is not None and p.get("max") is not None:
                desc += f" ({p['min']}–{p['max']})"
            param_docs.append(f"            {py_name}: {desc}")

        lines.append(f"    def {method_name}({sig}) -> RcImage:")
        lines.append(f'        """{doc}')
        if param_docs:
            lines.append("")
            lines.append("        Args:")
            lines.extend(param_docs)
        lines.append('        """')

        # Body — delegate to apply()
        if params:
            kwarg_pairs = ", ".join(
                f"{_snake_case(p['name'])}={_snake_case(p['name'])}" for p in params
            )
            lines.append(f'        return self.apply("{f["name"]}", {kwarg_pairs})  # type: ignore[return-value]')
        else:
            lines.append(f'        return self.apply("{f["name"]}")  # type: ignore[return-value]')
        lines.append("")

    # Generate encode methods from common formats
    common_formats = ["jpeg", "png", "webp", "avif", "tiff", "bmp", "ico", "qoi", "gif"]
    lines.append("    # ─── Encode operations ───")
    lines.append("")
    for fmt in common_formats:
        method_name = f"to_{fmt}"
        lines.append(f"    def {method_name}(self, **config: Any) -> bytes:")
        lines.append(f'        """Encode as {fmt.upper()}."""')
        lines.append(f'        return self.encode("{fmt}", **config)')
        lines.append("")

    return "\n".join(lines)


def generate_stub(manifest: dict, manifest_hash: str) -> str:
    """Generate a .pyi stub file for IDE autocomplete."""
    lines: list[str] = []
    filters = manifest.get("filters", [])

    lines.append("# Auto-generated type stubs for rasmcore typed SDK")
    lines.append(f"# Manifest hash: {manifest_hash}")
    lines.append("")
    lines.append("from typing import Any")
    lines.append("from rasmcore.runtime import RcImage as RcImageDynamic")
    lines.append("")
    lines.append("class RcImage(RcImageDynamic):")
    lines.append("    @classmethod")
    lines.append('    def load(cls, wasm_path: str, data: bytes) -> "RcImage": ...')

    for f in filters:
        method_name = _snake_case(f["name"])
        params = f.get("params", [])
        py_params = []
        for p in params:
            py_name = _snake_case(p["name"])
            py_type = _rust_type_to_python(p.get("type", ""))
            default = p.get("default")
            if default is not None and default != "":
                py_params.append(f"{py_name}: {py_type} = ...")
            else:
                py_params.append(f"{py_name}: {py_type}")
        sig = ", ".join(["self"] + py_params)
        lines.append(f'    def {method_name}({sig}) -> "RcImage": ...')

    common_formats = ["jpeg", "png", "webp", "avif", "tiff", "bmp", "ico", "qoi", "gif"]
    for fmt in common_formats:
        lines.append(f"    def to_{fmt}(self, **config: Any) -> bytes: ...")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate typed Python SDK from a rasmcore manifest."
    )
    parser.add_argument("manifest", help="Path to param-manifest.json")
    parser.add_argument("--output", "-o", help="Output .py file (default: stdout)")
    parser.add_argument("--stub", action="store_true", help="Also generate .pyi stub")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    filters = manifest.get("filters", [])

    # Look for hash file next to manifest
    hash_path = manifest_path.parent / "param-manifest.hash"
    manifest_hash = hash_path.read_text().strip() if hash_path.exists() else ""

    print(
        f"Read manifest: {len(filters)} operations, hash={manifest_hash[:8] or '(none)'}",
        file=sys.stderr,
    )

    output = generate_typed_sdk(manifest, manifest_hash)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Generated: {out_path}", file=sys.stderr)

        if args.stub:
            stub = generate_stub(manifest, manifest_hash)
            stub_path = out_path.with_suffix(".pyi")
            stub_path.write_text(stub)
            print(f"Generated stub: {stub_path}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
