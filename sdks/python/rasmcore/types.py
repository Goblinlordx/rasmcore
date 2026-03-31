"""Manifest types — describes all operations a WASM module supports.

These types are module-agnostic. Any WASM component that exposes
get_filter_manifest() is a valid module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamMeta:
    """Metadata for a single operation parameter."""

    name: str
    type: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    label: str = ""
    hint: str = ""


@dataclass
class OperationMeta:
    """Metadata for a single filter/transform operation."""

    name: str
    category: str
    group: str = ""
    variant: str = ""
    reference: str = ""
    params: list[ParamMeta] = field(default_factory=list)


@dataclass
class FilterManifest:
    """Full manifest describing all operations a WASM module supports."""

    filters: list[OperationMeta] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> FilterManifest:
        """Parse a manifest from the JSON returned by get_filter_manifest()."""
        filters = []
        for f in data.get("filters", []):
            params = [
                ParamMeta(
                    name=p["name"],
                    type=p.get("type", ""),
                    min=p.get("min"),
                    max=p.get("max"),
                    step=p.get("step"),
                    default=p.get("default"),
                    label=p.get("label", ""),
                    hint=p.get("hint", ""),
                )
                for p in f.get("params", [])
            ]
            filters.append(
                OperationMeta(
                    name=f["name"],
                    category=f.get("category", ""),
                    group=f.get("group", ""),
                    variant=f.get("variant", ""),
                    reference=f.get("reference", ""),
                    params=params,
                )
            )
        return cls(filters=filters)
