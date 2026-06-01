from __future__ import annotations

from typing import Any

from schema.emlakjet_filters import EMLAKJET_FILTERS


Scalar = str | int | float | bool


def embedding_text(record: dict[str, Any]) -> str:
    """Return the M3 document contract unchanged for embedding."""
    enriched_doc = record.get("enriched_doc")
    if not isinstance(enriched_doc, str) or not enriched_doc.strip():
        raise ValueError(f"Record {record.get('id')!r} is missing enriched_doc")
    return enriched_doc


def _metadata_value(field_name: str, value: Any) -> Scalar | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = sorted({str(item) for item in value if item is not None and str(item)})
        return "|".join(items) or None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Metadata field {field_name!r} must be scalar or list, got {type(value).__name__}")


def to_metadata(record: dict[str, Any]) -> dict[str, Scalar]:
    """Flatten an M3 labeled record into Chroma-compatible scalar metadata."""
    filters = record.get("filter_values") or {}
    metadata: dict[str, Scalar] = {}

    title = _metadata_value("title", record.get("title"))
    if title is not None:
        metadata["title"] = title

    for spec in EMLAKJET_FILTERS:
        value = _metadata_value(spec.slug, filters.get(spec.slug))
        if value is not None:
            metadata[spec.slug] = value
        if spec.value_type == "multi_enum":
            for option in filters.get(spec.slug) or []:
                if option not in spec.values.values():
                    raise ValueError(f"Metadata field {spec.slug!r} has unknown option {option!r}")
                metadata[f"{spec.slug}__{option}"] = True

    return metadata
