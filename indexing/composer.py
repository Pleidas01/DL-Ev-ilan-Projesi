from __future__ import annotations

from typing import Any

from llm.gold_benchmark import FACTS_GOLD_FIELDS, VISUAL_GOLD_FIELDS


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
    facts = record.get("facts_gold") or {}
    visual = (record.get("visual_qualities") or {}).get("aggregated") or {}
    metadata: dict[str, Scalar] = {}

    title = _metadata_value("title", record.get("title"))
    if title is not None:
        metadata["title"] = title

    for field_name in FACTS_GOLD_FIELDS:
        value = _metadata_value(field_name, facts.get(field_name))
        if value is not None:
            metadata[field_name] = value

    for field_name in VISUAL_GOLD_FIELDS:
        value = _metadata_value(field_name, visual.get(field_name))
        if value is not None:
            metadata[field_name] = value

    return metadata
