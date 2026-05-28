from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm.gold_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RAW_PATH,
    FACTS_GOLD_FIELDS,
    STRUCTURED_FACT_FIELDS,
    VISUAL_GOLD_FIELDS,
    load_dataset_index,
    load_raw_index,
)

# HYBRID = FACTS - STRUCTURED. Schema source of truth: gold_benchmark.py
HYBRID_FACT_FIELDS = tuple(f for f in FACTS_GOLD_FIELDS if f not in STRUCTURED_FACT_FIELDS)


def _normalize(text: object) -> str:
    # Translate Türkçe→ASCII lower()'dan ÖNCE — "İ".lower() Python'da combining char üretir.
    return str(text or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜI", "cgiosuCGIOSUI")).lower()


def _feature_blob(property_features: list[Any], title: str = "", description: str = "") -> str:
    return _normalize(" ".join(str(item) for item in property_features) + " " + title + " " + description)


def suggest_hybrid_facts(
    *,
    title: str,
    description: str,
    property_features: list[Any],
) -> dict[str, Any]:
    blob = _feature_blob(property_features, title, description)
    suggestions: dict[str, Any] = {field: None for field in HYBRID_FACT_FIELDS}

    checks = {
        "has_balcony": ("balkon", "teras"),
        "has_elevator": ("asansor",),
        "has_parking": ("otopark", "garaj", "park yeri"),
        "has_aircon": ("klima",),
        "near_metro": ("metro", "marmaray"),
        "near_metrobus": ("metrobus",),
    }
    for field, needles in checks.items():
        if any(needle in blob for needle in needles):
            suggestions[field] = True
    return suggestions


def suggest_visual_fields(property_features: list[Any]) -> dict[str, Any]:
    blob = _feature_blob(property_features)
    suggested: dict[str, Any] = {}
    if "somine" in blob:
        suggested["salon_ozellikleri"] = ["somine"]
    imkanlar = []
    for needle, value in (
        ("havuz", "havuz"),
        ("kapali otopark", "kapali_otopark"),
        ("acik otopark", "acik_otopark"),
        ("guvenlik", "guvenlik_kabini"),
        ("cocuk parki", "cocuk_parki"),
        ("spor", "spor_alani"),
        ("yesil alan", "yesil_alan_peyzaj"),
    ):
        if needle in blob:
            imkanlar.append(value)
    if imkanlar:
        suggested["imkanlar"] = imkanlar
    return suggested


def _structured_facts(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in STRUCTURED_FACT_FIELDS}


def _manual_todo_fields(record: dict[str, Any]) -> list[str]:
    auto_filled = set(STRUCTURED_FACT_FIELDS)
    return [field for field in FACTS_GOLD_FIELDS if field not in auto_filled] + list(VISUAL_GOLD_FIELDS)


def format_listing_view(
    listing_id: str,
    dataset_index: dict[str, dict[str, Any]],
    raw_index: dict[str, dict[str, Any]],
) -> str:
    record = dataset_index.get(listing_id)
    if not record:
        return f"Listing not found in dataset: {listing_id}"

    raw_record = raw_index.get(listing_id, {})
    description = raw_record.get("description") or record.get("description") or record.get("text", "")
    image_paths = record.get("all_image_paths") or []
    if not image_paths and record.get("image_path"):
        image_paths = [record["image_path"]]
    property_features = record.get("property_features") or (record.get("attributes") or {}).get("propertyFeatures") or []

    hybrid = suggest_hybrid_facts(
        title=str(record.get("title", "")),
        description=str(description),
        property_features=property_features,
    )
    visual = suggest_visual_fields(property_features)

    lines = [
        f"URL: {record.get('url', '')}",
        f"Title: {record.get('title', '')}",
        f"Parsed location: {record.get('city', '')} / {record.get('district', '')} / {record.get('neighborhood', '')}",
        "",
        "STRUCTURED facts:",
        json.dumps(_structured_facts(record), ensure_ascii=False, indent=2),
        "",
        "Property features:",
    ]
    lines.extend(f"  - {feature}" for feature in property_features)
    lines.extend([
        "",
        "Description:",
        str(description),
        "",
        "Image paths:",
    ])
    lines.extend(f"  [{index}] {path}" for index, path in enumerate(image_paths))
    lines.extend([
        "",
        "SUGGESTED hybrid facts (verify manually):",
        json.dumps(hybrid, ensure_ascii=False, indent=2),
        "",
        "SUGGESTED visual fields (from property_features cross-check):",
        json.dumps(visual, ensure_ascii=False, indent=2),
        "",
        "MANUAL TODO fields:",
    ])
    lines.extend(f"  - {field}" for field in _manual_todo_fields(record))
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Single-listing viewer for manual gold labeling")
    parser.add_argument("--listing", required=True, help="Listing ID from gold_listings_manual_todo.jsonl")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH))
    args = parser.parse_args()

    dataset_index = load_dataset_index(Path(args.dataset))
    raw_index = load_raw_index(Path(args.raw))
    print(format_listing_view(args.listing, dataset_index, raw_index))


if __name__ == "__main__":
    main()
