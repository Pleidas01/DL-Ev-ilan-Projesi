from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.clients import (
    ModelCandidate,
    candidate_by_id,
    complete_json,
    complete_vision_json,
    estimate_cost_usd,
    missing_environment,
)
from llm.gold_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GOLD_PATH,
    FACTS_GOLD_FIELDS,
    MULTI_SELECT_FIELDS,
    STRUCTURED_FACT_FIELDS,
    VISUAL_ENUMS,
    VISUAL_GOLD_FIELDS,
    field_score,
    gold_listing_ids,
    listing_image_paths,
    load_gold_rows,
    score_against_gold,
)
from llm.shootout_vision import VISION_SYSTEM_PROMPT, build_vision_user_prompt
from schema.emlakjet_filters import (
    empty_filter_values,
    parse_filter_value,
    spec_for_slug,
    specs_for_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "labeled.jsonl"
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "llm" / "selected.json"
DEFAULT_CLEAN_JSON_NAME = "clean_json.json"

TEXT_FACT_FIELDS = tuple(field for field in FACTS_GOLD_FIELDS if field not in STRUCTURED_FACT_FIELDS)
VISION_FIELDS = VISUAL_GOLD_FIELDS
PARKING_IMKANLAR = {"kapali_otopark", "acik_otopark"}

FACTS_THRESHOLD = 0.75
VISUAL_THRESHOLD = 0.70
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_VISION_IMAGE_TOKENS = 60
DEFAULT_TEXT_OUTPUT_TOKENS = 350
DEFAULT_VISION_OUTPUT_TOKENS = 700
DEFAULT_VISION_CHUNK_SIZE = 0
PROVIDER_MAX_RETRIES = 3

TEXT_SYSTEM_PROMPT = """Sen Turkce emlak ilani metninden kesin JSON alanlari cikaran bir asistansin.
Sadece gecerli JSON dondur. Aciklama yazma."""


def build_text_prompt(record: dict[str, Any]) -> str:
    specs = _null_specs_for_source(record, "description_llm")
    schema = {spec.slug: None for spec in specs}
    enums = {spec.slug: spec.values for spec in specs if spec.values}
    return f"""Ilan basligi:
{record.get("title") or ""}

Ilan aciklamasi:
{record.get("description") or ""}

JSON semasi:
{{"filters": {json.dumps(schema, ensure_ascii=False, indent=2)}}}

Enumlar:
{json.dumps(enums, ensure_ascii=False, indent=2)}

Kurallar:
- Yalnizca semadaki alanlari dondur.
- Boolean alanlarda true/false yalnizca baslik veya aciklamada acik kanit varsa kullan.
- Ulasim yakinligi icin yurume mesafesi veya <15 dakika anlami ara.
- Bilinmeyen alanlar icin null kullan.
"""


def build_vision_prompt(record: dict[str, Any]) -> str:
    return build_vision_user_prompt(_null_specs_for_source(record, "image_vlm"))


class CostCapExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    max_cost_usd: float | None
    estimated_usd: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, cost_usd: float, label: str) -> None:
        with self.lock:
            next_total = self.estimated_usd + cost_usd
            if self.max_cost_usd is not None and next_total > self.max_cost_usd:
                raise CostCapExceeded(
                    f"Cost cap exceeded before {label}: "
                    f"estimate {next_total:.4f} > cap {self.max_cost_usd:.4f}"
                )
            self.estimated_usd = next_total


def _fold(value: Any) -> str:
    table = str.maketrans("çğıöşüâîûÇĞİÖŞÜI", "cgiosuaiucgiosui")
    return str(value or "").strip().translate(table).lower()


def _clamp_confidence(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_transient_provider_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "tpd rate limit" in message or "tokens per day" in message:
        return False
    return "max organization concurrency" in message or "try again after" in message


def _provider_json_call(call, *args, **kwargs) -> str:
    for attempt in range(PROVIDER_MAX_RETRIES):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            if attempt == PROVIDER_MAX_RETRIES - 1 or not _is_transient_provider_error(exc):
                raise
            time.sleep(1.0 + attempt)
    return "{}"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    folded = _fold(value)
    if folded in {"true", "evet", "var", "yes", "1"}:
        return True
    if folded in {"false", "hayir", "hayır", "yok", "no", "0"}:
        return False
    return None


def _coerce_enum(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    folded = _fold(value)
    for allowed in VISUAL_ENUMS[field_name]:
        if folded == allowed:
            return allowed
    return None


def _coerce_enum_list(field_name: str, value: Any) -> list[str] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else str(value).replace("|", ",").split(",")
    normalized: list[str] = []
    for item in values:
        enum_value = _coerce_enum(field_name, item)
        if enum_value and enum_value not in normalized:
            normalized.append(enum_value)
    return normalized


def _current_filter_values(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    values = empty_filter_values()
    values.update(record.get("filter_values") or {})
    sources = dict(record.get("filter_sources") or {})
    for slug in values:
        if values[slug] is None and record.get(slug) is not None:
            values[slug] = record[slug]
            sources.setdefault(slug, "legacy_top_level")
    if values["price_amount"] is None and record.get("price_tl") is not None:
        values["price_amount"] = record["price_tl"]
        values["price_currency"] = "TL"
        sources.setdefault("price_amount", "legacy_top_level")
        sources.setdefault("price_currency", "legacy_top_level")
    return values, sources


def _null_specs_for_source(record: dict[str, Any], source: str):
    values, _sources = _current_filter_values(record)
    return tuple(spec for spec in specs_for_source(source) if values[spec.slug] is None)


def normalize_text_prediction(parsed: dict[str, Any], record: dict[str, Any] | None = None) -> dict[str, Any]:
    facts_source = parsed.get("filters") if isinstance(parsed.get("filters"), dict) else parsed.get("facts")
    facts_source = facts_source if isinstance(facts_source, dict) else parsed
    allowed = _null_specs_for_source(record, "description_llm") if record is not None else specs_for_source("description_llm")
    facts: dict[str, Any] = {}
    for spec in allowed:
        if spec.slug not in facts_source:
            continue
        value = parse_filter_value(spec, facts_source.get(spec.slug), strict=True)
        if value is not None:
            facts[spec.slug] = value
    imkanlar = _coerce_enum_list("imkanlar", parsed.get("imkanlar"))
    return {**facts, "imkanlar": imkanlar or []}


def merge_filter_values(record: dict[str, Any], prediction: dict[str, Any], source: str) -> tuple[dict[str, Any], dict[str, str]]:
    values, sources = _current_filter_values(record)
    required_spec_source = {"deepseek_description": "description_llm", "kimi_image": "image_vlm"}.get(source)
    for slug, value in prediction.items():
        spec = spec_for_slug(slug)
        if spec is None or required_spec_source and required_spec_source not in spec.sources:
            continue
        if source == "kimi_image" and spec.value_type == "bool" and value is not True:
            continue
        if values[slug] is None and value is not None:
            values[slug] = value
            sources[slug] = source
    return values, sources


def normalize_visual_filter_prediction(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    source = parsed.get("filters") if isinstance(parsed.get("filters"), dict) else parsed
    prediction: dict[str, Any] = {}
    for spec in _null_specs_for_source(record, "image_vlm"):
        if spec.slug not in source:
            continue
        value = parse_filter_value(spec, source.get(spec.slug), strict=True)
        if spec.value_type == "bool" and value is not True:
            continue
        if value is not None:
            prediction[spec.slug] = value
    return prediction


def normalize_visual_fields(raw_fields: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    source = raw_fields if isinstance(raw_fields, dict) else {}
    normalized: dict[str, Any] = {}
    for field_name in fields:
        if field_name in MULTI_SELECT_FIELDS:
            normalized[field_name] = _coerce_enum_list(field_name, source.get(field_name))
        else:
            normalized[field_name] = _coerce_enum(field_name, source.get(field_name))
    return normalized


def _visual_fields_source(parsed: dict[str, Any]) -> dict[str, Any]:
    for key in ("aggregated", "visual_gold", "fields"):
        value = parsed.get(key)
        if isinstance(value, dict):
            return value
    return parsed


def merge_facts(
    record: dict[str, Any],
    text_prediction: dict[str, Any],
    visual_aggregated: dict[str, Any] | None = None,
) -> dict[str, Any]:
    facts = {field: record.get(field) for field in STRUCTURED_FACT_FIELDS}
    for field_name in TEXT_FACT_FIELDS:
        existing_value = record.get(field_name)
        facts[field_name] = existing_value if existing_value is not None else text_prediction.get(field_name)

    visual = visual_aggregated or {}
    if facts.get("has_balcony") is None and visual.get("balkon_ozellikleri"):
        facts["has_balcony"] = True
    if facts.get("has_parking") is None and any(value in PARKING_IMKANLAR for value in visual.get("imkanlar") or []):
        facts["has_parking"] = True
    return {field: facts.get(field) for field in FACTS_GOLD_FIELDS}


def _path_for_image_index(image_paths: list[str], image_index: Any) -> str | None:
    try:
        index = int(image_index)
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(image_paths):
        return image_paths[index]
    return None


def _item_confidence(item: dict[str, Any], parsed: dict[str, Any]) -> float:
    if "confidence" in item:
        return _clamp_confidence(item.get("confidence"))
    confidence = parsed.get("confidence") or parsed.get("self_confidence")
    if isinstance(confidence, dict):
        values = [_clamp_confidence(value) for value in confidence.values()]
        return sum(values) / len(values) if values else 1.0
    return _clamp_confidence(confidence)


def _empty_visual_aggregate() -> dict[str, Any]:
    return {field_name: None for field_name in VISUAL_GOLD_FIELDS}


def _aggregate_from_items(per_image: list[dict[str, Any]], min_confidence: float) -> dict[str, Any]:
    accepted = [item for item in per_image if item["confidence"] >= min_confidence]
    aggregated = _empty_visual_aggregate()
    for field_name in VISION_FIELDS:
        if field_name in MULTI_SELECT_FIELDS:
            values: list[str] = []
            for item in accepted:
                for value in item["fields"].get(field_name) or []:
                    if value not in values:
                        values.append(value)
            aggregated[field_name] = values or None
        else:
            counts = Counter(item["fields"].get(field_name) for item in accepted if item["fields"].get(field_name))
            if counts:
                aggregated[field_name] = max(VISUAL_ENUMS[field_name], key=lambda value: (counts[value], -VISUAL_ENUMS[field_name].index(value)))
    return aggregated


def _merge_imkanlar(aggregated: dict[str, Any], text_imkanlar: list[str] | None, vision_imkanlar: list[str] | None) -> None:
    values: list[str] = []
    for source in (text_imkanlar or [], vision_imkanlar or []):
        for value in source:
            if value in VISUAL_ENUMS["imkanlar"] and value not in values:
                values.append(value)
    aggregated["imkanlar"] = values or None


def aggregate_visual_qualities(
    parsed: dict[str, Any],
    *,
    image_paths: list[str],
    text_imkanlar: list[str] | None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    confidence_method: str = "self",
) -> dict[str, Any]:
    raw_items = parsed.get("per_image") if isinstance(parsed.get("per_image"), list) else []
    per_image: list[dict[str, Any]] = []
    if raw_items:
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            fields = normalize_visual_fields(raw_item.get("fields"), VISION_FIELDS)
            per_image.append({
                "path": _path_for_image_index(image_paths, raw_item.get("image_index")),
                "image_index": raw_item.get("image_index"),
                "fields": fields,
                "confidence": _item_confidence(raw_item, parsed),
            })
    else:
        fields = normalize_visual_fields(_visual_fields_source(parsed), VISION_FIELDS)
        per_image.append({
            "path": None,
            "image_paths": image_paths,
            "fields": fields,
            "confidence": _item_confidence({}, parsed),
        })

    aggregated = _aggregate_from_items(per_image, min_confidence)
    raw_aggregated = _visual_fields_source(parsed)
    vision_imkanlar = []
    for value in aggregated.get("imkanlar") or []:
        if value not in vision_imkanlar:
            vision_imkanlar.append(value)
    for value in normalize_visual_fields(raw_aggregated, ("imkanlar",)).get("imkanlar") or []:
        if value not in vision_imkanlar:
            vision_imkanlar.append(value)
    _merge_imkanlar(aggregated, text_imkanlar, vision_imkanlar)
    return {
        "per_image": per_image,
        "aggregated": aggregated,
        "confidence_method": confidence_method,
        "min_confidence": min_confidence,
    }


def _agreement_aggregate(run_aggregates: list[dict[str, Any]], *, text_imkanlar: list[str] | None, min_confidence: float) -> dict[str, Any]:
    run_count = max(1, len(run_aggregates))
    aggregated = _empty_visual_aggregate()
    confidence_by_field: dict[str, float] = {}
    for field_name in VISION_FIELDS:
        if field_name in MULTI_SELECT_FIELDS:
            counts: Counter[str] = Counter()
            for fields in run_aggregates:
                counts.update(set(fields.get(field_name) or []))
            values = [value for value in VISUAL_ENUMS[field_name] if counts[value] / run_count >= min_confidence]
            aggregated[field_name] = values or None
            confidence_by_field[field_name] = max((counts[value] / run_count for value in VISUAL_ENUMS[field_name]), default=0.0)
        else:
            counts = Counter(fields.get(field_name) for fields in run_aggregates if fields.get(field_name))
            if counts:
                value, count = counts.most_common(1)[0]
                confidence_by_field[field_name] = count / run_count
                if count / run_count >= min_confidence:
                    aggregated[field_name] = value
            else:
                confidence_by_field[field_name] = 0.0
    _merge_imkanlar(aggregated, text_imkanlar, None)
    return {
        "per_image": [{
            "path": None,
            "fields": {field_name: aggregated[field_name] for field_name in VISION_FIELDS},
            "confidence": sum(confidence_by_field.values()) / len(confidence_by_field),
            "confidence_by_field": confidence_by_field,
        }],
        "aggregated": aggregated,
        "confidence_method": "agreement",
        "min_confidence": min_confidence,
        "agreement_runs": run_count,
    }


def _estimate_text_call(candidate: ModelCandidate, system_prompt: str, user_prompt: str) -> float:
    input_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
    return estimate_cost_usd(candidate, input_tokens, DEFAULT_TEXT_OUTPUT_TOKENS)


def _estimate_vision_call(candidate: ModelCandidate, user_prompt: str, image_count: int, image_tokens_per_image: int) -> float:
    input_tokens = max(1, len(user_prompt) // 4 + image_count * image_tokens_per_image)
    return estimate_cost_usd(candidate, input_tokens, DEFAULT_VISION_OUTPUT_TOKENS)


def extract_text_labels(record: dict[str, Any], candidate: ModelCandidate, cost_tracker: CostTracker) -> dict[str, Any]:
    user_prompt = build_text_prompt(record)
    cost_tracker.reserve(_estimate_text_call(candidate, TEXT_SYSTEM_PROMPT, user_prompt), "text labeling")
    raw = _provider_json_call(complete_json, candidate, TEXT_SYSTEM_PROMPT, user_prompt)
    return normalize_text_prediction(_parse_json_object(raw), record)


def _resolve_image_paths(record: dict[str, Any]) -> list[str]:
    resolved: list[str] = []
    for value in listing_image_paths(record, None):
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            resolved.append(str(path))
    return resolved


def _call_vision_once(
    image_paths: list[str],
    candidate: ModelCandidate,
    cost_tracker: CostTracker,
    image_tokens_per_image: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    user_prompt = build_vision_prompt(record)
    cost_tracker.reserve(
        _estimate_vision_call(candidate, user_prompt, len(image_paths), image_tokens_per_image),
        "vision labeling",
    )
    raw = _provider_json_call(complete_vision_json, candidate, VISION_SYSTEM_PROMPT, user_prompt, image_paths)
    return _parse_json_object(raw)


def _chunk_paths(image_paths: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size and chunk_size > 0 and len(image_paths) > chunk_size:
        return [image_paths[index : index + chunk_size] for index in range(0, len(image_paths), chunk_size)]
    return [image_paths]


def _vision_pass(
    image_paths: list[str],
    candidate: ModelCandidate,
    cost_tracker: CostTracker,
    image_tokens_per_image: int,
    *,
    record: dict[str, Any],
    chunk_size: int,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    # Büyük ilanlar tek çağrıda Moonshot timeout veriyor; fotoları chunk'lara bölüp
    # per_image sonuçlarını birleştiriyoruz — foto kaybı yok, union/vote aggregate aynı.
    per_image: list[dict[str, Any]] = []
    values, sources = _current_filter_values(record)
    for chunk in _chunk_paths(image_paths, chunk_size):
        current = {**record, "filter_values": values, "filter_sources": sources}
        parsed = _call_vision_once(chunk, candidate, cost_tracker, image_tokens_per_image, current)
        confidence = _item_confidence({}, parsed)
        prediction = normalize_visual_filter_prediction(parsed, current)
        if confidence >= min_confidence:
            values, sources = merge_filter_values(current, prediction, "kimi_image")
        per_image.append({
            "path": None,
            "image_paths": chunk,
            "fields": prediction,
            "confidence": confidence,
        })
    return per_image, values, sources


def _canonical_visual_compat(values: dict[str, Any]) -> dict[str, Any]:
    aggregated = _empty_visual_aggregate()
    balcony_map = {
        "acik_balkon": "acik_balkon",
        "acik_teras": "teras",
        "fransiz_balkon": "fransiz_balkon",
    }
    aggregated["balkon_ozellikleri"] = [
        balcony_map[value] for value in values.get("balcony_type") or [] if value in balcony_map
    ] or None
    view_map = {
        "has_sea_view": "deniz",
        "has_bosphorus_view": "bogaz",
        "has_green_view": "orman_yesil",
        "has_city_view": "sehir_panorama",
    }
    aggregated["manzara"] = [label for slug, label in view_map.items() if values.get(slug) is True] or None
    aggregated["mutfak_tipi"] = "amerikan_acik" if values.get("has_american_kitchen") is True else None
    bathroom_map = {"has_shower_cabin": "dusakabin", "has_bathtub": "kuvet", "has_jacuzzi": "jakuzi"}
    aggregated["banyo_ozellikleri"] = [label for slug, label in bathroom_map.items() if values.get(slug) is True] or None
    amenity_map = {
        "has_outdoor_pool": "havuz",
        "has_indoor_pool": "havuz",
        "has_private_pool": "havuz",
        "has_private_garden": "yesil_alan_peyzaj",
        "has_shared_garden": "yesil_alan_peyzaj",
        "has_garden": "yesil_alan_peyzaj",
        "has_closed_parking": "kapali_otopark",
        "has_open_parking": "acik_otopark",
        "has_playground": "cocuk_parki",
        "has_fitness": "spor_alani",
        "has_basketball_court": "spor_alani",
        "has_football_field": "spor_alani",
        "has_tennis_court": "spor_alani",
        "has_volleyball_court": "spor_alani",
    }
    aggregated["imkanlar"] = list(dict.fromkeys(
        label for slug, label in amenity_map.items() if values.get(slug) is True
    )) or None
    return aggregated


def extract_visual_labels(
    record: dict[str, Any],
    candidate: ModelCandidate,
    cost_tracker: CostTracker,
    *,
    text_imkanlar: list[str] | None,
    confidence_mode: str = "self",
    agreement_k: int = 3,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    image_tokens_per_image: int = DEFAULT_VISION_IMAGE_TOKENS,
    vision_chunk_size: int = DEFAULT_VISION_CHUNK_SIZE,
) -> dict[str, Any]:
    image_paths = _resolve_image_paths(record)
    if not image_paths:
        values, sources = _current_filter_values(record)
        aggregated = _canonical_visual_compat(values)
        _merge_imkanlar(aggregated, text_imkanlar, None)
        return {"per_image": [], "aggregated": aggregated, "filter_values": values, "filter_sources": sources, "confidence_method": confidence_mode, "min_confidence": min_confidence}

    if confidence_mode == "agreement":
        run_values = []
        per_image = []
        for _ in range(agreement_k):
            run_items, values, _sources = _vision_pass(
                image_paths, candidate, cost_tracker, image_tokens_per_image,
                record=record, chunk_size=vision_chunk_size, min_confidence=min_confidence,
            )
            per_image.extend(run_items)
            run_values.append(values)
        values, sources = _current_filter_values(record)
        for spec in specs_for_source("image_vlm"):
            observed = [run[spec.slug] for run in run_values if run.get(spec.slug) is not None]
            if not observed:
                continue
            if spec.value_type == "multi_enum":
                counts: Counter[str] = Counter(value for run in observed for value in run)
                accepted = [value for value, count in counts.items() if count / agreement_k >= min_confidence]
                prediction = accepted or None
            else:
                prediction, count = Counter(observed).most_common(1)[0]
                if count / agreement_k < min_confidence:
                    prediction = None
            values, sources = merge_filter_values(
                {**record, "filter_values": values, "filter_sources": sources},
                {spec.slug: prediction},
                "kimi_image",
            )
        aggregated = _canonical_visual_compat(values)
        _merge_imkanlar(aggregated, text_imkanlar, aggregated.get("imkanlar"))
        return {"per_image": per_image, "aggregated": aggregated, "filter_values": values, "filter_sources": sources, "confidence_method": "agreement", "min_confidence": min_confidence}

    per_image, values, sources = _vision_pass(
        image_paths, candidate, cost_tracker, image_tokens_per_image,
        record=record, chunk_size=vision_chunk_size, min_confidence=min_confidence,
    )
    aggregated = _canonical_visual_compat(values)
    vision_imkanlar = list(aggregated.get("imkanlar") or [])
    _merge_imkanlar(aggregated, text_imkanlar, vision_imkanlar)
    return {
        "per_image": per_image,
        "aggregated": aggregated,
        "filter_values": values,
        "filter_sources": sources,
        "confidence_method": "self",
        "min_confidence": min_confidence,
    }


def compose_enriched_doc(record: dict[str, Any], facts: dict[str, Any], visual: dict[str, Any]) -> str:
    fact_parts = [f"{field}: {facts.get(field)}" for field in FACTS_GOLD_FIELDS if facts.get(field) is not None]
    visual_parts = [f"{field}: {visual.get(field)}" for field in VISUAL_GOLD_FIELDS if visual.get(field)]
    return "\n".join([
        f"Baslik: {record.get('title') or ''}",
        f"Konum: {facts.get('city') or ''} / {facts.get('district') or ''} / {facts.get('neighborhood') or ''}",
        f"Fiyat: {facts.get('price_tl') or record.get('price') or ''} TL",
        "Facts: " + "; ".join(fact_parts),
        "Gorsel ozellikler: " + "; ".join(visual_parts),
        f"Aciklama: {record.get('description') or record.get('text') or ''}",
    ]).strip()


def _facts_with_filter_values(record: dict[str, Any], facts: dict[str, Any], filter_values: dict[str, Any]) -> dict[str, Any]:
    merged = dict(facts)
    for field_name in TEXT_FACT_FIELDS:
        if filter_values.get(field_name) is not None:
            merged[field_name] = filter_values[field_name]
    if merged.get("has_parking") is None and (
        filter_values.get("has_open_parking") is True or filter_values.get("has_closed_parking") is True
    ):
        merged["has_parking"] = True
    return merged


def label_text_record(
    record: dict[str, Any],
    text_candidate: ModelCandidate,
    *,
    cost_tracker: CostTracker,
) -> dict[str, Any]:
    text_prediction = extract_text_labels(record, text_candidate, cost_tracker)
    filter_values, filter_sources = merge_filter_values(record, text_prediction, "deepseek_description")
    aggregated = _canonical_visual_compat(filter_values)
    _merge_imkanlar(aggregated, text_prediction.get("imkanlar"), aggregated.get("imkanlar"))
    facts = merge_facts(record, text_prediction, aggregated)
    facts = _facts_with_filter_values(record, facts, filter_values)
    visual_qualities = {
        "per_image": [],
        "aggregated": aggregated,
        "filter_values": filter_values,
        "filter_sources": filter_sources,
        "confidence_method": None,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
    }
    return {
        **record,
        "facts_gold": facts,
        "filter_values": filter_values,
        "filter_sources": filter_sources,
        "visual_qualities": visual_qualities,
        "enriched_doc": compose_enriched_doc(record, facts, aggregated),
        "labeling_metadata": {
            **(record.get("labeling_metadata") or {}),
            "text_model": text_candidate.id,
        },
    }


def label_vision_record(
    record: dict[str, Any],
    vision_candidate: ModelCandidate,
    *,
    cost_tracker: CostTracker,
    confidence_mode: str = "self",
    agreement_k: int = 3,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    image_tokens_per_image: int = DEFAULT_VISION_IMAGE_TOKENS,
    vision_chunk_size: int = DEFAULT_VISION_CHUNK_SIZE,
) -> dict[str, Any]:
    previous_visual = record.get("visual_qualities") or {}
    text_imkanlar = (previous_visual.get("aggregated") or {}).get("imkanlar")
    visual_qualities = extract_visual_labels(
        record,
        vision_candidate,
        cost_tracker,
        text_imkanlar=text_imkanlar,
        confidence_mode=confidence_mode,
        agreement_k=agreement_k,
        min_confidence=min_confidence,
        image_tokens_per_image=image_tokens_per_image,
        vision_chunk_size=vision_chunk_size,
    )
    filter_values = visual_qualities.get("filter_values") or {}
    filter_sources = visual_qualities.get("filter_sources") or {}
    facts = dict(record.get("facts_gold") or {})
    visual_facts = merge_facts(record, {}, visual_qualities["aggregated"])
    for field_name, value in visual_facts.items():
        if facts.get(field_name) is None and value is not None:
            facts[field_name] = value
    facts = _facts_with_filter_values(record, facts, filter_values)
    return {
        **record,
        "facts_gold": facts,
        "filter_values": filter_values,
        "filter_sources": filter_sources,
        "visual_qualities": visual_qualities,
        "enriched_doc": compose_enriched_doc(record, facts, visual_qualities["aggregated"]),
        "labeling_metadata": {
            **(record.get("labeling_metadata") or {}),
            "vision_model": vision_candidate.id,
            "confidence_mode": visual_qualities.get("confidence_method"),
        },
    }


def label_record(
    record: dict[str, Any],
    text_candidate: ModelCandidate,
    vision_candidate: ModelCandidate,
    *,
    cost_tracker: CostTracker,
    confidence_mode: str = "self",
    agreement_k: int = 3,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    image_tokens_per_image: int = DEFAULT_VISION_IMAGE_TOKENS,
    vision_chunk_size: int = DEFAULT_VISION_CHUNK_SIZE,
) -> dict[str, Any]:
    text_record = label_text_record(record, text_candidate, cost_tracker=cost_tracker)
    return label_vision_record(
        text_record,
        vision_candidate,
        cost_tracker=cost_tracker,
        confidence_mode=confidence_mode,
        agreement_k=agreement_k,
        min_confidence=min_confidence,
        image_tokens_per_image=image_tokens_per_image,
        vision_chunk_size=vision_chunk_size,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_json_row(row: dict[str, Any]) -> dict[str, Any]:
    values = row.get("filter_values") or {}
    sources = row.get("filter_sources") or {}
    scraper_sources = {"scraper_info", "scraper_property_feature"}
    scraper = {
        slug: value for slug, value in values.items()
        if value is not None and sources.get(slug) in scraper_sources
    }
    deepseek = {
        slug: value for slug, value in values.items()
        if isinstance(value, bool) and sources.get(slug) == "deepseek_description"
    }
    kimi = {
        slug: value for slug, value in values.items()
        if value is True and sources.get(slug) == "kimi_image"
    }
    return {
        "id": row.get("id"),
        "url": row.get("url"),
        "title": row.get("title"),
        "description": row.get("description"),
        "scraper": scraper,
        "deepseek": deepseek,
        "kimi": kimi,
    }


def write_clean_json(source_path: Path, clean_json_path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(source_path) if source_path.exists() else []
    clean_rows = [clean_json_row(row) for row in rows]
    clean_json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = clean_json_path.with_suffix(clean_json_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(clean_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(clean_json_path)
    return clean_rows


def _refresh_clean_json(source_path: Path, clean_json_path: Path) -> None:
    write_clean_json(source_path, clean_json_path)


def _completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    for row in load_jsonl(output_path):
        if row.get("id") is not None:
            ids.add(str(row["id"]))
    return ids


def run_labeling(
    *,
    input_path: Path,
    output_path: Path,
    text_model_id: str,
    vision_model_id: str,
    batch_size: int,
    resume: bool,
    max_cost_usd: float | None,
    listing_ids: list[str] | None = None,
    confidence_mode: str = "self",
    agreement_k: int = 3,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    image_tokens_per_image: int = DEFAULT_VISION_IMAGE_TOKENS,
    vision_chunk_size: int = DEFAULT_VISION_CHUNK_SIZE,
    phase: str = "combined",
    cost_tracker: CostTracker | None = None,
    clean_json_path: Path | None = None,
) -> list[dict[str, Any]]:
    text_candidate = candidate_by_id(text_model_id) if phase != "vision" else None
    vision_candidate = candidate_by_id(vision_model_id) if phase != "text" else None
    records = load_jsonl(input_path)
    if listing_ids is not None:
        wanted = set(str(value) for value in listing_ids)
        order = {listing_id: index for index, listing_id in enumerate(listing_ids)}
        records = sorted((row for row in records if str(row.get("id")) in wanted), key=lambda row: order[str(row["id"])])

    done = _completed_ids(output_path) if resume else set()
    clean_json_path = clean_json_path or output_path.parent / DEFAULT_CLEAN_JSON_NAME
    if not resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    _refresh_clean_json(output_path, clean_json_path)
    pending = [record for record in records if str(record.get("id")) not in done]
    cost_tracker = cost_tracker or CostTracker(max_cost_usd)
    written: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def process(record: dict[str, Any]) -> dict[str, Any]:
        if phase == "text":
            return label_text_record(record, text_candidate, cost_tracker=cost_tracker)
        if phase == "vision":
            return label_vision_record(
                record,
                vision_candidate,
                cost_tracker=cost_tracker,
                confidence_mode=confidence_mode,
                agreement_k=agreement_k,
                min_confidence=min_confidence,
                image_tokens_per_image=image_tokens_per_image,
                vision_chunk_size=vision_chunk_size,
            )
        return label_record(
            record,
            text_candidate,
            vision_candidate,
            cost_tracker=cost_tracker,
            confidence_mode=confidence_mode,
            agreement_k=agreement_k,
            min_confidence=min_confidence,
            image_tokens_per_image=image_tokens_per_image,
            vision_chunk_size=vision_chunk_size,
        )

    if batch_size <= 1:
        for record in pending:
            row = process(record)
            _append_jsonl(output_path, row)
            _refresh_clean_json(output_path, clean_json_path)
            written.append(row)
        return written

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(process, record): record for record in pending}
        for future in as_completed(futures):
            row = future.result()
            with write_lock:
                _append_jsonl(output_path, row)
                _refresh_clean_json(output_path, clean_json_path)
                written.append(row)
    return written


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score_rows(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]], fields: tuple[str, ...], gold_key: str, pred_getter) -> dict[str, Any]:
    gold_by_id = {str(row["listing_id"]): row for row in gold_rows}
    sample_scores = []
    per_field_values: dict[str, list[float]] = {field_name: [] for field_name in fields}
    samples = []
    for prediction in predictions:
        listing_id = str(prediction.get("id"))
        gold = gold_by_id.get(listing_id, {}).get(gold_key) or {}
        predicted = pred_getter(prediction)
        scored = score_against_gold(predicted, gold, fields)
        if scored["accuracy"] is not None:
            sample_scores.append(scored["accuracy"])
        for field_name, detail in scored["per_field"].items():
            if detail["field_score"] is not None:
                per_field_values[field_name].append(detail["field_score"])
        samples.append({"listing_id": listing_id, **scored})
    return {
        "accuracy": _mean(sample_scores),
        "sample_count": len(predictions),
        "scored_samples": len(sample_scores),
        "per_field_accuracy": {field_name: _mean(values) for field_name, values in per_field_values.items()},
        "samples": samples,
    }


def score_preflight(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts_all = _score_rows(predictions, gold_rows, FACTS_GOLD_FIELDS, "facts_gold", lambda row: row.get("facts_gold") or {})
    facts_llm = _score_rows(predictions, gold_rows, TEXT_FACT_FIELDS, "facts_gold", lambda row: row.get("facts_gold") or {})
    visual = _score_rows(
        predictions,
        gold_rows,
        VISUAL_GOLD_FIELDS,
        "visual_gold",
        lambda row: (row.get("visual_qualities") or {}).get("aggregated") or {},
    )
    # Gate, tam birleştirilmiş pipeline (structured + vision + text) çıktısını ölçen
    # facts_all'a bağlı. facts_llm sadece teşhis: text-only model fotoğraftan dolan
    # HYBRID alanlarda (has_balcony=false vb.) null üretir → null-vs-false yüzünden
    # yapısal olarak 0 alır; bu metrik kapı olarak kullanılırsa güçlü modeli haksızca eler.
    facts_accuracy = facts_all["accuracy"] if facts_all["accuracy"] is not None else 0.0
    visual_accuracy = visual["accuracy"] if visual["accuracy"] is not None else 0.0
    return {
        "facts_all": facts_all,
        "facts_llm": facts_llm,
        "visual": visual,
        "thresholds": {"facts": FACTS_THRESHOLD, "visual": VISUAL_THRESHOLD},
        "passes_thresholds": facts_accuracy >= FACTS_THRESHOLD and visual_accuracy >= VISUAL_THRESHOLD,
    }


def _load_selected(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["text_model"], data["vision_model"]


def _validate_environment(text_model_id: str, vision_model_id: str, *, phase: str = "combined") -> None:
    model_ids = [text_model_id, vision_model_id]
    if phase == "text":
        model_ids = [text_model_id]
    elif phase == "vision":
        model_ids = [vision_model_id]
    missing = sorted(set(value for model_id in model_ids for value in missing_environment(candidate_by_id(model_id))))
    if missing:
        raise RuntimeError(f"Missing environment for selected models: {', '.join(missing)}")


def _validate_full_batch_guard(args: argparse.Namespace) -> None:
    if not args.preflight_gold and not args.allow_full_batch:
        raise RuntimeError("Full batch is intentionally blocked. Re-run with --allow-full-batch only after top-up and user approval.")


def _records_for_ids(input_path: Path, listing_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(str(value) for value in listing_ids)
    order = {listing_id: index for index, listing_id in enumerate(listing_ids)}
    return sorted((row for row in load_jsonl(input_path) if str(row.get("id")) in wanted), key=lambda row: order[str(row["id"])])


def calibrate_confidence(
    *,
    input_path: Path,
    gold_path: Path,
    listing_ids: list[str],
    text_model_id: str,
    vision_model_id: str,
    cost_tracker: CostTracker,
    agreement_k: int,
    min_confidence: float,
    image_tokens_per_image: int,
    vision_chunk_size: int,
) -> dict[str, Any]:
    text_candidate = candidate_by_id(text_model_id)
    vision_candidate = candidate_by_id(vision_model_id)
    records = _records_for_ids(input_path, listing_ids)
    gold_rows = [row for row in load_gold_rows(gold_path) if str(row["listing_id"]) in set(listing_ids)]
    mode_reports: dict[str, Any] = {}
    for mode in ("self", "agreement"):
        predictions = [
            label_record(
                record,
                text_candidate,
                vision_candidate,
                cost_tracker=cost_tracker,
                confidence_mode=mode,
                agreement_k=agreement_k,
                min_confidence=min_confidence,
                image_tokens_per_image=image_tokens_per_image,
                vision_chunk_size=vision_chunk_size,
            )
            for record in records
        ]
        mode_reports[mode] = score_preflight(predictions, gold_rows)["visual"]
    self_accuracy = mode_reports["self"]["accuracy"] or 0.0
    agreement_accuracy = mode_reports["agreement"]["accuracy"] or 0.0
    selected = "agreement" if agreement_accuracy > self_accuracy else "self"
    return {
        "sample_listing_ids": listing_ids,
        "modes": mode_reports,
        "selected_mode": selected,
    }


def run_preflight(args: argparse.Namespace, text_model_id: str, vision_model_id: str) -> dict[str, Any]:
    gold_path = Path(args.preflight_gold)
    selected_ids = gold_listing_ids(gold_path, limit=args.preflight_limit)
    output_path = Path(args.output)
    cost_tracker = CostTracker(args.max_cost_usd)
    confidence_calibration = None
    confidence_mode = args.confidence_mode
    if confidence_mode == "auto":
        calibration_ids = selected_ids[: args.confidence_calibration_size]
        confidence_calibration = calibrate_confidence(
            input_path=Path(args.input),
            gold_path=gold_path,
            listing_ids=calibration_ids,
            text_model_id=text_model_id,
            vision_model_id=vision_model_id,
            cost_tracker=cost_tracker,
            agreement_k=args.agreement_k,
            min_confidence=args.min_confidence,
            image_tokens_per_image=args.vision_image_tokens,
            vision_chunk_size=args.vision_chunk_size,
        )
        confidence_mode = confidence_calibration["selected_mode"]
    predictions = run_labeling(
        input_path=Path(args.input),
        output_path=output_path,
        text_model_id=text_model_id,
        vision_model_id=vision_model_id,
        batch_size=args.batch_size,
        resume=args.resume,
        max_cost_usd=args.max_cost_usd,
        listing_ids=selected_ids,
        confidence_mode=confidence_mode,
        agreement_k=args.agreement_k,
        min_confidence=args.min_confidence,
        image_tokens_per_image=args.vision_image_tokens,
        vision_chunk_size=args.vision_chunk_size,
        cost_tracker=cost_tracker,
    )
    if args.resume:
        predictions = [row for row in load_jsonl(output_path) if str(row.get("id")) in set(selected_ids)]
    gold_rows = [row for row in load_gold_rows(gold_path) if str(row["listing_id"]) in set(selected_ids)]
    report = score_preflight(predictions, gold_rows)
    report["listing_ids"] = selected_ids
    report["output_path"] = str(output_path)
    report["confidence_mode"] = confidence_mode
    report["confidence_calibration"] = confidence_calibration
    report["estimated_cost_usd"] = cost_tracker.estimated_usd
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label listings with text LLM + multi-image VLM")
    parser.add_argument("--input", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--phase", choices=("combined", "text", "vision"), default="combined")
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED_PATH))
    parser.add_argument("--preflight-gold", default=None)
    parser.add_argument("--preflight-limit", type=int, default=10)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-full-batch", action="store_true")
    parser.add_argument("--confidence-mode", choices=("self", "agreement", "auto"), default="self")
    parser.add_argument("--confidence-calibration-size", type=int, default=1)
    parser.add_argument("--agreement-k", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--vision-image-tokens", type=int, default=DEFAULT_VISION_IMAGE_TOKENS)
    parser.add_argument("--vision-chunk-size", type=int, default=DEFAULT_VISION_CHUNK_SIZE,
                        help="Bir ilanın fotoğraflarını kaç'arlı VLM çağrısına böl (timeout azaltır). 0 = tek çağrı.")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args()
    selected_text_model, selected_vision_model = _load_selected(Path(args.selected))
    text_model_id = args.text_model or selected_text_model
    vision_model_id = args.vision_model or selected_vision_model
    _validate_environment(text_model_id, vision_model_id, phase=args.phase)
    _validate_full_batch_guard(args)

    if args.preflight_gold:
        report = run_preflight(args, text_model_id, vision_model_id)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passes_thresholds"]:
            raise SystemExit(2)
        return

    rows = run_labeling(
        input_path=Path(args.input),
        output_path=Path(args.output),
        text_model_id=text_model_id,
        vision_model_id=vision_model_id,
        batch_size=args.batch_size,
        resume=args.resume,
        max_cost_usd=args.max_cost_usd,
        confidence_mode=args.confidence_mode,
        agreement_k=args.agreement_k,
        min_confidence=args.min_confidence,
        image_tokens_per_image=args.vision_image_tokens,
        vision_chunk_size=args.vision_chunk_size,
        phase=args.phase,
    )
    print(json.dumps({"written": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
