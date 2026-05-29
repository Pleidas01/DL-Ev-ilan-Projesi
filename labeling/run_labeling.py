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
    load_gold_rows,
    score_against_gold,
)
from llm.shootout_vision import VISION_SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "labeled.jsonl"
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "llm" / "selected.json"

TEXT_FACT_FIELDS = tuple(field for field in FACTS_GOLD_FIELDS if field not in STRUCTURED_FACT_FIELDS)
VISION_FIELDS = VISUAL_GOLD_FIELDS

FACTS_THRESHOLD = 0.75
VISUAL_THRESHOLD = 0.70
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_VISION_IMAGE_TOKENS = 60
DEFAULT_TEXT_OUTPUT_TOKENS = 350
DEFAULT_VISION_OUTPUT_TOKENS = 700
PROVIDER_MAX_RETRIES = 3

TEXT_SYSTEM_PROMPT = """Sen Turkce emlak ilani metninden kesin JSON alanlari cikaran bir asistansin.
Sadece gecerli JSON dondur. Aciklama yazma."""


def build_text_prompt(record: dict[str, Any]) -> str:
    features = record.get("property_features") or (record.get("attributes") or {}).get("propertyFeatures") or []
    existing = {field: record.get(field) for field in TEXT_FACT_FIELDS}
    return f"""Ilan basligi:
{record.get("title") or ""}

Ilan aciklamasi:
{record.get("description") or record.get("text") or ""}

Property features:
{json.dumps(features, ensure_ascii=False)}

Mevcut structured/keyword ipuclari:
{json.dumps(existing, ensure_ascii=False)}

JSON semasi:
{{
  "facts": {{
    "has_balcony": null,
    "has_elevator": null,
    "has_parking": null,
    "has_aircon": null,
    "near_metro": null,
    "near_metrobus": null
  }},
  "imkanlar": []
}}

Kurallar:
- heating_type ve is_furnished dondurme; bunlar structured alandir.
- has_* alanlari icin true yalnizca metin/features/title acik kanit veriyorsa.
- near_metro ve near_metrobus icin true yalnizca yurume mesafesi veya <15 dakika anlami varsa.
- Bilinmeyen alanlar icin null kullan.
- imkanlar sadece su enumlardan liste olsun: havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani."""


def build_vision_prompt(image_count: int) -> str:
    return f"""Bu {image_count} fotografin her birini sira numarasina gore etiketle ve sonra aggregate et.
Fotografta gorunmeyen alanlar icin null kullan. imkanlar hibrittir: fotografta acikca gorunen havuz, spor alani, cocuk parki, otopark, guvenlik kabini veya yesil alan varsa etiketle; sadece ic mekan fotografindan site imkani tahmin etme.

ONEMLI: Degerlerde Turkce karakter KULLANMA. Sadece asagidaki enum degerlerini aynen kullan.

JSON semasi:
{{
  "per_image": [
    {{
      "image_index": 0,
      "fields": {{
        "balkon_ozellikleri": null,
        "manzara": null,
        "mutfak_tipi": null,
        "banyo_ozellikleri": null,
        "salon_ozellikleri": null,
        "imkanlar": null
      }},
      "confidence": 0.0
    }}
  ],
  "aggregated": {{
    "balkon_ozellikleri": null,
    "manzara": null,
    "mutfak_tipi": null,
    "banyo_ozellikleri": null,
    "salon_ozellikleri": null,
    "imkanlar": null
  }}
}}

Enumlar:
- balkon_ozellikleri (liste): cam_balkon, acik_balkon, fransiz_balkon, cikma_balkon, teras
- manzara (liste): deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari
- mutfak_tipi (tek deger): amerikan_acik | kapali_ayri
- banyo_ozellikleri (liste): dusakabin, kuvet, jakuzi, banyoda_pencere, birden_fazla_banyo
- salon_ozellikleri (liste): somine, nis, acik_plan_genis, ayri_yemek_alani
- imkanlar (liste): havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani

Confidence 0 ile 1 arasinda sayi olmali. Belirsiz veya uzaktan gorunen alanlarda confidence dusuk ver."""


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


def normalize_text_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    facts_source = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else parsed
    facts = {field: _coerce_bool(facts_source.get(field)) for field in TEXT_FACT_FIELDS}
    imkanlar = _coerce_enum_list("imkanlar", parsed.get("imkanlar"))
    return {**facts, "imkanlar": imkanlar or []}


def normalize_visual_fields(raw_fields: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    source = raw_fields if isinstance(raw_fields, dict) else {}
    normalized: dict[str, Any] = {}
    for field_name in fields:
        if field_name in MULTI_SELECT_FIELDS:
            normalized[field_name] = _coerce_enum_list(field_name, source.get(field_name))
        else:
            normalized[field_name] = _coerce_enum(field_name, source.get(field_name))
    return normalized


def merge_facts(record: dict[str, Any], text_prediction: dict[str, Any]) -> dict[str, Any]:
    facts = {field: record.get(field) for field in STRUCTURED_FACT_FIELDS}
    for field_name in TEXT_FACT_FIELDS:
        existing_value = record.get(field_name)
        facts[field_name] = existing_value if existing_value is not None else text_prediction.get(field_name)
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
        fields = normalize_visual_fields(parsed.get("aggregated") if isinstance(parsed.get("aggregated"), dict) else parsed, VISION_FIELDS)
        per_image.append({
            "path": None,
            "image_paths": image_paths,
            "fields": fields,
            "confidence": _item_confidence({}, parsed),
        })

    aggregated = _aggregate_from_items(per_image, min_confidence)
    raw_aggregated = parsed.get("aggregated") if isinstance(parsed.get("aggregated"), dict) else parsed
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
    return normalize_text_prediction(_parse_json_object(raw))


def _resolve_image_paths(record: dict[str, Any]) -> list[str]:
    paths = record.get("all_image_paths") or []
    if not paths and record.get("image_path"):
        paths = [record["image_path"]]
    resolved: list[str] = []
    for value in paths:
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
) -> dict[str, Any]:
    user_prompt = build_vision_prompt(len(image_paths))
    cost_tracker.reserve(
        _estimate_vision_call(candidate, user_prompt, len(image_paths), image_tokens_per_image),
        "vision labeling",
    )
    raw = _provider_json_call(complete_vision_json, candidate, VISION_SYSTEM_PROMPT, user_prompt, image_paths)
    return _parse_json_object(raw)


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
) -> dict[str, Any]:
    image_paths = _resolve_image_paths(record)
    if not image_paths:
        aggregated = _empty_visual_aggregate()
        _merge_imkanlar(aggregated, text_imkanlar, None)
        return {"per_image": [], "aggregated": aggregated, "confidence_method": confidence_mode, "min_confidence": min_confidence}

    if confidence_mode == "agreement":
        run_aggregates = []
        for _ in range(agreement_k):
            parsed = _call_vision_once(image_paths, candidate, cost_tracker, image_tokens_per_image)
            run_aggregates.append(
                aggregate_visual_qualities(
                    parsed,
                    image_paths=image_paths,
                    text_imkanlar=[],
                    min_confidence=min_confidence,
                    confidence_method="self",
                )["aggregated"]
            )
        return _agreement_aggregate(run_aggregates, text_imkanlar=text_imkanlar, min_confidence=min_confidence)

    parsed = _call_vision_once(image_paths, candidate, cost_tracker, image_tokens_per_image)
    return aggregate_visual_qualities(
        parsed,
        image_paths=image_paths,
        text_imkanlar=text_imkanlar,
        min_confidence=min_confidence,
        confidence_method="self",
    )


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
) -> dict[str, Any]:
    text_prediction = extract_text_labels(record, text_candidate, cost_tracker)
    facts = merge_facts(record, text_prediction)
    visual_qualities = extract_visual_labels(
        record,
        vision_candidate,
        cost_tracker,
        text_imkanlar=text_prediction.get("imkanlar"),
        confidence_mode=confidence_mode,
        agreement_k=agreement_k,
        min_confidence=min_confidence,
        image_tokens_per_image=image_tokens_per_image,
    )
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "url": record.get("url"),
        "price_tl": record.get("price_tl"),
        "facts_gold": facts,
        "visual_qualities": visual_qualities,
        "enriched_doc": compose_enriched_doc(record, facts, visual_qualities["aggregated"]),
        "labeling_metadata": {
            "text_model": text_candidate.id,
            "vision_model": vision_candidate.id,
            "confidence_mode": visual_qualities.get("confidence_method"),
        },
    }


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
    cost_tracker: CostTracker | None = None,
) -> list[dict[str, Any]]:
    text_candidate = candidate_by_id(text_model_id)
    vision_candidate = candidate_by_id(vision_model_id)
    records = load_jsonl(input_path)
    if listing_ids is not None:
        wanted = set(str(value) for value in listing_ids)
        order = {listing_id: index for index, listing_id in enumerate(listing_ids)}
        records = sorted((row for row in records if str(row.get("id")) in wanted), key=lambda row: order[str(row["id"])])

    done = _completed_ids(output_path) if resume else set()
    if not resume and output_path.exists():
        output_path.write_text("", encoding="utf-8")
    pending = [record for record in records if str(record.get("id")) not in done]
    cost_tracker = cost_tracker or CostTracker(max_cost_usd)
    written: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def process(record: dict[str, Any]) -> dict[str, Any]:
        return label_record(
            record,
            text_candidate,
            vision_candidate,
            cost_tracker=cost_tracker,
            confidence_mode=confidence_mode,
            agreement_k=agreement_k,
            min_confidence=min_confidence,
            image_tokens_per_image=image_tokens_per_image,
        )

    if batch_size <= 1:
        for record in pending:
            row = process(record)
            _append_jsonl(output_path, row)
            written.append(row)
        return written

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(process, record): record for record in pending}
        for future in as_completed(futures):
            row = future.result()
            with write_lock:
                _append_jsonl(output_path, row)
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
    facts_accuracy = facts_llm["accuracy"] if facts_llm["accuracy"] is not None else 0.0
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


def _validate_environment(text_model_id: str, vision_model_id: str) -> None:
    missing = sorted(set(missing_environment(candidate_by_id(text_model_id)) + missing_environment(candidate_by_id(vision_model_id))))
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
    _validate_environment(text_model_id, vision_model_id)
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
    )
    print(json.dumps({"written": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
