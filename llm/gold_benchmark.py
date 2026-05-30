from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = PROJECT_ROOT / "labeling" / "gold_listings_manual_todo.jsonl"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.jsonl"
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "listings.jsonl"

GOLD_BENCHMARK_LISTING_COUNT = 16
GOLD_BENCHMARK_PHOTO_COUNT = 12

FACTS_GOLD_FIELDS = (
    # --- STRUCTURED (scraper "İlan Bilgileri" tablosundan kesin) ---
    "city",
    "district",
    "neighborhood",
    "price_tl",
    "room_count",
    "gross_size_m2",
    "net_size_m2",
    "building_age",
    "floor",
    "total_floors",
    "deposit_tl",
    "in_gated_complex",
    "title_deed_status",
    "heating_type",
    "is_furnished",
    # --- HYBRID (features/title + description LLM + visual VLM fallback) ---
    "has_balcony",
    "has_elevator",
    "has_parking",
    "has_aircon",
    # --- DESCRIPTION-ONLY (LLM extraction) ---
    "near_metro",
    "near_metrobus",
)

STRUCTURED_FACT_FIELDS = (
    "city",
    "district",
    "neighborhood",
    "price_tl",
    "room_count",
    "gross_size_m2",
    "net_size_m2",
    "building_age",
    "floor",
    "total_floors",
    "deposit_tl",
    "in_gated_complex",
    "title_deed_status",
    "heating_type",
    "is_furnished",
)

VISUAL_GOLD_FIELDS = (
    "balkon_ozellikleri",
    "manzara",
    "mutfak_tipi",
    "banyo_ozellikleri",
    "imkanlar",
)

MULTI_SELECT_FIELDS = {
    "balkon_ozellikleri",
    "manzara",
    "banyo_ozellikleri",
    "imkanlar",
}

# visual_gold enum sözlüğü — single source of truth.
# Manuel gold doldurma için prefilled template + scoring için referans.
# single-select: mutfak_tipi, zemin_tipi. Diğerleri multi-select.
VISUAL_ENUMS = {
    "balkon_ozellikleri": ["cam_balkon", "acik_balkon", "fransiz_balkon", "cikma_balkon", "teras"],
    "manzara": ["deniz", "bogaz", "orman_yesil", "park", "sehir_panorama", "dag", "ic_avlu", "komsu_duvari"],
    "mutfak_tipi": ["amerikan_acik", "kapali_ayri"],
    "banyo_ozellikleri": ["dusakabin", "kuvet", "jakuzi", "banyoda_pencere", "birden_fazla_banyo"],
    "imkanlar": ["havuz", "yesil_alan_peyzaj", "guvenlik_kabini", "kapali_otopark", "acik_otopark", "cocuk_parki", "spor_alani"],
}


def build_prefilled_visual_gold() -> dict[str, Any]:
    """Gold template için visual_gold prefill:
    - multi-select alanlar: tüm enum array içinde (kullanıcı olmayanları siler)
    - single-select alanlar: '|'li string (kullanıcı tek değer bırakır)
    """
    prefilled: dict[str, Any] = {}
    for field in VISUAL_GOLD_FIELDS:
        values = VISUAL_ENUMS[field]
        if field in MULTI_SELECT_FIELDS:
            prefilled[field] = list(values)
        else:
            prefilled[field] = " | ".join(values)
    return prefilled


# facts_gold HYBRID prefill (kullanıcı manuel doldurma kolaylığı):
HYBRID_PREFILL_ENUMS = {
    "has_balcony": ["true", "false"],
    "has_elevator": ["true", "false"],
    "has_parking": ["true", "false"],
    "has_aircon": ["true", "false"],
    "near_metro": ["true", "false"],
    "near_metrobus": ["true", "false"],
}


def build_prefilled_hybrid_facts() -> dict[str, str]:
    """Gold template için facts_gold HYBRID prefill — hepsi '|'li string.
    Kullanıcı: tek değer bırak, quote'lar opsiyonel (normalize_gold_value true/false→bool çevirir).
    """
    return {field: " | ".join(values) for field, values in HYBRID_PREFILL_ENUMS.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold_rows(path: Path = DEFAULT_GOLD_PATH) -> list[dict[str, Any]]:
    return load_jsonl(path)


def gold_listing_ids(path: Path = DEFAULT_GOLD_PATH, limit: int = GOLD_BENCHMARK_LISTING_COUNT) -> list[str]:
    return [row["listing_id"] for row in load_gold_rows(path)[:limit]]


def load_dataset_index(path: Path = DEFAULT_DATASET_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def load_raw_index(path: Path = DEFAULT_RAW_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in load_jsonl(path)}


def listing_description(listing_id: str, dataset_index: dict[str, dict[str, Any]], raw_index: dict[str, dict[str, Any]]) -> str:
    # Production parity: cleaned dataset description (MAX_DESCRIPTION_LEN ile kırpılmış) öncelikli.
    # Raw description sadece dataset'te yoksa fallback — shootout production input boyutuyla aynı olmalı.
    record = dataset_index.get(listing_id, {})
    cleaned = str(record.get("description") or record.get("text") or "").strip()
    if cleaned:
        return cleaned
    raw = raw_index.get(listing_id)
    if raw and raw.get("description"):
        return str(raw["description"]).strip()
    return ""


def listing_image_paths(record: dict[str, Any], max_photos: int | None = GOLD_BENCHMARK_PHOTO_COUNT) -> list[str]:
    paths = record.get("all_image_paths") or []
    if not paths and record.get("image_path"):
        paths = [record["image_path"]]
    # max_photos=None -> tüm fotoğraflar (paths[:None] tüm listeyi döndürür)
    return [str(Path(str(path).replace("\\", "/"))) for path in paths[:max_photos]]


def gold_is_filled(gold: dict[str, Any] | None) -> bool:
    if not gold:
        return False
    return any(value is not None for value in gold.values())


# Türkçe → ASCII fold: VLM "boğaz"/"açık_otopark" üretebilir ama enum + gold ASCII
# ("bogaz"/"acik_otopark"). translate ÖNCE (İ→I), sonra lower (Python "İ".lower() bug guard).
_TR_ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜI", "cgiosuaiucgiosui")


def _fold(value: Any) -> str:
    return str(value).strip().translate(_TR_ASCII).lower()


def normalize_gold_value(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(_fold(item) for item in value)
    if isinstance(value, str):
        s = _fold(value)
        # Boolean string'leri Python bool'a normalize et (kullanıcı "true"/"false" yazsa da çalışsın)
        if s == "true":
            return True
        if s == "false":
            return False
        if s in ("null", "none", ""):
            return None
        return s
    return value


def _list_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {_fold(item) for item in value if _fold(item)}
    if _fold(value) == "":
        return set()
    return {_fold(value)}


def field_score(predicted: Any, gold: Any, *, multi_select: bool = False) -> float | None:
    if gold is None:
        return None
    if multi_select:
        predicted_set = _list_set(predicted)
        gold_set = _list_set(gold)
        if not predicted_set and not gold_set:
            return 1.0
        union = predicted_set | gold_set
        return len(predicted_set & gold_set) / len(union) if union else 1.0
    return 1.0 if normalize_gold_value(predicted) == normalize_gold_value(gold) else 0.0


def field_matches(predicted: Any, gold: Any) -> bool | None:
    score = field_score(predicted, gold)
    return None if score is None else score == 1.0


def score_against_gold(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """Score predictions against gold labels.

    Fields with gold value null are skipped. Multi-select fields use Jaccard
    score |intersection| / |union|; if both gold and prediction are empty,
    the field score is 1. Confidence is intentionally ignored for M3.
    """
    per_field: dict[str, Any] = {}
    scored = 0
    correct = 0
    score_sum = 0.0
    for field in fields:
        gold_value = gold.get(field)
        score = field_score(
            predicted.get(field),
            gold_value,
            multi_select=field in MULTI_SELECT_FIELDS,
        )
        if score is None:
            per_field[field] = {"gold": None, "predicted": predicted.get(field), "match": None, "field_score": None}
            continue
        match = score == 1.0
        per_field[field] = {
            "gold": gold_value,
            "predicted": predicted.get(field),
            "match": match,
            "field_score": score,
        }
        scored += 1
        correct += int(match)
        score_sum += score
    return {
        "per_field": per_field,
        "scored_fields": scored,
        "correct_fields": correct,
        "accuracy": (score_sum / scored) if scored else None,
    }


def aggregate_model_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # "sample_count" (not "samples") — shootout sonuçlarında "samples" detay listesi
    # **aggregate ile ezilmesin diye ayrı isim kullanılır.
    scored = [row["accuracy"] for row in rows if row.get("accuracy") is not None]
    if not scored:
        return {"accuracy": None, "sample_count": len(rows), "scored_samples": 0}
    return {
        "accuracy": sum(scored) / len(scored),
        "sample_count": len(rows),
        "scored_samples": len(scored),
    }
