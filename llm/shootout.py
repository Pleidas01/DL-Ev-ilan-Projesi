from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.clients import CANDIDATES, candidate_by_id, complete_json, estimate_cost_usd, missing_environment
from schema.emlakjet_filters import EMLAKJET_FILTERS


SLOT_SYSTEM_PROMPT = """Sen Turkce emlak arama sorgularini JSON slotlarina ayiran bir asistansin.
Sadece gecerli JSON dondur. Aciklama yazma."""


def _empty_filter_values() -> dict[str, Any]:
    return {spec.slug: None for spec in EMLAKJET_FILTERS}


FEW_SHOT_SLOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "query": "Kadikoy'de 30 bin alti 1+1 ogrenciye uygun daire",
        "output": {
            "hard_filters": {
                "filters": {"district": ["Kadikoy"], "room_count": ["1+1"], "price_currency": "TL"},
                "max_price_amount": 30000,
            },
            "free_form_tr": "Kadikoy'de 30 bin alti 1+1 ogrenciye uygun daire",
        },
    },
    {
        "query": "metroya yakin esyali 1+1 kiralik",
        "output": {
            "hard_filters": {"filters": {"room_count": ["1+1"], "near_metro": True, "is_furnished": True}},
            "free_form_tr": "metroya yakin esyali 1+1 kiralik",
        },
    },
    {
        "query": "genis salonlu denize yakin 2+1 asansorlu ev",
        "output": {
            "hard_filters": {"filters": {"room_count": ["2+1"], "has_elevator": True, "has_sea_view": True}},
            "free_form_tr": "genis salonlu denize yakin 2+1 asansorlu ev",
        },
    },
]


# Canonical registry alanlarıyla slot benchmark. Generic "otoparklı" gibi birden
# fazla typed alana karşılık gelebilen ifadeler prompt'ta any_of OR grubu olarak
# öğretilir; benchmark expected listesi yalnız tek anlamlı slotları puanlar.
BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {"query": "Kadikoy'de 30 bin alti 1+1 ogrenciye uygun daire", "expected": {"room_count": ["1+1"], "district": ["Kadikoy"], "max_price_amount": 30000}},
    {"query": "genis salonlu denize yakin 2+1 asansorlu ev", "expected": {"room_count": ["2+1"], "has_elevator": True, "has_sea_view": True}},
    {"query": "Besiktas 3+1 60 bin TL alti site icinde", "expected": {"room_count": ["3+1"], "district": ["Besiktas"], "max_price_amount": 60000, "in_gated_complex": True}},
    {"query": "metroya yakin esyali 1+1 kiralik", "expected": {"room_count": ["1+1"], "near_metro": True, "is_furnished": True}},
    {"query": "amerikan mutfakli modern 2+1", "expected": {"room_count": ["2+1"], "has_american_kitchen": True}},
    {"query": "otoparkli guvenlikli aileye uygun 3+1", "expected": {"room_count": ["3+1"], "has_security": True}},
    {"query": "Bostanci sahile yakin deniz manzarali daire", "expected": {"district": ["Bostanci"], "has_sea_view": True}},
    {"query": "35 bin alti kombili 2+1", "expected": {"room_count": ["2+1"], "max_price_amount": 35000}},
    {"query": "parke zeminli ev", "expected": {"has_parquet_floor": True}},
    {"query": "cocuklu aile icin site icinde guvenlikli 4+1", "expected": {"room_count": ["4+1"], "in_gated_complex": True, "has_security": True, "has_playground": True}},
    {"query": "Pendik'te 25 bin TL alti 1+1", "expected": {"room_count": ["1+1"], "district": ["Pendik"], "max_price_amount": 25000}},
    {"query": "klimali esyali studyo ya da 1+1", "expected": {"room_count": ["1+1"], "has_aircon": True, "is_furnished": True}},
    {"query": "terasli 3+1 daire", "expected": {"room_count": ["3+1"], "balcony_type": ["acik_teras"]}},
    {"query": "kapali otoparkli luks rezidans", "expected": {"has_closed_parking": True}},
    {"query": "cam balkonlu 2+1", "expected": {"room_count": ["2+1"], "has_balcony": True}},
    {"query": "bahce cikisli kiralik", "expected": {"has_garden": True}},
    {"query": "Suadiye'de deniz goren daire", "expected": {"district": ["Suadiye"], "has_sea_view": True}},
    {"query": "Basaksehir site icinde 4+1 otoparkli", "expected": {"room_count": ["4+1"], "district": ["Basaksehir"], "in_gated_complex": True}},
    {"query": "asansorlu dogalgazli uygun fiyatli 2+1", "expected": {"room_count": ["2+1"], "has_elevator": True}},
    {"query": "amerikan mutfakli 1+1 daire", "expected": {"room_count": ["1+1"], "has_american_kitchen": True}},
    {"query": "manzarasi guzel yuksek kat 3+1", "expected": {"room_count": ["3+1"], "has_city_view": True}},
    {"query": "merkezi isitma buyuk salonlu ev", "expected": {}},
    {"query": "ogrenci icin metro yakini ucuz 1+1", "expected": {"room_count": ["1+1"], "near_metro": True}},
    {"query": "laminat zeminli 2+1", "expected": {"room_count": ["2+1"], "has_laminate_floor": True}},
    {"query": "bahce kati sakin guvenlikli site", "expected": {"in_gated_complex": True, "has_security": True}},
    {"query": "Avcilar 3+1 50 bin alti", "expected": {"room_count": ["3+1"], "district": ["Avcilar"], "max_price_amount": 50000}},
    {"query": "deniz manzarali balkonlu ferah daire", "expected": {"has_sea_view": True, "has_balcony": True}},
    {"query": "ankastreli parke zeminli yeni ev", "expected": {"has_builtin_kitchen": True, "has_parquet_floor": True}},
    {"query": "site icinde otoparkli guvenlikli 2+1", "expected": {"room_count": ["2+1"], "in_gated_complex": True, "has_security": True}},
    {"query": "Kadikoy sahile yakin 2+1 maksimum 45 bin", "expected": {"room_count": ["2+1"], "district": ["Kadikoy"], "max_price_amount": 45000}},
]


def build_slot_prompt(query: str, *, include_few_shot: bool = True) -> str:
    sections: list[str] = []
    if include_few_shot:
        for index, example in enumerate(FEW_SHOT_SLOT_EXAMPLES, start=1):
            sections.append(
                f"Ornek {index}:\n"
                f"Sorgu: {example['query']}\n"
                f"Beklenen cikti:\n{json.dumps(example['output'], ensure_ascii=False, indent=2)}"
            )
        sections.append("---")
    schema = {
        "hard_filters": {
            "filters": _empty_filter_values(),
            "any_of": [],
            "min_price_amount": None,
            "max_price_amount": None,
            "min_gross_size_m2": None,
            "max_gross_size_m2": None,
        },
        "free_form_tr": query,
    }
    sections.append(
        f"Sorgu: {query}\n\nBeklenen JSON semasi:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Kurallar:\n"
        "- Kullanicinin acikca istedigi her filtreyi hard_filters icine koy.\n"
        "- Tek canonical alana indirgenemeyen acik talepleri any_of OR grubu yap. Ornek: otoparkli -> has_open_parking veya has_closed_parking.\n"
        "- Istek olmayan filtreleri null birak."
    )
    return "\n\n".join(sections)


def flatten_slots(parsed: dict[str, Any]) -> dict[str, Any]:
    hard = parsed.get("hard_filters") or {}
    filters = hard.get("filters") or {}
    soft = parsed.get("soft_features") or {}
    image = soft.get("visual_gold") or soft.get("image") or {}
    facts = soft.get("facts_gold") or soft.get("text_extracted") or {}
    return {**hard, **filters, **image, **facts}


def score_json_adherence(raw_response: str) -> tuple[float, dict[str, Any] | None]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return 0.0, None
    required = ["hard_filters", "free_form_tr"]
    return sum(key in parsed for key in required) / len(required), parsed


def _actual_values_for_list_match(actual: Any) -> set[str]:
    if actual is None:
        return set()
    if isinstance(actual, list):
        return {str(value) for value in actual}
    if isinstance(actual, (str, int, float, bool)):
        return {str(actual)}
    try:
        return {str(value) for value in actual}
    except TypeError:
        return {str(actual)}


def score_expected_slots(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    flat = flatten_slots(parsed)
    if not expected:
        return 1.0
    hits = 0
    for key, expected_value in expected.items():
        actual = flat.get(key)
        if isinstance(expected_value, list):
            expected_set = {str(value) for value in expected_value}
            actual_set = _actual_values_for_list_match(actual)
            hits += int(bool(actual_set) and expected_set.issubset(actual_set))
        else:
            hits += int(actual == expected_value)
    return hits / len(expected)


def choose_winners(rows: list[dict[str, Any]], max_cost_100k_usd: float = 500) -> dict[str, str]:
    feasible = [row for row in rows if row["cost_100k_usd"] <= max_cost_100k_usd]
    if not feasible:
        raise ValueError("No feasible model rows under the 100K listing cost threshold")

    text_pool = [row for row in feasible if not row["supports_vision"]] or feasible
    vision_pool = [row for row in feasible if row["supports_vision"]]
    if not vision_pool:
        raise ValueError("No feasible vision-capable model row")

    def sort_key(row: dict[str, Any]) -> tuple[float, float]:
        return (row["quality_score"], -row["cost_100k_usd"])

    return {
        "text_model": max(text_pool, key=sort_key)["model_id"],
        "vision_model": max(vision_pool, key=sort_key)["model_id"],
    }


def run_text_slot_benchmark(model_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model_id in model_ids:
        candidate = candidate_by_id(model_id)
        missing = missing_environment(candidate)
        if missing:
            rows.append({
                "model_id": model_id,
                "supports_vision": candidate.supports_vision,
                "quality_score": 0.0,
                "json_score": 0.0,
                "slot_score": 0.0,
                "cost_100k_usd": estimate_cost_usd(candidate, 700 * 100_000, 200 * 100_000),
                "status": f"missing_env:{','.join(missing)}",
            })
            continue

        json_scores = []
        slot_scores = []
        try:
            for item in BENCHMARK_QUERIES:
                raw = complete_json(candidate, SLOT_SYSTEM_PROMPT, build_slot_prompt(item["query"]))
                json_score, parsed = score_json_adherence(raw)
                json_scores.append(json_score)
                slot_scores.append(score_expected_slots(parsed, item["expected"]) if parsed else 0.0)
        except Exception as exc:
            rows.append({
                "model_id": model_id,
                "supports_vision": candidate.supports_vision,
                "quality_score": 0.0,
                "json_score": 0.0,
                "slot_score": 0.0,
                "cost_100k_usd": estimate_cost_usd(candidate, 700 * 100_000, 200 * 100_000),
                "status": f"error:{type(exc).__name__}:{exc}",
            })
            continue

        json_avg = sum(json_scores) / len(json_scores)
        slot_avg = sum(slot_scores) / len(slot_scores)
        rows.append({
            "model_id": model_id,
            "supports_vision": candidate.supports_vision,
            "quality_score": 0.4 * json_avg + 0.6 * slot_avg,
            "json_score": json_avg,
            "slot_score": slot_avg,
            "cost_100k_usd": estimate_cost_usd(candidate, 700 * 100_000, 200 * 100_000),
            "status": "ok",
        })
    return rows


def write_selected(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    winners = choose_winners([row for row in rows if row["status"] == "ok"])
    payload = {
        **winners,
        "candidate_rows": rows,
        "cost_budget_usd": 100,
        "max_100k_listing_cost_usd": 500,
        "notes_tr": "GPT-5.5 ve Claude Opus 4.7 aday seti disinda birakildi: TR erisim ve production maliyeti uygun degil.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Feasibility-first Turkish LLM shootout")
    parser.add_argument("--models", nargs="*", default=[candidate.id for candidate in CANDIDATES])
    parser.add_argument("--out", default="llm/selected.json")
    parser.add_argument("--rows-out", default="llm/shootout_rows.json")
    args = parser.parse_args()

    rows = run_text_slot_benchmark(args.models)
    Path(args.rows_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        missing = {row["model_id"]: row["status"] for row in rows}
        raise SystemExit(f"No runnable model candidates. Missing/access statuses: {missing}")
    payload = write_selected(rows, Path(args.out))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
