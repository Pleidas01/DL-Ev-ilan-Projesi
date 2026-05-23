from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.clients import CANDIDATES, candidate_by_id, complete_json, estimate_cost_usd, missing_environment


SLOT_SYSTEM_PROMPT = """Sen Türkçe emlak arama sorgularını JSON slotlarına ayıran bir asistansın.
Sadece geçerli JSON döndür. Açıklama yazma."""


BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {"query": "Kadıköy'de 30 bin altı 1+1 öğrenciye uygun daire", "expected": {"rooms": ["1+1"], "districts": ["Kadıköy"], "max_price_tl": 30000}},
    {"query": "geniş salonlu denize yakın 2+1 asansörlü ev", "expected": {"rooms": ["2+1"], "elevator": True, "salon_size": "genis"}},
    {"query": "Beşiktaş 3+1 60 bin TL altı site içinde", "expected": {"rooms": ["3+1"], "districts": ["Beşiktaş"], "max_price_tl": 60000, "in_gated_complex": True}},
    {"query": "metroya yakın eşyalı 1+1 kiralık", "expected": {"rooms": ["1+1"], "near_transit": True, "furnished_text": True}},
    {"query": "ankastre mutfaklı modern 2+1", "expected": {"rooms": ["2+1"], "kitchen_type": "ankastre", "modernity": "modern"}},
    {"query": "otoparklı güvenlikli aileye uygun 3+1", "expected": {"rooms": ["3+1"], "parking": "acik", "security": True}},
    {"query": "Bostancı sahile yakın deniz manzaralı daire", "expected": {"districts": ["Bostancı"], "sea_view_mentioned": True}},
    {"query": "35 bin altı kombili 2+1", "expected": {"rooms": ["2+1"], "max_price_tl": 35000, "heating": "kombi"}},
    {"query": "ferah aydınlık parke zeminli ev", "expected": {"spaciousness": "ferah", "natural_light": "high", "floor_material": "parke"}},
    {"query": "çocuklu aile için site içinde güvenlikli 4+1", "expected": {"rooms": ["4+1"], "in_gated_complex": True, "security": True}},
    {"query": "Pendik'te 25 bin TL altı 1+1", "expected": {"rooms": ["1+1"], "districts": ["Pendik"], "max_price_tl": 25000}},
    {"query": "klimalı eşyalı stüdyo ya da 1+1", "expected": {"rooms": ["1+1"], "heating": "klima", "furnished_text": True}},
    {"query": "yeni tadilatlı ferah 3+1 daire", "expected": {"rooms": ["3+1"], "overall_condition": "tadilatli", "spaciousness": "ferah"}},
    {"query": "kapalı otoparklı lüks rezidans", "expected": {"parking": "kapali"}},
    {"query": "okula yakın güvenli 2+1", "expected": {"rooms": ["2+1"], "school_nearby": True, "security": True}},
    {"query": "evcil hayvana uygun bahçeli kiralık", "expected": {"pet_friendly": True}},
    {"query": "Suadiye'de deniz gören modern daire", "expected": {"districts": ["Suadiye"], "view": "deniz", "modernity": "modern"}},
    {"query": "Başakşehir site içinde 4+1 otoparklı", "expected": {"rooms": ["4+1"], "districts": ["Başakşehir"], "in_gated_complex": True}},
    {"query": "asansörlü doğalgazlı uygun fiyatlı 2+1", "expected": {"rooms": ["2+1"], "elevator": True, "heating": "dogalgaz"}},
    {"query": "amerikan mutfaklı 1+1 modern daire", "expected": {"rooms": ["1+1"], "kitchen_type": "amerikan", "modernity": "modern"}},
    {"query": "manzarası güzel yüksek kat 3+1", "expected": {"rooms": ["3+1"], "view": "sehir"}},
    {"query": "merkezi ısıtmalı büyük salonlu ev", "expected": {"heating": "merkezi", "salon_size": "genis"}},
    {"query": "öğrenci için metro yakını ucuz 1+1", "expected": {"rooms": ["1+1"], "near_transit": True}},
    {"query": "temiz bakımlı laminat zeminli 2+1", "expected": {"rooms": ["2+1"], "floor_material": "laminat", "overall_condition": "temiz"}},
    {"query": "bahçe katı sakin güvenlikli site", "expected": {"in_gated_complex": True, "security": True}},
    {"query": "Avcılar 3+1 50 bin altı", "expected": {"rooms": ["3+1"], "districts": ["Avcılar"], "max_price_tl": 50000}},
    {"query": "deniz manzaralı balkonlu ferah daire", "expected": {"view": "deniz", "balcony_visual": True, "spaciousness": "ferah"}},
    {"query": "ankastreli parke zeminli yeni ev", "expected": {"kitchen_type": "ankastre", "floor_material": "parke", "overall_condition": "yeni"}},
    {"query": "site içinde otoparklı güvenlikli 2+1", "expected": {"rooms": ["2+1"], "in_gated_complex": True, "security": True}},
    {"query": "Kadıköy sahile yakın 2+1 maksimum 45 bin", "expected": {"rooms": ["2+1"], "districts": ["Kadıköy"], "max_price_tl": 45000}},
]


def build_slot_prompt(query: str) -> str:
    return f"""Sorgu: {query}

Beklenen JSON şeması:
{{
  "hard_filters": {{"rooms": null, "districts": null, "max_price_tl": null, "min_size_m2": null}},
  "soft_features": {{
    "image": {{"salon_size": null, "view": null, "natural_light": null, "kitchen_type": null, "floor_material": null, "overall_condition": null, "modernity": null, "spaciousness": null, "balcony_visual": null}},
    "text_extracted": {{"elevator": null, "in_gated_complex": null, "security": null, "furnished_text": null, "heating": null, "parking": null, "near_transit": null, "sea_view_mentioned": null, "school_nearby": null, "pet_friendly": null}}
  }},
  "free_form_tr": "{query}"
}}"""


def flatten_slots(parsed: dict[str, Any]) -> dict[str, Any]:
    hard = parsed.get("hard_filters") or {}
    soft = parsed.get("soft_features") or {}
    image = soft.get("image") or {}
    text = soft.get("text_extracted") or {}
    return {**hard, **image, **text}


def score_json_adherence(raw_response: str) -> tuple[float, dict[str, Any] | None]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return 0.0, None
    required = ["hard_filters", "soft_features", "free_form_tr"]
    return sum(key in parsed for key in required) / len(required), parsed


def _actual_values_for_list_match(actual: Any) -> set[str]:
    """Normalize LLM slot values so list expectations never crash on scalars."""
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
        "notes_tr": "GPT-5.5 ve Claude Opus 4.7 aday seti dışında bırakıldı: TR erişim ve production maliyeti uygun değil.",
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
