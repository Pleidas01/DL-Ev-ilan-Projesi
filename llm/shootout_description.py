from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.clients import candidate_by_id, complete_json, estimate_cost_usd, missing_environment
from llm.gold_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GOLD_PATH,
    aggregate_model_scores,
    gold_is_filled,
    gold_listing_ids,
    listing_description,
    load_dataset_index,
    load_gold_rows,
    load_raw_index,
    score_against_gold,
)

DESCRIPTION_MODEL_IDS = ("deepseek_v4_flash", "kimi_k2_6", "gemma_4_local")
# NOT (2026-05-29): text_model=kimi_k2_6 kilitli (slot M1 + vision M1.5 kazananı);
# description shootout yalnızca opsiyonel cross-check. heating_type + is_furnished
# STRUCTURED alanlar olduğu için çıkarıldı — LLM'i, description'dan çıkaramayacağı
# structured truth'a karşı haksız cezalandırıyordu (bkz. STATUS.md M1.5).
DESCRIPTION_FACT_FIELDS = (
    "has_balcony",
    "has_elevator",
    "has_parking",
    "near_metro",
    "near_metrobus",
)

DESCRIPTION_SYSTEM_PROMPT = """Sen Turkce emlak ilani aciklamalarindan JSON alanlari cikaran bir asistansin.
Sadece gecerli JSON dondur. Aciklama yazma."""


def build_description_prompt(description: str) -> str:
    return f"""Ilan aciklamasi:
{description}

JSON semasi:
{{
  "has_balcony": null,
  "has_elevator": null,
  "has_parking": null,
  "near_metro": null,
  "near_metrobus": null
}}

near_metro sadece <15dk yurume anlami varsa true olmali. Emin olmadigin alanlar icin null kullan."""


def parse_description_json(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_description_gold_benchmark(
    model_ids: list[str],
    gold_path: Path = DEFAULT_GOLD_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    gold_rows = {row["listing_id"]: row for row in load_gold_rows(gold_path)}
    dataset_index = load_dataset_index(dataset_path)
    raw_index = load_raw_index()
    listing_ids = gold_listing_ids(gold_path)

    model_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        candidate = candidate_by_id(model_id)
        missing = missing_environment(candidate)
        if missing:
            model_rows.append({
                "model_id": model_id,
                "status": f"missing_env:{','.join(missing)}",
                "accuracy": None,
                "cost_estimate_usd": estimate_cost_usd(candidate, 900 * 10, 180 * 10),
                "samples": [],
            })
            continue

        samples: list[dict[str, Any]] = []
        try:
            for listing_id in listing_ids:
                gold_row = gold_rows[listing_id]
                facts_gold = gold_row.get("facts_gold") or {}
                desc_gold = {field: facts_gold.get(field) for field in DESCRIPTION_FACT_FIELDS}
                if not gold_is_filled(desc_gold):
                    samples.append({"listing_id": listing_id, "status": "gold_not_ready", "accuracy": None})
                    continue

                description = listing_description(listing_id, dataset_index, raw_index)
                if not description:
                    samples.append({"listing_id": listing_id, "status": "missing_description", "accuracy": None})
                    continue

                raw = complete_json(candidate, DESCRIPTION_SYSTEM_PROMPT, build_description_prompt(description))
                predicted = parse_description_json(raw)
                scored = score_against_gold(predicted, facts_gold, DESCRIPTION_FACT_FIELDS)
                samples.append({
                    "listing_id": listing_id,
                    "status": "ok",
                    "predicted": predicted,
                    "score": scored,
                    "accuracy": scored["accuracy"],
                })
        except Exception as exc:
            model_rows.append({
                "model_id": model_id,
                "status": f"error:{type(exc).__name__}:{exc}",
                "accuracy": None,
                "cost_estimate_usd": estimate_cost_usd(candidate, 900 * 10, 180 * 10),
                "samples": samples,
            })
            continue

        aggregate = aggregate_model_scores(samples)
        model_rows.append({
            "model_id": model_id,
            "status": "ok" if aggregate["scored_samples"] else "gold_not_ready",
            "accuracy": aggregate["accuracy"],
            "cost_estimate_usd": estimate_cost_usd(candidate, 900 * 10, 180 * 10),
            "samples": samples,
            **aggregate,
        })
    return model_rows


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Description shootout against facts_gold 8 LLM fields")
    parser.add_argument("--models", nargs="*", default=list(DESCRIPTION_MODEL_IDS))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default="llm/shootout_description_rows.json")
    args = parser.parse_args()

    rows = run_description_gold_benchmark(args.models, Path(args.gold), Path(args.dataset))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
