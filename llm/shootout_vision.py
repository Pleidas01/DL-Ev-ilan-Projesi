from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm.clients import candidate_by_id, complete_vision_json, estimate_cost_usd, missing_environment
from llm.gold_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GOLD_PATH,
    GOLD_BENCHMARK_PHOTO_COUNT,
    MULTI_SELECT_FIELDS,
    VISUAL_GOLD_FIELDS,
    aggregate_model_scores,
    gold_is_filled,
    gold_listing_ids,
    listing_image_paths,
    load_dataset_index,
    load_gold_rows,
    score_against_gold,
)

VISION_MODEL_IDS = ("gemma_4_local", "kimi_k2_6")

VISION_SYSTEM_PROMPT = """Sen emlak ilani fotograflarini JSON alanlarina etiketleyen bir asistansin.
Sadece gecerli JSON dondur. Aciklama yazma."""

VISION_USER_PROMPT = """Bu fotograflardan asagidaki 7 visual_gold alanini cikar. Fotografta gorunmeyen alanlar icin null kullan.

JSON semasi:
{
  "balkon_ozellikleri": null,
  "manzara": null,
  "mutfak_tipi": null,
  "banyo_ozellikleri": null,
  "zemin_tipi": null,
  "salon_ozellikleri": null,
  "imkanlar": null
}

Enumlar (sadece bu degerleri kullan):
- balkon_ozellikleri (liste): cam_balkon, acik_balkon, fransiz_balkon, cikma_balkon, teras
- manzara (liste): deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari
- mutfak_tipi (tek deger): amerikan_acik | kapali_ayri
- banyo_ozellikleri (liste): dusakabin, kuvet, jakuzi, banyoda_pencere, birden_fazla_banyo
- zemin_tipi (tek deger): parke | laminat | seramik | granit | mermer | hali | karma
- salon_ozellikleri (liste): somine, nis, acik_plan_genis, ayri_yemek_alani
- imkanlar (liste): havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani

Liste (multi-select) alanlar JSON array olmalidir: balkon_ozellikleri, manzara, banyo_ozellikleri, salon_ozellikleri, imkanlar."""


def parse_vision_json(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def aggregate_per_image_predictions(per_image_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Bir ilanın N fotosundan gelen per-image VLM çıktılarını tek bir visual_gold
    JSON'una indirger. Multi-select alanlar: union. Single-select alanlar: majority
    vote (eşitlikte ilk gelen). Bir alanı hiçbir foto görmediyse null.
    """
    aggregated: dict[str, Any] = {}
    for field in VISUAL_GOLD_FIELDS:
        if field in MULTI_SELECT_FIELDS:
            union: set[str] = set()
            for pred in per_image_preds:
                value = pred.get(field)
                if isinstance(value, list):
                    union.update(str(v).strip().lower() for v in value if v)
            aggregated[field] = sorted(union) if union else None
        else:
            counter: Counter[str] = Counter()
            for pred in per_image_preds:
                value = pred.get(field)
                if value is None:
                    continue
                normalized = str(value).strip().lower()
                if not normalized or normalized in ("null", "none"):
                    continue
                counter[normalized] += 1
            aggregated[field] = counter.most_common(1)[0][0] if counter else None
    return aggregated


# Cost tahmini için varsayım: 10 listing × 5 foto = 50 VLM çağrısı,
# her çağrı ~1200 input + 250 output token (resim base64 + prompt + JSON).
COST_EST_INPUT_TOKENS = 1200 * 50
COST_EST_OUTPUT_TOKENS = 250 * 50


def run_vision_gold_benchmark(
    model_ids: list[str],
    gold_path: Path = DEFAULT_GOLD_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    gold_rows = {row["listing_id"]: row for row in load_gold_rows(gold_path)}
    dataset_index = load_dataset_index(dataset_path)
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
                "cost_estimate_usd": estimate_cost_usd(candidate, COST_EST_INPUT_TOKENS, COST_EST_OUTPUT_TOKENS),
                "samples": [],
            })
            continue

        samples: list[dict[str, Any]] = []
        try:
            for listing_id in listing_ids:
                gold_row = gold_rows[listing_id]
                visual_gold = gold_row.get("visual_gold") or {}
                if not gold_is_filled(visual_gold):
                    samples.append({"listing_id": listing_id, "status": "gold_not_ready", "accuracy": None})
                    continue

                record = dataset_index.get(listing_id)
                if not record:
                    samples.append({"listing_id": listing_id, "status": "missing_dataset_record", "accuracy": None})
                    continue

                image_paths = listing_image_paths(record, GOLD_BENCHMARK_PHOTO_COUNT)
                if not image_paths:
                    samples.append({"listing_id": listing_id, "status": "no_images", "accuracy": None})
                    continue

                per_image: list[dict[str, Any]] = []
                missing_files = 0
                for image_path in image_paths:
                    resolved = Path(image_path)
                    if not resolved.is_file():
                        missing_files += 1
                        continue
                    raw = complete_vision_json(candidate, VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, str(resolved))
                    per_image.append({"image_path": str(resolved), "predicted": parse_vision_json(raw)})

                if not per_image:
                    samples.append({"listing_id": listing_id, "status": "all_images_missing", "accuracy": None})
                    continue

                predicted_aggregated = aggregate_per_image_predictions([item["predicted"] for item in per_image])
                scored = score_against_gold(predicted_aggregated, visual_gold, VISUAL_GOLD_FIELDS)
                samples.append({
                    "listing_id": listing_id,
                    "status": "ok",
                    "photo_count": len(per_image),
                    "missing_files": missing_files,
                    "per_image": per_image,
                    "predicted_aggregated": predicted_aggregated,
                    "score": scored,
                    "accuracy": scored["accuracy"],
                })
        except Exception as exc:
            model_rows.append({
                "model_id": model_id,
                "status": f"error:{type(exc).__name__}:{exc}",
                "accuracy": None,
                "cost_estimate_usd": estimate_cost_usd(candidate, COST_EST_INPUT_TOKENS, COST_EST_OUTPUT_TOKENS),
                "samples": samples,
            })
            continue

        aggregate = aggregate_model_scores(samples)
        model_rows.append({
            "model_id": model_id,
            "status": "ok" if aggregate["scored_samples"] else "gold_not_ready",
            "accuracy": aggregate["accuracy"],
            "cost_estimate_usd": estimate_cost_usd(candidate, COST_EST_INPUT_TOKENS, COST_EST_OUTPUT_TOKENS),
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

    parser = argparse.ArgumentParser(description="Vision shootout against visual_gold")
    parser.add_argument("--models", nargs="*", default=list(VISION_MODEL_IDS))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default="llm/shootout_vision_rows.json")
    args = parser.parse_args()

    rows = run_vision_gold_benchmark(args.models, Path(args.gold), Path(args.dataset))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
