from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.clients import candidate_by_id, complete_vision_json, estimate_cost_usd, missing_environment
from llm.gold_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GOLD_PATH,
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

VISION_USER_PROMPT = """Bu fotograflardan asagidaki 6 visual_gold alanini cikar. Fotografta gorunmeyen alanlar icin null kullan.

ONEMLI: Degerlerde Turkce karakter KULLANMA. c/g/i/o/s/u yaz; ç/ğ/ı/ö/ş/ü DEGIL (orn. "bogaz", "acik_otopark"). Sadece asagidaki listede yer alan degerleri aynen kullan.

JSON semasi:
{
  "balkon_ozellikleri": null,
  "manzara": null,
  "mutfak_tipi": null,
  "banyo_ozellikleri": null,
  "salon_ozellikleri": null,
  "imkanlar": null
}

Enumlar (sadece bu degerleri kullan):
- balkon_ozellikleri (liste): cam_balkon, acik_balkon, fransiz_balkon, cikma_balkon, teras
- manzara (liste): deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari
- mutfak_tipi (tek deger): amerikan_acik | kapali_ayri
- banyo_ozellikleri (liste): dusakabin, kuvet, jakuzi, banyoda_pencere, birden_fazla_banyo
- salon_ozellikleri (liste): somine, nis, acik_plan_genis, ayri_yemek_alani
- imkanlar (liste): havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani

Liste (multi-select) alanlar JSON array olmalidir: balkon_ozellikleri, manzara, banyo_ozellikleri, salon_ozellikleri, imkanlar."""


def parse_vision_json(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Cost tahmini (kaba): multi-image tek çağrı/listing. 10 listing, her çağrı
# ~8 foto × 5000 input token (Kimi image) + prompt, ~300 output token/listing.
# Gerçek maliyet Moonshot bakiye farkıyla ölçülür; bu sadece sıra-büyüklük tahmini.
COST_EST_INPUT_TOKENS = 5000 * 8 * 10
COST_EST_OUTPUT_TOKENS = 300 * 10


def run_vision_gold_benchmark(
    model_ids: list[str],
    gold_path: Path = DEFAULT_GOLD_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    max_photos: int | None = None,
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

                # max_photos=None -> tüm fotoğraflar (per-image ile adil karşılaştırma).
                # İsteğe bağlı tavan (Kimi foto limiti / maliyet) için --max-photos.
                image_paths = listing_image_paths(record, max_photos)
                if not image_paths:
                    samples.append({"listing_id": listing_id, "status": "no_images", "accuracy": None})
                    continue

                resolved_paths = [str(p) for p in image_paths if Path(p).is_file()]
                missing_files = len(image_paths) - len(resolved_paths)
                if not resolved_paths:
                    samples.append({"listing_id": listing_id, "status": "all_images_missing", "accuracy": None})
                    continue

                # Tüm fotoğraflar TEK çağrıda; model tek bütünleşik JSON döndürür.
                # Listing-başına resilience: bir ilan patlarsa (örn. Kimi foto limiti /
                # timeout) sadece o "error" işaretlenir, diğer ilanlar devam eder.
                try:
                    raw = complete_vision_json(candidate, VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, resolved_paths)
                except Exception as exc:
                    samples.append({
                        "listing_id": listing_id,
                        "status": f"error:{type(exc).__name__}:{exc}",
                        "photo_count": len(resolved_paths),
                        "missing_files": missing_files,
                        "accuracy": None,
                    })
                    continue

                predicted = parse_vision_json(raw)
                scored = score_against_gold(predicted, visual_gold, VISUAL_GOLD_FIELDS)
                samples.append({
                    "listing_id": listing_id,
                    "status": "ok",
                    "photo_count": len(resolved_paths),
                    "missing_files": missing_files,
                    "predicted": predicted,
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
    # Windows cp1254 konsol Türkçe/özel karakter print'inde crash eder; UTF-8'e zorla.
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

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
    parser.add_argument("--max-photos", type=int, default=None,
                        help="Listing başına TEK çağrıya giren maks foto (varsayılan: tümü). "
                             "Kimi foto limiti / maliyet için tavan koymak istersen ayarla.")
    args = parser.parse_args()

    rows = run_vision_gold_benchmark(args.models, Path(args.gold), Path(args.dataset), args.max_photos)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
