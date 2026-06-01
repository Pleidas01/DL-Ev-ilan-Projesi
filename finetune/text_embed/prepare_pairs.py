from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from evaluation.run_retrieval_eval import synthetic_query_for_record


DEFAULT_INPUT = Path("data/processed/labeled.jsonl")
DEFAULT_OUTPUT_DIR = Path("finetune/text_embed/data")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_ids(listing_ids: list[str], validation_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if len(listing_ids) < 4:
        raise ValueError("At least four listings are required for split-local negatives")
    shuffled = sorted(listing_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(2, round(len(shuffled) * validation_ratio))
    if len(shuffled) - validation_count < 2:
        raise ValueError("Split must leave at least two listings in train and validation")
    validation_ids = sorted(shuffled[:validation_count])
    train_ids = sorted(shuffled[validation_count:])
    return train_ids, validation_ids


def _rows_for_split(records_by_id: dict[str, dict[str, Any]], listing_ids: list[str]) -> list[dict[str, str]]:
    rows = []
    for index, listing_id in enumerate(listing_ids):
        record = records_by_id[listing_id]
        negative_id = listing_ids[(index + 1) % len(listing_ids)]
        rows.append({
            "source_listing_id": listing_id,
            "query": synthetic_query_for_record(record),
            "positive": record["enriched_doc"],
            "negative_listing_id": negative_id,
            "negative": records_by_id[negative_id]["enriched_doc"],
        })
    return rows


def build_pair_rows(
    records: list[dict[str, Any]],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    records_by_id = {str(record["id"]): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Listing IDs must be unique")
    if any(not record.get("enriched_doc") for record in records_by_id.values()):
        raise ValueError("Every listing must have a non-empty enriched_doc")
    train_ids, validation_ids = _split_ids(list(records_by_id), validation_ratio, seed)
    return {
        "train": _rows_for_split(records_by_id, train_ids),
        "validation": _rows_for_split(records_by_id, validation_ids),
        "manifest": {
            "seed": seed,
            "validation_ratio": validation_ratio,
            "negative_strategy": "deterministic_next_split_listing_id",
            "train_listing_ids": train_ids,
            "validation_listing_ids": validation_ids,
        },
    }


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_pair_dataset(dataset: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dataset["train"], output_dir / "train.jsonl")
    _write_jsonl(dataset["validation"], output_dir / "validation.jsonl")
    (output_dir / "split_manifest.json").write_text(
        json.dumps(dataset["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic BGE-M3 LoRA pairs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = build_pair_rows(
        _load_jsonl(args.input),
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    write_pair_dataset(dataset, args.output_dir)
    print(f"train={len(dataset['train'])} validation={len(dataset['validation'])}")


if __name__ == "__main__":
    main()
