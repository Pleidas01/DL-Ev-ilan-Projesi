from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_LISTINGS = Path("data/processed/labeled.jsonl")
DEFAULT_QUERIES = Path("finetune/text_embed/data/validation.jsonl")
DEFAULT_OUTPUT = Path("finetune/text_embed/results/baseline_dense.json")
DEFAULT_MODEL = "BAAI/bge-m3"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dense_metrics(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks = []
    for case in cases:
        source_id = str(case["source_listing_id"])
        ranked_ids = [str(value) for value in case["ranked_listing_ids"]]
        ranks.append(ranked_ids.index(source_id) + 1 if source_id in ranked_ids else None)
    total = len(ranks)

    def recall_at(cutoff: int) -> float:
        return sum(rank is not None and rank <= cutoff for rank in ranks) / total if total else 0.0

    return {
        "queries": total,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total if total else 0.0,
    }


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def evaluate_dense(
    queries: list[dict[str, Any]],
    listings: list[dict[str, Any]],
    embedder: Any,
) -> dict[str, Any]:
    listing_ids = [str(listing["id"]) for listing in listings]
    listing_vectors = embedder.encode(
        [listing["enriched_doc"] for listing in listings],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vectors = embedder.encode(
        [row["query"] for row in queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cases = []
    for row, query_vector in zip(queries, query_vectors):
        ranked = sorted(
            zip(listing_ids, listing_vectors),
            key=lambda item: _cosine(query_vector, item[1]),
            reverse=True,
        )
        cases.append({
            "source_listing_id": str(row["source_listing_id"]),
            "query": row["query"],
            "ranked_listing_ids": [listing_id for listing_id, _vector in ranked[:10]],
        })
    return {"metrics": dense_metrics(cases), "cases": cases}


def _load_embedder(model_name: str, adapter_path: Path | None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    if adapter_path is not None:
        model.load_adapter(str(adapter_path))
    return model


def write_result(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense-only held-out retrieval evaluation")
    parser.add_argument("--listings", type=Path, default=DEFAULT_LISTINGS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()
    report = evaluate_dense(
        _load_jsonl(args.queries),
        _load_jsonl(args.listings),
        _load_embedder(args.model, args.adapter_path),
    )
    write_result(report, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
