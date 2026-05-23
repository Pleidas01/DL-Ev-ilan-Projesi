"""
Retrieval Evaluation — Recall@K & MRR
======================================
dataset.jsonl içindeki test setini sorgu olarak kullanır.
Her ilan için kendi görselini sorgu yapıp listenin ilk K sonucunda
o ilanı bulabiliyor muyuz diye bakar (image-to-text retrieval).

Kullanım:
    python evaluation/metrics.py
    python evaluation/metrics.py --k 1 5 10 --checkpoint model/checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.retriever import Retriever


def load_test_split(jsonl_path: str, split_ratio: float = 0.1, seed: int = 42) -> list[dict]:
    """dataset.jsonl'nin son %10'unu test seti olarak döndürür."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_test = max(1, int(len(records) * split_ratio))
    test_indices = indices[-n_test:]
    return [records[i] for i in test_indices]


def recall_at_k(results: list[dict], target_id: str, k: int) -> float:
    top_k_ids = [r["metadata"]["id"] for r in results[:k]]
    return 1.0 if target_id in top_k_ids else 0.0


def reciprocal_rank(results: list[dict], target_id: str) -> float:
    for rank, r in enumerate(results, 1):
        if r["metadata"]["id"] == target_id:
            return 1.0 / rank
    return 0.0


def evaluate(
    retriever: Retriever,
    test_records: list[dict],
    k_list: list[int],
    query_mode: str = "image",
    lam: float = 0.5,
) -> dict:
    """
    query_mode: "image" | "text" | "multimodal"
    """
    from PIL import Image

    max_k = max(k_list)
    recall_scores = {k: [] for k in k_list}
    rr_scores = []

    print(f"\n[Eval] {len(test_records)} test örneği, mode={query_mode}")

    for rec in test_records:
        target_id = rec["id"]

        if query_mode == "image":
            results = retriever.query_image(rec["image_path"], k=max_k, lam=lam)
        elif query_mode == "text":
            results = retriever.query_text(rec["text"], k=max_k, lam=lam)
        else:
            img = Image.open(rec["image_path"]).convert("RGB")
            results = retriever.query_multimodal(rec["text"], img, k=max_k, lam=lam)

        for k in k_list:
            recall_scores[k].append(recall_at_k(results, target_id, k))
        rr_scores.append(reciprocal_rank(results, target_id))

    metrics = {f"Recall@{k}": float(np.mean(recall_scores[k])) for k in k_list}
    metrics["MRR"] = float(np.mean(rr_scores))
    return metrics


def print_metrics(metrics: dict, label: str = "") -> None:
    print(f"\n{'=' * 45}")
    if label:
        print(f"  {label}")
    print(f"{'=' * 45}")
    for name, val in metrics.items():
        print(f"  {name:<15}: {val:.4f}  ({val*100:.1f}%)")
    print(f"{'=' * 45}")


def main(args: argparse.Namespace) -> None:
    print(f"[Retriever] Yükleniyor… checkpoint={args.checkpoint}")
    retriever = Retriever(checkpoint=args.checkpoint)

    test_records = load_test_split(args.data)
    print(f"[Test seti] {len(test_records)} örnek")

    results_image = evaluate(retriever, test_records, args.k, "image", args.lam)
    print_metrics(results_image, "Image Query")

    results_text = evaluate(retriever, test_records, args.k, "text", args.lam)
    print_metrics(results_text, "Text Query")

    results_mm = evaluate(retriever, test_records, args.k, "multimodal", args.lam)
    print_metrics(results_mm, "Multimodal Query")

    # JSON rapor
    report = {
        "image":      results_image,
        "text":       results_text,
        "multimodal": results_mm,
        "config": {"k_list": args.k, "lam": args.lam, "checkpoint": args.checkpoint},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Rapor] {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval Evaluation Metrics")
    p.add_argument("--data",       default="data/processed/dataset.jsonl")
    p.add_argument("--out",        default="evaluation/results")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--k",          nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--lam",        type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
