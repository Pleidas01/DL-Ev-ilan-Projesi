"""
Late Fusion α Sweep
====================
Text-only, image-only ve çeşitli α değerlerindeki late-fusion
performanslarını karşılaştırır. En iyi α'yı bulur.

Kullanım:
    python evaluation/fusion_sweep.py
    python evaluation/fusion_sweep.py --steps 11 --checkpoint model/checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import evaluate, load_test_split
from retrieval.retriever import Retriever


def sweep(
    retriever: Retriever,
    test_records: list[dict],
    k_list: list[int],
    alpha_steps: int,
    lam: float,
) -> dict:
    alphas = np.linspace(0.0, 1.0, alpha_steps).tolist()
    results = {}

    for alpha in alphas:
        retriever._sweep_alpha = alpha  # geçici alpha override
        metrics = evaluate(retriever, test_records, k_list, "multimodal", lam)
        results[round(alpha, 3)] = metrics
        k1 = metrics.get("Recall@1", 0)
        mrr = metrics["MRR"]
        print(f"  α={alpha:.2f}  Recall@1={k1:.3f}  MRR={mrr:.3f}")

    return results


def main(args: argparse.Namespace) -> None:
    print(f"[Retriever] Yükleniyor… checkpoint={args.checkpoint}")
    retriever = Retriever(checkpoint=args.checkpoint)

    test_records = load_test_split(args.data)
    print(f"[Test seti] {len(test_records)} örnek")
    print(f"\n[Sweep] {args.steps} α değeri  K={args.k}  λ={args.lam}")
    print("=" * 55)

    sweep_results = sweep(retriever, test_records, args.k, args.steps, args.lam)

    # En iyi α (Recall@1 bazında)
    k1_key = f"Recall@{args.k[0]}"
    best_alpha = max(sweep_results, key=lambda a: sweep_results[a].get(k1_key, 0))
    print(f"\n[Sonuç] En iyi α = {best_alpha}  ({k1_key}={sweep_results[best_alpha][k1_key]:.3f})")

    # Matplotlib grafik
    try:
        import matplotlib.pyplot as plt

        alphas = list(sweep_results.keys())
        mrrs   = [sweep_results[a]["MRR"] for a in alphas]
        r1s    = [sweep_results[a].get(k1_key, 0) for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(alphas, mrrs, label="MRR",      marker="o")
        ax.plot(alphas, r1s,  label=k1_key,    marker="s")
        ax.axvline(best_alpha, color="red", linestyle="--", label=f"Best α={best_alpha}")
        ax.set_xlabel("α (text weight)")
        ax.set_ylabel("Score")
        ax.set_title("Late Fusion α Sweep")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        plot_path = out / "fusion_sweep.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[Grafik] {plot_path}")
    except ImportError:
        print("[Grafik] matplotlib bulunamadı, atlanıyor.")

    # JSON rapor
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "fusion_sweep.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"sweep": sweep_results, "best_alpha": best_alpha}, f, indent=2)
    print(f"[Rapor] {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Late Fusion α Sweep")
    p.add_argument("--data",       default="data/processed/dataset.jsonl")
    p.add_argument("--out",        default="evaluation/results")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--k",          nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--lam",        type=float, default=0.5)
    p.add_argument("--steps",      type=int,   default=11,
                   help="α için kaç adım (0.0 → 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
