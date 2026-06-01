from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from llm.shootout import BENCHMARK_QUERIES
from retrieval.retriever import Retriever


DEFAULT_INPUT = Path("data/processed/labeled.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluation/results")
QUERY_BOOL_LABELS = (
    ("has_elevator", "Asansor"),
    ("is_furnished", "Esyali"),
    ("in_gated_complex", "Site Icinde"),
    ("has_balcony", "Balkonlu"),
    ("near_metro", "Metroya Yakin"),
    ("near_metrobus", "Metrobuse Yakin"),
    ("has_sea_view", "Deniz Manzarali"),
    ("has_closed_parking", "Kapali Otoparkli"),
    ("has_open_parking", "Acik Otoparkli"),
    ("has_american_kitchen", "Amerikan Mutfakli"),
    ("has_aircon", "Klimali"),
    ("has_security", "Guvenlikli"),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selected_bool_filters(filters: dict[str, Any]) -> list[tuple[str, str]]:
    return [(slug, label) for slug, label in QUERY_BOOL_LABELS if filters.get(slug) is True][:3]


def synthetic_query_for_record(record: dict[str, Any]) -> str:
    filters = record.get("filter_values") or {}
    parts: list[str] = []
    for slug in ("neighborhood", "district", "room_count"):
        value = filters.get(slug)
        if value and str(value) not in parts:
            parts.append(str(value))
    if filters.get("price_amount") is not None:
        parts.append(str(filters["price_amount"]))
        parts.append(str(filters.get("price_currency") or "TL"))
        parts.append("alti")
    for _slug, label in _selected_bool_filters(filters):
        parts.append(label)
    parts.extend(["kiralik", "daire"])
    return " ".join(parts)


def synthetic_slots_for_record(record: dict[str, Any]) -> dict[str, Any]:
    values = record.get("filter_values") or {}
    filters: dict[str, Any] = {}
    for slug in ("trade_type", "property_type", "price_currency"):
        if values.get(slug) is not None:
            filters[slug] = values[slug]
    for slug in ("district", "neighborhood", "room_count"):
        if values.get(slug) is not None:
            filters[slug] = [values[slug]]
    for slug, _label in _selected_bool_filters(values):
        filters[slug] = True
    hard_filters: dict[str, Any] = {"filters": filters}
    if values.get("price_amount") is not None:
        hard_filters["max_price_amount"] = values["price_amount"]
    return {"hard_filters": hard_filters}


def retrieval_metrics(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks: list[int | None] = []
    for case in cases:
        target_id = str(case["target_id"])
        result_ids = [str(value) for value in case.get("result_ids") or []]
        ranks.append(result_ids.index(target_id) + 1 if target_id in result_ids else None)
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


def _values(value: Any) -> set[Any]:
    if isinstance(value, str):
        return set(value.split("|"))
    if isinstance(value, list):
        return set(value)
    return {value}


def satisfies_expected(filters: dict[str, Any], expected: dict[str, Any]) -> bool:
    range_fields = {
        "max_price_amount": ("price_amount", lambda actual, target: actual <= target),
        "min_price_amount": ("price_amount", lambda actual, target: actual >= target),
        "max_gross_size_m2": ("gross_size_m2", lambda actual, target: actual <= target),
        "min_gross_size_m2": ("gross_size_m2", lambda actual, target: actual >= target),
    }
    for slug, target in expected.items():
        if slug in range_fields:
            actual_slug, compare = range_fields[slug]
            actual = filters.get(actual_slug)
            if actual is None or not compare(actual, target):
                return False
            continue
        actual = filters.get(slug)
        if isinstance(target, list):
            if not set(target).issubset(_values(actual)):
                return False
        elif actual != target:
            return False
    return True


def filter_satisfaction_metrics(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    covered_queries = sum(bool(case.get("results")) for case in cases)
    constrained_results = [
        (result, case["expected"])
        for case in cases
        if case.get("expected")
        for result in case.get("results") or []
    ]
    satisfied_results = sum(
        satisfies_expected(result.get("filters") or {}, expected)
        for result, expected in constrained_results
    )
    returned_results = len(constrained_results)
    queries = len(cases)
    return {
        "queries": queries,
        "covered_queries": covered_queries,
        "query_coverage": covered_queries / queries if queries else 0.0,
        "returned_results": returned_results,
        "satisfied_results": satisfied_results,
        "filter_satisfaction": satisfied_results / returned_results if returned_results else 0.0,
    }


def _report(known_cases: list[dict[str, Any]], constraint_cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "known_item": retrieval_metrics(known_cases),
        "filter_constraints": filter_satisfaction_metrics(constraint_cases),
        "known_item_cases": known_cases,
        "filter_constraint_cases": constraint_cases,
    }


def evaluate(
    records: list[dict[str, Any]],
    retriever: Any,
    *,
    benchmarks: list[dict[str, Any]],
    benchmark_retriever: Any | None = None,
    known_limit: int | None = None,
    benchmark_limit: int | None = None,
    checkpoint_every: int = 10,
    progress_fn: Callable[[str, int, int], None] | None = None,
    checkpoint_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    benchmark_retriever = benchmark_retriever or retriever
    constraint_cases = []
    selected_benchmarks = benchmarks[:benchmark_limit] if benchmark_limit is not None else benchmarks
    for index, benchmark in enumerate(selected_benchmarks, start=1):
        results = benchmark_retriever.retrieve(benchmark["query"], top_k=10)
        constraint_cases.append({
            "query": benchmark["query"],
            "expected": benchmark["expected"],
            "results": results,
        })
        if progress_fn:
            progress_fn("benchmark", index, len(selected_benchmarks))
        if checkpoint_fn:
            checkpoint_fn(_report([], constraint_cases))

    selected_records = records[:known_limit] if known_limit is not None else records
    known_cases = []
    for index, record in enumerate(selected_records, start=1):
        query = synthetic_query_for_record(record)
        results = retriever.retrieve(query, top_k=10)
        known_cases.append({
            "query": query,
            "target_id": str(record["id"]),
            "result_ids": [str(result["id"]) for result in results],
        })
        if progress_fn:
            progress_fn("known_item", index, len(selected_records))
        if checkpoint_fn and checkpoint_every > 0 and (
            index % checkpoint_every == 0 or index == len(selected_records)
        ):
            checkpoint_fn(_report(known_cases, constraint_cases))

    return _report(known_cases, constraint_cases)


def _markdown(report: dict[str, Any]) -> str:
    known = report["known_item"]
    constraints = report["filter_constraints"]
    return "\n".join([
        "# Retrieval Evaluation",
        "",
        "Synthetic known-item metrics use deterministic source-record slots. They are controlled proxy metrics, not manual relevance gold.",
        "",
        "## Synthetic Known-Item Retrieval",
        "",
        f"- Queries: {known['queries']}",
        f"- R@1: {known['recall_at_1']:.4f}",
        f"- R@5: {known['recall_at_5']:.4f}",
        f"- R@10: {known['recall_at_10']:.4f}",
        f"- MRR: {known['mrr']:.4f}",
        "",
        "## Hard-Filter Constraints",
        "",
        f"- Queries: {constraints['queries']}",
        f"- Covered queries: {constraints['covered_queries']}",
        f"- Query coverage: {constraints['query_coverage']:.4f}",
        f"- Returned constrained results: {constraints['returned_results']}",
        f"- Satisfied results: {constraints['satisfied_results']}",
        f"- Filter satisfaction: {constraints['filter_satisfaction']:.4f}",
        "",
    ])


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "retrieval_eval.md").write_text(_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold-free retrieval evaluation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--known-limit", type=int)
    parser.add_argument("--benchmark-limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    records = _load_jsonl(args.input)
    live_retriever = Retriever()
    synthetic_slots = {
        synthetic_query_for_record(record): synthetic_slots_for_record(record)
        for record in records
    }
    known_item_retriever = Retriever(
        collection=live_retriever.collection,
        embedder=live_retriever.embedder,
        reranker=live_retriever.reranker,
        slot_extractor=synthetic_slots.__getitem__,
    )
    report = evaluate(
        records,
        known_item_retriever,
        benchmarks=BENCHMARK_QUERIES,
        benchmark_retriever=live_retriever,
        known_limit=args.known_limit,
        benchmark_limit=args.benchmark_limit,
        checkpoint_every=args.checkpoint_every,
        progress_fn=lambda stage, completed, total: print(
            f"[{stage}] {completed}/{total}",
            flush=True,
        ),
        checkpoint_fn=lambda partial: write_report(partial, args.output_dir),
    )
    write_report(report, args.output_dir)
    print(_markdown(report))


if __name__ == "__main__":
    main()
