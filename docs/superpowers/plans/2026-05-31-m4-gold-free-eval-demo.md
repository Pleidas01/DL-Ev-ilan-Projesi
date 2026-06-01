# M4 Gold-Free Eval and Demo Implementation Plan

> **Uygulama notu (2026-06-01):** Baseline tamamlandı. MacBook Air CPU-only ortamında full `303` sorgu eval bilinçli olarak koşulmadı; checkpoint'li bounded örneklem kullanıldı. Gerçek sonuçlar ve güvenli komutlar için `docs/HANDOFF.md §1.5` oku.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real 303-listing Chroma index, add a deterministic gold-free retrieval evaluation harness, and verify the Streamlit baseline demo.

**Architecture:** Keep the existing canonical retrieval path unchanged. Add one focused evaluation module with pure metric helpers and a CLI runner. Use synthetic known-item queries for retrieval proxy metrics and the existing slot benchmark `expected` fields for hard-filter satisfaction and coverage.

**Tech Stack:** Python, pytest, ChromaDB, SentenceTransformers BGE-M3, bge-reranker-v2-m3, DeepSeek API, Streamlit.

---

## Task 1: Verify Final M3 Contract and Build Baseline Index

**Files:**
- Read: `data/processed/labeled.jsonl`
- Read: `indexing/build_chroma.py`
- Read: `indexing/composer.py`

- [ ] Run a read-only integrity check that asserts `303` unique IDs, dataset ID parity, non-empty `enriched_doc`, and non-empty visual sections.
- [ ] Run: `.venv/bin/python -m indexing.build_chroma --input data/processed/labeled.jsonl`
- [ ] Verify collection count is `303`.
- [ ] Inspect representative metadata and assert values are scalar-only, `None` is absent, and multi-enum flags such as `balcony_type__acik_balkon` are booleans when present.

## Task 2: Add Pure Gold-Free Metric Helpers with TDD

**Files:**
- Create: `tests/test_retrieval_eval.py`
- Create: `evaluation/run_retrieval_eval.py`

- [ ] Write failing tests for:
  - deterministic synthetic query generation from district, room count, price, and true boolean facts;
  - R@1/R@5/R@10 and MRR calculation;
  - filter satisfaction using `expected` constraints;
  - query coverage counting empty result sets without inflating satisfaction;
  - pipe-joined multi-enum metadata matching expected list values.
- [ ] Run: `.venv/bin/python -m pytest -q tests/test_retrieval_eval.py`
- [ ] Confirm RED because `evaluation.run_retrieval_eval` does not exist.
- [ ] Implement minimal pure helpers:

```python
def synthetic_query_for_record(record: dict[str, Any]) -> str: ...
def retrieval_metrics(cases: list[dict[str, Any]]) -> dict[str, float]: ...
def satisfies_expected(filters: dict[str, Any], expected: dict[str, Any]) -> bool: ...
def filter_satisfaction_metrics(cases: list[dict[str, Any]]) -> dict[str, float | int]: ...
```

- [ ] Run: `.venv/bin/python -m pytest -q tests/test_retrieval_eval.py`
- [ ] Confirm GREEN.

## Task 3: Add Evaluation CLI

**Files:**
- Modify: `evaluation/run_retrieval_eval.py`
- Modify: `tests/test_retrieval_eval.py`

- [ ] Write failing tests with a fake retriever proving the CLI-level evaluator:
  - calls known-item retrieval for each selected record;
  - calls benchmark retrieval for each selected benchmark;
  - returns both metric groups in one report payload.
- [ ] Run the focused test and confirm RED.
- [ ] Implement:

```python
def evaluate(
    records: list[dict[str, Any]],
    retriever: Any,
    *,
    benchmarks: list[dict[str, Any]],
    known_limit: int | None = None,
) -> dict[str, Any]: ...
```

- [ ] Add CLI options `--input`, `--output-dir`, and `--known-limit`.
- [ ] Write `evaluation/results/retrieval_eval.json` and `evaluation/results/retrieval_eval.md`.
- [ ] Run: `.venv/bin/python -m pytest -q tests/test_retrieval_eval.py`

## Task 4: Run Real Evaluation and Demo QA

**Files:**
- Create by CLI: `evaluation/results/retrieval_eval.json`
- Create by CLI: `evaluation/results/retrieval_eval.md`

- [ ] Run a small real smoke: `.venv/bin/python -m evaluation.run_retrieval_eval --known-limit 5`
- [ ] Inspect failures before starting the full paid run.
- [ ] Run full evaluation: `.venv/bin/python -m evaluation.run_retrieval_eval`
- [ ] Run: `.venv/bin/python -m pytest -q`
- [ ] Start Streamlit: `.venv/bin/streamlit run ui/app.py --server.headless true`
- [ ] Verify representative searches render listing cards, Turkish labels, deterministic match chips, and a DeepSeek answer.

## Task 5: Record Baseline Numbers

**Files:**
- Modify: `docs/STATUS.md`
- Modify: `docs/HANDOFF.md`

- [ ] Record collection count, synthetic R@K/MRR, filter satisfaction, coverage, test count, and smoke outcome.
- [ ] State explicitly that synthetic R@K is a proxy metric rather than manual relevance gold.
- [ ] Keep LoRA as the next independent experiment; do not claim an improvement before held-out comparison.
