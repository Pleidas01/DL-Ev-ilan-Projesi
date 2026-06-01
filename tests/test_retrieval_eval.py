from evaluation.run_retrieval_eval import (
    evaluate,
    filter_satisfaction_metrics,
    retrieval_metrics,
    satisfies_expected,
    synthetic_query_for_record,
    synthetic_slots_for_record,
    write_report,
)


def test_synthetic_query_uses_stable_listing_facts_for_known_item_retrieval():
    record = {
        "filter_values": {
            "district": "Kadikoy",
            "room_count": "2+1",
            "price_amount": 45000,
            "price_currency": "TL",
            "has_elevator": True,
            "has_sea_view": True,
            "has_aircon": False,
        },
    }

    assert synthetic_query_for_record(record) == (
        "Kadikoy 2+1 45000 TL alti Asansor Deniz Manzarali kiralik daire"
    )


def test_synthetic_slots_use_source_record_constraints_without_llm_interpretation():
    record = {
        "filter_values": {
            "trade_type": "kiralik",
            "property_type": "daire",
            "district": "Kadikoy",
            "neighborhood": "Moda Mahallesi",
            "room_count": "2+1",
            "price_amount": 45000,
            "price_currency": "TL",
            "has_elevator": True,
            "has_sea_view": True,
            "has_aircon": False,
        },
    }

    assert synthetic_slots_for_record(record) == {
        "hard_filters": {
            "filters": {
                "trade_type": "kiralik",
                "property_type": "daire",
                "district": ["Kadikoy"],
                "neighborhood": ["Moda Mahallesi"],
                "room_count": ["2+1"],
                "price_currency": "TL",
                "has_elevator": True,
                "has_sea_view": True,
            },
            "max_price_amount": 45000,
        },
    }


def test_retrieval_metrics_measure_known_item_rank_at_each_cutoff():
    metrics = retrieval_metrics(
        [
            {"target_id": "a", "result_ids": ["a", "x"]},
            {"target_id": "b", "result_ids": ["x", "y", "b"]},
            {"target_id": "c", "result_ids": ["x"]},
        ]
    )

    assert metrics == {
        "queries": 3,
        "recall_at_1": 1 / 3,
        "recall_at_5": 2 / 3,
        "recall_at_10": 2 / 3,
        "mrr": (1 + 1 / 3) / 3,
    }


def test_satisfies_expected_supports_ranges_and_pipe_joined_multi_enum_values():
    filters = {
        "district": "Kadikoy",
        "price_amount": 30000,
        "balcony_type": "acik_balkon|acik_teras",
    }

    assert satisfies_expected(
        filters,
        {
            "district": ["Kadikoy"],
            "max_price_amount": 35000,
            "balcony_type": ["acik_teras"],
        },
    )
    assert not satisfies_expected(filters, {"min_price_amount": 35000})


def test_filter_satisfaction_reports_empty_queries_as_uncovered_not_satisfied():
    metrics = filter_satisfaction_metrics(
        [
            {
                "query": "asansorlu",
                "expected": {"has_elevator": True},
                "results": [{"filters": {"has_elevator": True}}],
            },
            {
                "query": "Kadikoy",
                "expected": {"district": ["Kadikoy"]},
                "results": [],
            },
            {
                "query": "terasli",
                "expected": {"balcony_type": ["acik_teras"]},
                "results": [{"filters": {"balcony_type": "acik_balkon"}}],
            },
        ]
    )

    assert metrics == {
        "queries": 3,
        "covered_queries": 2,
        "query_coverage": 2 / 3,
        "returned_results": 2,
        "satisfied_results": 1,
        "filter_satisfaction": 1 / 2,
    }


def test_evaluate_combines_known_item_and_benchmark_metrics_with_injected_retriever():
    class FakeRetriever:
        def retrieve(self, query, *, top_k=10):
            if query == "Kadikoy 2+1 30000 TL alti Asansor kiralik daire":
                return [{"id": "listing-1", "filters": {"district": "Kadikoy"}}]
            if query == "Kadikoy'de asansorlu daire":
                return [{"id": "listing-1", "filters": {"district": "Kadikoy", "has_elevator": True}}]
            return []

    report = evaluate(
        [
            {
                "id": "listing-1",
                "filter_values": {
                    "district": "Kadikoy",
                    "room_count": "2+1",
                    "price_amount": 30000,
                    "price_currency": "TL",
                    "has_elevator": True,
                },
            },
        ],
        FakeRetriever(),
        benchmarks=[
            {
                "query": "Kadikoy'de asansorlu daire",
                "expected": {"district": ["Kadikoy"], "has_elevator": True},
            },
        ],
    )

    assert report["known_item"]["recall_at_10"] == 1.0
    assert report["filter_constraints"]["filter_satisfaction"] == 1.0
    assert report["filter_constraints"]["query_coverage"] == 1.0


def test_evaluate_can_use_live_benchmark_retriever_separately_from_known_item_retriever():
    class KnownRetriever:
        def retrieve(self, _query, *, top_k=10):
            return [{"id": "listing-1", "filters": {}}]

    class BenchmarkRetriever:
        def retrieve(self, _query, *, top_k=10):
            return [{"id": "listing-2", "filters": {"district": "Kadikoy"}}]

    report = evaluate(
        [{"id": "listing-1", "filter_values": {"district": "Kadikoy"}}],
        KnownRetriever(),
        benchmark_retriever=BenchmarkRetriever(),
        benchmarks=[{"query": "Kadikoy", "expected": {"district": ["Kadikoy"]}}],
    )

    assert report["known_item"]["recall_at_1"] == 1.0
    assert report["filter_constraints"]["filter_satisfaction"] == 1.0


def test_evaluate_runs_live_benchmark_constraints_before_long_known_item_proxy():
    events = []

    class KnownRetriever:
        def retrieve(self, _query, *, top_k=10):
            events.append("known")
            return []

    class BenchmarkRetriever:
        def retrieve(self, _query, *, top_k=10):
            events.append("benchmark")
            return []

    evaluate(
        [{"id": "listing-1", "filter_values": {}}],
        KnownRetriever(),
        benchmark_retriever=BenchmarkRetriever(),
        benchmarks=[{"query": "Kadikoy", "expected": {"district": ["Kadikoy"]}}],
    )

    assert events == ["benchmark", "known"]


def test_evaluate_limits_benchmarks_and_checkpoints_partial_progress():
    events = []
    checkpoints = []

    class KnownRetriever:
        def retrieve(self, query, *, top_k=10):
            return [{"id": query, "filters": {}}]

    class BenchmarkRetriever:
        def retrieve(self, query, *, top_k=10):
            return [{"id": query, "filters": {}}]

    records = [
        {"id": "one", "filter_values": {}},
        {"id": "two", "filter_values": {}},
    ]
    report = evaluate(
        records,
        KnownRetriever(),
        benchmark_retriever=BenchmarkRetriever(),
        benchmarks=[
            {"query": "first", "expected": {}},
            {"query": "second", "expected": {}},
        ],
        benchmark_limit=1,
        checkpoint_every=1,
        progress_fn=lambda stage, completed, total: events.append((stage, completed, total)),
        checkpoint_fn=lambda partial: checkpoints.append(partial),
    )

    assert report["filter_constraints"]["queries"] == 1
    assert events == [
        ("benchmark", 1, 1),
        ("known_item", 1, 2),
        ("known_item", 2, 2),
    ]
    assert [checkpoint["known_item"]["queries"] for checkpoint in checkpoints] == [0, 1, 2]


def test_evaluate_checkpoints_each_completed_live_benchmark():
    checkpoints = []

    class Retriever:
        def retrieve(self, query, *, top_k=10):
            return [{"id": query, "filters": {}}]

    evaluate(
        [],
        Retriever(),
        benchmarks=[
            {"query": "first", "expected": {}},
            {"query": "second", "expected": {}},
        ],
        checkpoint_fn=lambda partial: checkpoints.append(partial),
    )

    assert [checkpoint["filter_constraints"]["queries"] for checkpoint in checkpoints] == [1, 2]


def test_write_report_creates_json_and_markdown_presentation_artifacts(tmp_path):
    report = {
        "known_item": {
            "queries": 1,
            "recall_at_1": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
        },
        "filter_constraints": {
            "queries": 1,
            "covered_queries": 1,
            "query_coverage": 1.0,
            "returned_results": 1,
            "satisfied_results": 1,
            "filter_satisfaction": 1.0,
        },
    }

    write_report(report, tmp_path)

    assert '"recall_at_10": 1.0' in (tmp_path / "retrieval_eval.json").read_text(encoding="utf-8")
    assert "controlled proxy metrics" in (tmp_path / "retrieval_eval.md").read_text(encoding="utf-8")
