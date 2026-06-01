import json

from finetune.text_embed.evaluate_dense import dense_metrics, evaluate_dense, write_result


def test_dense_metrics_reports_recall_cutoffs_and_mrr():
    metrics = dense_metrics(
        [
            {"source_listing_id": "a", "ranked_listing_ids": ["a", "x"]},
            {"source_listing_id": "b", "ranked_listing_ids": ["x", "y", "b"]},
            {"source_listing_id": "c", "ranked_listing_ids": ["x"]},
        ]
    )

    assert metrics == {
        "queries": 3,
        "recall_at_1": 1 / 3,
        "recall_at_5": 2 / 3,
        "recall_at_10": 2 / 3,
        "mrr": (1 + 1 / 3) / 3,
    }


def test_evaluate_dense_uses_injected_embedder_without_filters_or_reranker():
    class FakeEmbedder:
        vectors = {
            "query a": [1.0, 0.0],
            "query b": [0.0, 1.0],
            "doc a": [1.0, 0.0],
            "doc b": [0.0, 1.0],
            "doc c": [0.6, 0.4],
        }

        def encode(self, texts, **_kwargs):
            return [self.vectors[text] for text in texts]

    report = evaluate_dense(
        [
            {"source_listing_id": "a", "query": "query a"},
            {"source_listing_id": "b", "query": "query b"},
        ],
        [
            {"id": "a", "enriched_doc": "doc a"},
            {"id": "b", "enriched_doc": "doc b"},
            {"id": "c", "enriched_doc": "doc c"},
        ],
        FakeEmbedder(),
    )

    assert report["metrics"]["recall_at_1"] == 1.0
    assert report["cases"][0]["ranked_listing_ids"] == ["a", "c", "b"]


def test_write_result_creates_json_artifact(tmp_path):
    output = tmp_path / "result.json"
    write_result({"metrics": {"mrr": 1.0}}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"metrics": {"mrr": 1.0}}
