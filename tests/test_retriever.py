import json

from retrieval.retriever import Retriever, extract_query_slots, slots_to_where


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        return [[float(len(text))] for text in texts]


class FakeReranker:
    def predict(self, pairs):
        return [float(len(document)) for _query, document in pairs]


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.last_where = None

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        self.last_where = where
        rows = [row for row in self.rows if _matches(row["metadata"], where)][:n_results]
        return {
            "ids": [[row["id"] for row in rows]],
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
        }


def _matches(metadata, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, condition) for condition in where["$and"])
    field, expected = next(iter(where.items()))
    actual = metadata.get(field)
    if not isinstance(expected, dict):
        return actual == expected
    operator, value = next(iter(expected.items()))
    if operator == "$in":
        return actual in value
    if operator == "$lte":
        return actual <= value
    if operator == "$gte":
        return actual >= value
    raise AssertionError(f"Unexpected operator: {operator}")


def test_slots_to_where_maps_structured_and_boolean_filters():
    where = slots_to_where(
        {
            "hard_filters": {
                "rooms": ["2+1", "3+1"],
                "districts": ["Kadikoy"],
                "max_price_tl": 30000,
                "min_size_m2": 80,
            },
            "soft_features": {"facts_gold": {"near_metro": True, "has_elevator": True}},
        }
    )

    assert where == {
        "$and": [
            {"room_count": {"$in": ["2+1", "3+1"]}},
            {"district": {"$in": ["Kadikoy"]}},
            {"price_tl": {"$lte": 30000}},
            {"gross_size_m2": {"$gte": 80}},
            {"has_elevator": True},
            {"near_metro": True},
        ]
    }


def test_retrieve_filters_out_over_budget_and_wrong_district_before_reranking():
    collection = FakeCollection(
        [
            {
                "id": "ok",
                "document": "uygun ilan",
                "metadata": {"title": "Uygun", "district": "Kadikoy", "price_tl": 30000, "room_count": "2+1"},
            },
            {
                "id": "expensive",
                "document": "butce ustu ilan",
                "metadata": {"title": "Pahali", "district": "Kadikoy", "price_tl": 30001, "room_count": "2+1"},
            },
            {
                "id": "wrong-district",
                "document": "yanlis ilce ilan",
                "metadata": {"title": "Yanlis", "district": "Besiktas", "price_tl": 20000, "room_count": "2+1"},
            },
        ]
    )
    retriever = Retriever(
        collection=collection,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        slot_extractor=lambda _query: {
            "hard_filters": {"rooms": ["2+1"], "districts": ["Kadikoy"], "max_price_tl": 30000, "min_size_m2": None},
            "soft_features": {},
        },
    )

    results = retriever.retrieve("Kadikoy 2+1 30 bin alti")

    assert [result["id"] for result in results] == ["ok"]
    assert results[0]["title"] == "Uygun"
    assert results[0]["price_tl"] == 30000


def test_extract_query_slots_reads_selected_text_model_without_live_api(tmp_path, monkeypatch):
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"text_model": "chosen"}), encoding="utf-8")
    calls = []
    monkeypatch.setattr("retrieval.retriever.candidate_by_id", lambda model_id: f"candidate:{model_id}")
    monkeypatch.setattr(
        "retrieval.retriever.complete_json",
        lambda candidate, system_prompt, user_prompt: calls.append((candidate, system_prompt, user_prompt))
        or '{"hard_filters": {}, "soft_features": {}, "free_form_tr": "Kadikoy"}',
    )

    parsed = extract_query_slots("Kadikoy", selected_path)

    assert parsed["free_form_tr"] == "Kadikoy"
    assert calls[0][0] == "candidate:chosen"
    assert "Kadikoy" in calls[0][2]
