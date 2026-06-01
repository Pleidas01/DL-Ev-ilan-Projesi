import json

from retrieval.retriever import (
    Retriever,
    extract_query_slots,
    matched_filter_labels,
    slots_to_where,
)


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
    if "$or" in where:
        return any(_matches(metadata, condition) for condition in where["$or"])
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


def test_slots_to_where_maps_canonical_numeric_enum_multi_enum_boolean_and_any_of_filters():
    where = slots_to_where(
        {
            "hard_filters": {
                "filters": {
                    "room_count": ["2+1", "3+1"],
                    "district": ["Kadikoy"],
                    "price_currency": "TL",
                    "has_elevator": True,
                    "near_metro": True,
                    "balcony_type": ["acik_balkon"],
                },
                "any_of": [{"has_open_parking": True, "has_closed_parking": True}],
                "max_price_amount": 30000,
                "min_gross_size_m2": 80,
            },
        }
    )

    assert where == {
        "$and": [
            {"room_count": {"$in": ["2+1", "3+1"]}},
            {"district": {"$in": ["Kadikoy"]}},
            {"price_currency": "TL"},
            {"has_elevator": True},
            {"near_metro": True},
            {"balcony_type__acik_balkon": True},
            {"$or": [{"has_open_parking": True}, {"has_closed_parking": True}]},
            {"price_amount": {"$lte": 30000}},
            {"gross_size_m2": {"$gte": 80}},
        ]
    }


def test_slots_to_where_ignores_registry_outside_enum_values_from_llm():
    where = slots_to_where(
        {
            "hard_filters": {
                "filters": {
                    "trade_type": "for_rent",
                    "property_type": "apartment",
                    "district": ["Kadikoy"],
                },
            },
        }
    )

    assert where == {"district": {"$in": ["Kadikoy"]}}


def test_slots_to_where_flattens_single_valid_any_of_alternative_for_real_chroma():
    where = slots_to_where(
        {
            "hard_filters": {
                "any_of": [{"has_open_parking": True}],
            },
        }
    )

    assert where == {"has_open_parking": True}


def test_matched_filter_labels_render_satisfied_query_filters_as_turkish_chips():
    """Match reason çipleri: sorgunun istediği VE ilanın karşıladığı filtreler.

    WHY: PROJECT §2 her sonuç için 'neden eşleşti' ister. Bu deterministik (Rule 5:
    kod cevaplıyor) — slot + metadata'dan türetilir, LLM'e bırakılmaz. Çip kullanıcıya
    Türkçe etiket gösterir (slug/bool değil).
    """
    slots = {
        "hard_filters": {
            "filters": {
                "room_count": ["2+1"],
                "district": ["Kadikoy"],
                "has_elevator": True,
                "balcony_type": ["acik_balkon"],
            },
        },
    }
    metadata = {
        "room_count": "2+1",
        "district": "Kadikoy",
        "has_elevator": True,
        "balcony_type": "acik_balkon",
        "balcony_type__acik_balkon": True,
    }

    labels = matched_filter_labels(slots, metadata)

    assert labels == ["2+1", "Kadikoy", "Asansör", "Açık Balkon"]


def test_matched_filter_labels_drops_a_filter_the_listing_does_not_satisfy():
    """Çip yalnız gerçekten karşılanan filtre için çıkar.

    WHY (Rule 9): where-clause gevşetilirse veya metadata değişirse çip yanlış
    'neden eşleşti' iddia etmemeli. has_elevator metadata'da yoksa çip kaybolur.
    """
    slots = {"hard_filters": {"filters": {"district": ["Kadikoy"], "has_elevator": True}}}
    metadata = {"district": "Kadikoy"}  # asansör bilgisi yok

    labels = matched_filter_labels(slots, metadata)

    assert labels == ["Kadikoy"]


def test_matched_filter_labels_render_only_the_satisfied_any_of_alternative():
    """any_of (otoparklı = açık VEYA kapalı): ilanda hangisi varsa o çip çıkar."""
    slots = {"hard_filters": {"filters": {}, "any_of": [{"has_open_parking": True, "has_closed_parking": True}]}}
    metadata = {"has_closed_parking": True, "has_closed_parking__x": False}

    labels = matched_filter_labels(slots, metadata)

    assert labels == ["Kapalı Otopark"]


def test_retrieve_filters_out_over_budget_and_wrong_district_before_reranking():
    collection = FakeCollection(
        [
            {
                "id": "ok",
                "document": "uygun ilan",
                "metadata": {"title": "Uygun", "district": "Kadikoy", "price_amount": 30000, "price_currency": "TL", "room_count": "2+1", "has_elevator": True},
            },
            {
                "id": "expensive",
                "document": "butce ustu ilan",
                "metadata": {"title": "Pahali", "district": "Kadikoy", "price_amount": 30001, "price_currency": "TL", "room_count": "2+1", "has_elevator": True},
            },
            {
                "id": "wrong-district",
                "document": "yanlis ilce ilan",
                "metadata": {"title": "Yanlis", "district": "Besiktas", "price_amount": 20000, "price_currency": "TL", "room_count": "2+1", "has_elevator": True},
            },
            {
                "id": "unknown-elevator",
                "document": "asansor bilinmiyor",
                "metadata": {"title": "Eksik", "district": "Kadikoy", "price_amount": 20000, "price_currency": "TL", "room_count": "2+1"},
            },
        ]
    )
    retriever = Retriever(
        collection=collection,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        slot_extractor=lambda _query: {
            "hard_filters": {
                "filters": {"room_count": ["2+1"], "district": ["Kadikoy"], "price_currency": "TL", "has_elevator": True},
                "max_price_amount": 30000,
            },
        },
    )

    results = retriever.retrieve("Kadikoy 2+1 30 bin alti")

    assert [result["id"] for result in results] == ["ok"]
    assert results[0]["title"] == "Uygun"
    assert results[0]["price_amount"] == 30000


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
