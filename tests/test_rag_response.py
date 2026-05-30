import json

from chat.rag_response import compose_answer


def _results():
    return [
        {
            "id": "19387277",
            "score": 0.91,
            "title": "Moda'da balkonlu 2+1",
            "price_tl": 30000,
            "facts": {"district": "Kadikoy", "room_count": "2+1", "has_balcony": True},
            "enriched_doc": "Baslik: Moda'da balkonlu 2+1\nAciklama: Metroya yakin.",
        }
    ]


def test_compose_answer_returns_llm_answer_with_listing_citation():
    answer = compose_answer(
        "Kadikoy'de balkonlu 2+1",
        _results(),
        candidate="fake-candidate",
        llm_fn=lambda *_args: json.dumps(
            {"answer": "[ilan:19387277] Moda'da balkonlu 2+1, 30.000 TL. Balkonlu olduğu için sorgunuza uyuyor."}
        ),
    )

    assert "[ilan:19387277]" in answer
    assert "30.000 TL" in answer


def test_compose_answer_does_not_call_llm_when_retrieval_is_empty():
    def fail_if_called(*_args):
        raise AssertionError("LLM must not be called for an empty retrieval result")

    answer = compose_answer("Kadikoy'de 2+1", [], llm_fn=fail_if_called)

    assert answer == "Aramanıza uygun ilan bulunamadı."


def test_compose_answer_reads_selected_model_and_sends_grounded_prompt(tmp_path, monkeypatch):
    from chat import rag_response

    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"text_model": "deepseek_v4_pro"}), encoding="utf-8")
    model_ids = []
    calls = []
    monkeypatch.setattr(rag_response, "candidate_by_id", lambda model_id: model_ids.append(model_id) or "selected-candidate")

    def fake_llm(candidate, system_prompt, user_prompt):
        calls.append((candidate, system_prompt, user_prompt))
        return json.dumps({"answer": "[ilan:19387277] Uygun ilan."})

    compose_answer("Kadikoy'de balkonlu ev", _results(), llm_fn=fake_llm, selected_path=selected_path)

    assert model_ids == ["deepseek_v4_pro"]
    assert calls[0][0] == "selected-candidate"
    assert "[ilan:<id>]" in calls[0][1]
    assert "Kadikoy'de balkonlu ev" in calls[0][2]
    assert "19387277" in calls[0][2]
    assert "Moda'da balkonlu 2+1" in calls[0][2]
    assert "30000" in calls[0][2]
    assert "Metroya yakin" in calls[0][2]


def test_ui_search_reports_missing_index_without_crashing():
    from ui.app import _run_search

    def missing_index():
        raise RuntimeError("Collection [listings] does not exist")

    retriever, results, answer, error = _run_search("Kadikoy 2+1", missing_index)

    assert retriever is None
    assert results == []
    assert answer is None
    assert error == "Arama başlatılamadı: Collection [listings] does not exist"
