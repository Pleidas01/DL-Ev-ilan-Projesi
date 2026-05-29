import json
from pathlib import Path

from llm.gold_benchmark import VISUAL_GOLD_FIELDS


def _sample_record():
    return {
        "id": "1",
        "url": "https://example.test/1",
        "title": "Metroya yakin balkonlu 2+1",
        "price_tl": 30000,
        "city": "Istanbul",
        "district": "Kadikoy",
        "neighborhood": "Moda Mahallesi",
        "room_count": "2+1",
        "gross_size_m2": 90,
        "net_size_m2": 75,
        "building_age": "5-10",
        "floor": "3.Kat",
        "total_floors": 6,
        "deposit_tl": 30000,
        "in_gated_complex": False,
        "title_deed_status": "Kat Mulkiyeti",
        "heating_type": "kombi",
        "is_furnished": True,
        "has_balcony": None,
        "has_elevator": True,
        "has_parking": None,
        "has_aircon": None,
        "near_metro": None,
        "near_metrobus": None,
        "description": "Metroya 5 dakika, balkonlu, kapali otoparkli.",
        "property_features": ["Asansor", "Kapali Otopark"],
        "all_image_paths": ["image-a.jpg", "image-b.jpg"],
    }


def test_merge_facts_preserves_structured_and_prefers_existing_hybrid():
    from labeling.run_labeling import TEXT_FACT_FIELDS, merge_facts

    record = _sample_record()
    text_prediction = {
        "has_balcony": True,
        "has_elevator": False,
        "has_parking": True,
        "has_aircon": None,
        "near_metro": True,
        "near_metrobus": False,
    }

    merged = merge_facts(record, text_prediction)

    assert merged["heating_type"] == "kombi"
    assert merged["is_furnished"] is True
    assert merged["has_elevator"] is True
    assert merged["has_balcony"] is True
    assert merged["has_parking"] is True
    assert merged["near_metro"] is True
    assert set(TEXT_FACT_FIELDS).issubset(merged)
    assert len(merged) == 21


def test_visual_aggregation_unions_text_and_high_confidence_visual_imkanlar():
    from labeling.run_labeling import aggregate_visual_qualities

    parsed = {
        "per_image": [
            {
                "image_index": 0,
                "fields": {
                    "manzara": ["deniz"],
                    "mutfak_tipi": "amerikan_acik",
                    "banyo_ozellikleri": ["dusakabin"],
                    "imkanlar": ["spor_alani"],
                },
                "confidence": 0.91,
            },
            {
                "image_index": 1,
                "fields": {
                    "manzara": ["bogaz"],
                    "mutfak_tipi": "kapali_ayri",
                    "balkon_ozellikleri": ["fransiz_balkon"],
                    "imkanlar": ["acik_otopark"],
                },
                "confidence": 0.42,
            },
        ]
    }

    result = aggregate_visual_qualities(
        parsed,
        image_paths=["image-a.jpg", "image-b.jpg"],
        text_imkanlar=["kapali_otopark"],
        min_confidence=0.7,
        confidence_method="self",
    )

    assert result["aggregated"]["manzara"] == ["deniz"]
    assert result["aggregated"]["mutfak_tipi"] == "amerikan_acik"
    assert result["aggregated"]["balkon_ozellikleri"] is None
    assert result["aggregated"]["imkanlar"] == ["kapali_otopark", "spor_alani"]
    assert {field for field in result["aggregated"]} == set(VISUAL_GOLD_FIELDS)


def test_score_preflight_reports_llm_facts_separately():
    from labeling.run_labeling import score_preflight

    predictions = [
        {
            "id": "1",
            "facts_gold": {
                **{field: None for field in (
                    "city",
                    "district",
                    "neighborhood",
                    "price_tl",
                    "room_count",
                    "gross_size_m2",
                    "net_size_m2",
                    "building_age",
                    "floor",
                    "total_floors",
                    "deposit_tl",
                    "in_gated_complex",
                    "title_deed_status",
                    "heating_type",
                    "is_furnished",
                )},
                "has_balcony": True,
                "has_elevator": True,
                "has_parking": False,
                "has_aircon": None,
                "near_metro": True,
                "near_metrobus": None,
            },
            "visual_qualities": {
                "aggregated": {
                    "balkon_ozellikleri": ["fransiz_balkon"],
                    "manzara": ["deniz"],
                    "mutfak_tipi": "amerikan_acik",
                    "banyo_ozellikleri": [],
                    "salon_ozellikleri": None,
                    "imkanlar": ["kapali_otopark"],
                }
            },
        }
    ]
    gold_rows = [
        {
            "listing_id": "1",
            "facts_gold": {
                "has_balcony": True,
                "has_elevator": False,
                "has_parking": False,
                "has_aircon": None,
                "near_metro": True,
                "near_metrobus": None,
            },
            "visual_gold": {
                "balkon_ozellikleri": ["fransiz_balkon"],
                "manzara": ["deniz", "bogaz"],
                "mutfak_tipi": "amerikan_acik",
                "banyo_ozellikleri": [],
                "salon_ozellikleri": None,
                "imkanlar": ["kapali_otopark"],
            },
        }
    ]

    report = score_preflight(predictions, gold_rows)

    assert report["facts_llm"]["accuracy"] == 0.75
    assert report["visual"]["accuracy"] == 0.9
    assert "visual_no_imkanlar" not in report
    assert "imkanlar_text" not in report
    assert report["passes_thresholds"] is True


def test_resume_skips_existing_output_ids(tmp_path, monkeypatch):
    from labeling import run_labeling

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({**_sample_record(), "id": value}) for value in ("1", "2")) + "\n",
        encoding="utf-8",
    )
    output_path.write_text(json.dumps({"id": "1"}) + "\n", encoding="utf-8")

    called = []

    def fake_label_record(record, *_args, **_kwargs):
        called.append(record["id"])
        return {"id": record["id"], "facts_gold": {}, "visual_qualities": {"aggregated": {}}, "enriched_doc": ""}

    monkeypatch.setattr(run_labeling, "label_record", fake_label_record)

    rows = run_labeling.run_labeling(
        input_path=input_path,
        output_path=output_path,
        text_model_id="kimi_k2_6",
        vision_model_id="kimi_k2_6",
        batch_size=1,
        resume=True,
        max_cost_usd=1.0,
    )

    assert called == ["2"]
    assert [row["id"] for row in rows] == ["2"]


def test_text_extraction_retries_transient_concurrency_rate_limit(monkeypatch):
    from llm.clients import candidate_by_id
    from labeling import run_labeling

    calls = []

    def flaky_complete_json(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("request reached max organization concurrency: 3")
        return json.dumps({
            "facts": {
                "has_balcony": True,
                "has_elevator": None,
                "has_parking": None,
                "has_aircon": None,
                "near_metro": True,
                "near_metrobus": None,
            },
            "imkanlar": ["spor_alani"],
        })

    monkeypatch.setattr(run_labeling, "complete_json", flaky_complete_json)
    monkeypatch.setattr(run_labeling.time, "sleep", lambda _seconds: None)

    result = run_labeling.extract_text_labels(
        _sample_record(),
        candidate_by_id("kimi_k2_6"),
        run_labeling.CostTracker(max_cost_usd=1.0),
    )

    assert len(calls) == 2
    assert result["has_balcony"] is True
    assert result["near_metro"] is True
    assert result["imkanlar"] == ["spor_alani"]
