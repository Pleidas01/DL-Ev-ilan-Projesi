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


def test_resolve_image_paths_accepts_windows_separators_after_transfer(tmp_path, monkeypatch):
    from labeling import run_labeling

    image_path = tmp_path / "data" / "images" / "1" / "00.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    monkeypatch.setattr(run_labeling, "PROJECT_ROOT", tmp_path)

    resolved = run_labeling._resolve_image_paths({"all_image_paths": [r"data\images\1\00.jpg"]})

    assert resolved == [str(image_path)]


def test_label_record_applies_visual_fallback_to_hybrid_facts(monkeypatch):
    from llm.clients import candidate_by_id
    from labeling import run_labeling

    record = _sample_record()
    record["has_balcony"] = None
    record["has_parking"] = None

    monkeypatch.setattr(
        run_labeling,
        "extract_text_labels",
        lambda *_args, **_kwargs: {
            "has_balcony": None,
            "has_elevator": None,
            "has_parking": None,
            "has_aircon": None,
            "near_metro": None,
            "near_metrobus": None,
            "imkanlar": [],
        },
    )
    monkeypatch.setattr(
        run_labeling,
        "extract_visual_labels",
        lambda *_args, **_kwargs: {
            "per_image": [],
            "aggregated": {
                "balkon_ozellikleri": ["acik_balkon"],
                "manzara": None,
                "mutfak_tipi": None,
                "banyo_ozellikleri": None,
                "salon_ozellikleri": None,
                "imkanlar": ["kapali_otopark"],
            },
            "confidence_method": "self",
        },
    )

    labeled = run_labeling.label_record(
        record,
        candidate_by_id("kimi_k2_6"),
        candidate_by_id("kimi_k2_6"),
        cost_tracker=run_labeling.CostTracker(max_cost_usd=1.0),
    )

    assert labeled["facts_gold"]["has_balcony"] is True
    assert labeled["facts_gold"]["has_parking"] is True


def test_build_vision_prompt_reuses_shootout_winner():
    from labeling.run_labeling import build_vision_prompt
    from llm.shootout_vision import VISION_USER_PROMPT

    assert build_vision_prompt(9) == VISION_USER_PROMPT


def test_visual_aggregation_accepts_visual_gold_wrapper():
    from labeling.run_labeling import aggregate_visual_qualities

    result = aggregate_visual_qualities(
        {
            "visual_gold": {
                "mutfak_tipi": "amerikan_acik",
                "banyo_ozellikleri": ["dusakabin"],
                "imkanlar": ["spor_alani"],
            }
        },
        image_paths=["image-a.jpg"],
        text_imkanlar=["kapali_otopark"],
        min_confidence=0.7,
    )

    assert result["aggregated"]["mutfak_tipi"] == "amerikan_acik"
    assert result["aggregated"]["banyo_ozellikleri"] == ["dusakabin"]
    assert result["aggregated"]["imkanlar"] == ["kapali_otopark", "spor_alani"]


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


def test_passes_thresholds_uses_facts_all_not_facts_llm():
    """Gate full pipeline (facts_all) skoruna bağlı olmalı; text-only modelin
    HYBRID alanlarda null-vs-false yüzünden düşük aldığı facts_llm'e DEĞİL.
    Aksi halde güçlü bir text modeli yapısal artefakt yüzünden haksızca elenir."""
    from labeling.run_labeling import FACTS_THRESHOLD, score_preflight

    structured = {
        "city": "istanbul",
        "district": "kadikoy",
        "neighborhood": "caddebostan",
        "price_tl": 30000,
        "room_count": "1+1",
        "gross_size_m2": 75,
        "net_size_m2": 65,
        "building_age": 5,
        "floor": 3,
        "total_floors": 8,
        "deposit_tl": 30000,
        "in_gated_complex": True,
        "title_deed_status": "kat_mulkiyeti",
        "heating_type": "dogalgaz",
        "is_furnished": True,
    }
    visual = {
        "balkon_ozellikleri": ["cam_balkon"],
        "manzara": ["deniz"],
        "mutfak_tipi": "amerikan_acik",
        "banyo_ozellikleri": ["dusakabin"],
        "salon_ozellikleri": ["somine"],
        "imkanlar": ["havuz"],
    }
    predictions = [
        {
            "id": "1",
            "facts_gold": {
                **structured,
                # text-only model: kanıt yoksa null üretir (asla false demez)
                "has_balcony": None,
                "has_elevator": None,
                "has_parking": None,
                "has_aircon": None,
                "near_metro": None,
                "near_metrobus": True,
            },
            "visual_qualities": {"aggregated": dict(visual)},
        }
    ]
    gold_rows = [
        {
            "listing_id": "1",
            "facts_gold": {
                **structured,
                "has_balcony": False,
                "has_elevator": False,
                "has_parking": False,
                "has_aircon": False,
                "near_metro": False,
                "near_metrobus": True,
            },
            "visual_gold": dict(visual),
        }
    ]

    report = score_preflight(predictions, gold_rows)

    # facts_llm (yalnız 6 HYBRID/desc alan): 5 null-vs-false + 1 doğru → eşiğin altında
    assert report["facts_llm"]["accuracy"] < FACTS_THRESHOLD
    # facts_all (structured dahil tam pipeline) eşiğin üstünde
    assert report["facts_all"]["accuracy"] >= FACTS_THRESHOLD
    # Gate facts_all'a baktığı için GEÇER; eski facts_llm davranışında KALIRDI
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
