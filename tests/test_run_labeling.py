import json
from pathlib import Path

import pytest

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


def test_text_prompt_contains_only_title_description_and_current_null_schema():
    from labeling.run_labeling import build_text_prompt

    record = _sample_record()
    record["attributes"] = {"private": "must not leak"}
    prompt = build_text_prompt(record)

    assert record["title"] in prompt
    assert record["description"] in prompt
    assert "property_features" not in prompt.lower()
    assert "must not leak" not in prompt
    assert "has_balcony" in prompt
    assert "has_fiber" in prompt
    assert "has_elevator" not in prompt


def test_deepseek_filter_merge_fills_only_null_and_rejects_unknown_enum():
    from labeling.run_labeling import merge_filter_values, normalize_text_prediction

    record = _sample_record()
    record["filter_values"] = {"has_elevator": True, "has_balcony": None, "balcony_type": None}
    record["filter_sources"] = {"has_elevator": "scraper_property_feature"}
    prediction = normalize_text_prediction(
        {
            "filters": {
                "has_elevator": False,
                "has_balcony": True,
                "balcony_type": ["uydurma"],
                "has_fiber": True,
                "not_a_filter": True,
            }
        },
        record,
    )

    values, sources = merge_filter_values(record, prediction, "deepseek_description")

    assert values["has_elevator"] is True
    assert sources["has_elevator"] == "scraper_property_feature"
    assert values["has_balcony"] is True
    assert sources["has_balcony"] == "deepseek_description"
    assert values["has_fiber"] is True
    assert values["balcony_type"] is None
    assert "not_a_filter" not in values


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


def test_build_vision_prompt_contains_only_null_image_permitted_filters():
    from labeling.run_labeling import build_vision_prompt

    record = _sample_record()
    prompt = build_vision_prompt(record)

    assert "has_balcony" in prompt
    assert "has_aircon" in prompt
    assert "has_elevator" not in prompt
    assert "near_metro" not in prompt
    assert "salon_ozellikleri" not in prompt


def test_kimi_filter_merge_is_null_only_and_boolean_true_only():
    from labeling.run_labeling import merge_filter_values, normalize_visual_filter_prediction

    record = _sample_record()
    record["filter_values"] = {"has_elevator": True, "has_aircon": None, "has_balcony": None, "balcony_type": None}
    record["filter_sources"] = {"has_elevator": "scraper_property_feature"}
    prediction = normalize_visual_filter_prediction(
        {
            "filters": {
                "has_elevator": False,
                "has_aircon": False,
                "has_balcony": True,
                "balcony_type": ["Açık Balkon"],
                "near_metro": True,
            }
        },
        record,
    )

    values, sources = merge_filter_values(record, prediction, "kimi_image")

    assert values["has_elevator"] is True
    assert sources["has_elevator"] == "scraper_property_feature"
    assert values["has_aircon"] is None
    assert values["has_balcony"] is True
    assert values["balcony_type"] == ["acik_balkon"]
    assert sources["has_balcony"] == "kimi_image"
    assert values["near_metro"] is None


def test_first_kimi_validation_defaults_to_unchunked_calls():
    from labeling.run_labeling import DEFAULT_VISION_CHUNK_SIZE

    assert DEFAULT_VISION_CHUNK_SIZE == 0


def test_extract_visual_labels_persists_kimi_canonical_filters(monkeypatch):
    from llm.clients import candidate_by_id
    from labeling import run_labeling

    record = _sample_record()
    record["filter_values"] = {"has_balcony": None, "has_aircon": None}
    record["filter_sources"] = {}
    monkeypatch.setattr(run_labeling, "_resolve_image_paths", lambda _record: ["image-a.jpg"])
    monkeypatch.setattr(
        run_labeling,
        "complete_vision_json",
        lambda *_args, **_kwargs: json.dumps({"filters": {"has_balcony": True, "has_aircon": False}, "confidence": 0.9}),
    )

    visual = run_labeling.extract_visual_labels(
        record,
        candidate_by_id("kimi_k2_6"),
        run_labeling.CostTracker(max_cost_usd=1.0),
        text_imkanlar=[],
    )

    assert visual["filter_values"]["has_balcony"] is True
    assert visual["filter_sources"]["has_balcony"] == "kimi_image"
    assert visual["filter_values"]["has_aircon"] is None


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


def test_text_phase_preserves_second_pass_inputs_and_accepts_explicit_false(monkeypatch):
    from llm.clients import candidate_by_id
    from labeling import run_labeling

    record = _sample_record()
    monkeypatch.setattr(
        run_labeling,
        "extract_text_labels",
        lambda *_args, **_kwargs: {"has_balcony": False, "imkanlar": []},
    )
    monkeypatch.setattr(
        run_labeling,
        "extract_visual_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("vision must not run")),
    )

    labeled = run_labeling.label_text_record(
        record,
        candidate_by_id("deepseek_v4_pro"),
        cost_tracker=run_labeling.CostTracker(max_cost_usd=1.0),
    )

    assert labeled["description"] == record["description"]
    assert labeled["all_image_paths"] == record["all_image_paths"]
    assert labeled["filter_values"]["has_balcony"] is False
    assert labeled["filter_sources"]["has_balcony"] == "deepseek_description"
    assert labeled["labeling_metadata"]["text_model"] == "deepseek_v4_pro"
    assert "vision_model" not in labeled["labeling_metadata"]


def test_vision_phase_uses_text_output_without_calling_text_and_keeps_false_unknown(monkeypatch):
    from llm.clients import candidate_by_id
    from labeling import run_labeling

    record = _sample_record()
    record["filter_values"] = {"has_balcony": False, "has_aircon": None}
    record["filter_sources"] = {"has_balcony": "deepseek_description"}
    record["facts_gold"] = {"has_balcony": False}
    record["visual_qualities"] = {"aggregated": {"imkanlar": ["kapali_otopark"]}}
    record["labeling_metadata"] = {"text_model": "deepseek_v4_pro"}
    monkeypatch.setattr(
        run_labeling,
        "extract_text_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("text must not run")),
    )
    monkeypatch.setattr(
        run_labeling,
        "extract_visual_labels",
        lambda *_args, **_kwargs: {
            "per_image": [],
            "aggregated": {"balkon_ozellikleri": None, "imkanlar": ["kapali_otopark"]},
            "filter_values": {"has_balcony": False, "has_aircon": None},
            "filter_sources": {"has_balcony": "deepseek_description"},
            "confidence_method": "self",
        },
    )

    labeled = run_labeling.label_vision_record(
        record,
        candidate_by_id("kimi_k2_6"),
        cost_tracker=run_labeling.CostTracker(max_cost_usd=1.0),
    )

    assert labeled["filter_values"]["has_balcony"] is False
    assert labeled["filter_values"]["has_aircon"] is None
    assert labeled["labeling_metadata"]["text_model"] == "deepseek_v4_pro"
    assert labeled["labeling_metadata"]["vision_model"] == "kimi_k2_6"


def test_validate_environment_checks_only_models_used_by_phase(monkeypatch):
    from labeling import run_labeling

    checked = []
    monkeypatch.setattr(run_labeling, "candidate_by_id", lambda model_id: model_id)
    monkeypatch.setattr(run_labeling, "missing_environment", lambda candidate: checked.append(candidate) or [])

    run_labeling._validate_environment("text-model", "vision-model", phase="text")
    assert checked == ["text-model"]

    checked.clear()
    run_labeling._validate_environment("text-model", "vision-model", phase="vision")
    assert checked == ["vision-model"]


def test_clean_json_row_groups_only_readable_source_facts_without_image_paths():
    from labeling.run_labeling import clean_json_row

    row = {
        "id": "1",
        "url": "https://example.test/1",
        "title": "Balkonlu daire",
        "description": "Metroya yakın.",
        "image_path": "data/images/1/00.jpg",
        "all_image_paths": ["data/images/1/00.jpg"],
        "filter_values": {
            "price_amount": 30000,
            "has_elevator": True,
            "has_balcony": False,
            "near_metro": True,
            "balcony_type": ["acik_balkon"],
            "has_aircon": None,
        },
        "filter_sources": {
            "price_amount": "scraper_info",
            "has_elevator": "scraper_property_feature",
            "has_balcony": "deepseek_description",
            "near_metro": "kimi_image",
            "balcony_type": "kimi_image",
        },
    }

    assert clean_json_row(row) == {
        "id": "1",
        "url": "https://example.test/1",
        "title": "Balkonlu daire",
        "description": "Metroya yakın.",
        "scraper": {
            "price_amount": 30000,
            "has_elevator": True,
        },
        "deepseek": {
            "has_balcony": False,
        },
        "kimi": {
            "near_metro": True,
        },
    }


def test_run_labeling_refreshes_clean_json_after_each_written_row(tmp_path, monkeypatch):
    from labeling import run_labeling

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    clean_path = tmp_path / "clean_json.json"
    input_path.write_text(
        "\n".join(json.dumps({**_sample_record(), "id": value}) for value in ("1", "2")) + "\n",
        encoding="utf-8",
    )

    def fake_label_record(record, *_args, **_kwargs):
        return {
            "id": record["id"],
            "url": record["url"],
            "title": record["title"],
            "description": record["description"],
            "filter_values": {"has_balcony": True},
            "filter_sources": {"has_balcony": "deepseek_description"},
        }

    refresh_snapshots = []

    def capture_refresh(source_path, clean_json_path):
        refresh_snapshots.append([row["id"] for row in run_labeling.load_jsonl(source_path)])
        return run_labeling.write_clean_json(source_path, clean_json_path)

    monkeypatch.setattr(run_labeling, "label_record", fake_label_record)
    monkeypatch.setattr(run_labeling, "_refresh_clean_json", capture_refresh)

    run_labeling.run_labeling(
        input_path=input_path,
        output_path=output_path,
        clean_json_path=clean_path,
        text_model_id="kimi_k2_6",
        vision_model_id="kimi_k2_6",
        batch_size=1,
        resume=False,
        max_cost_usd=1.0,
    )

    assert refresh_snapshots == [[], ["1"], ["1", "2"]]
    assert json.loads(clean_path.read_text(encoding="utf-8")) == [
        {
            "id": "1",
            "url": "https://example.test/1",
            "title": "Metroya yakin balkonlu 2+1",
            "description": "Metroya 5 dakika, balkonlu, kapali otoparkli.",
            "scraper": {},
            "deepseek": {"has_balcony": True},
            "kimi": {},
        },
        {
            "id": "2",
            "url": "https://example.test/1",
            "title": "Metroya yakin balkonlu 2+1",
            "description": "Metroya 5 dakika, balkonlu, kapali otoparkli.",
            "scraper": {},
            "deepseek": {"has_balcony": True},
            "kimi": {},
        },
    ]


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


def test_provider_json_call_retries_timeout(monkeypatch):
    from labeling import run_labeling

    calls = []

    def flaky_call():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Request timed out.")
        return "{}"

    monkeypatch.setattr(run_labeling.time, "sleep", lambda _seconds: None)

    assert run_labeling._provider_json_call(flaky_call) == "{}"
    assert len(calls) == 2


def test_parallel_labeling_saves_successful_rows_before_reporting_failed_ids(tmp_path, monkeypatch):
    from labeling import run_labeling

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({**_sample_record(), "id": value}) for value in ("1", "2")) + "\n",
        encoding="utf-8",
    )

    def fake_label_record(record, *_args, **_kwargs):
        if record["id"] == "1":
            raise RuntimeError("Request timed out.")
        return {
            "id": record["id"],
            "url": record["url"],
            "title": record["title"],
            "description": record["description"],
            "filter_values": {},
            "filter_sources": {},
        }

    monkeypatch.setattr(run_labeling, "label_record", fake_label_record)

    with pytest.raises(RuntimeError, match=r"1.*Request timed out"):
        run_labeling.run_labeling(
            input_path=input_path,
            output_path=output_path,
            text_model_id="kimi_k2_6",
            vision_model_id="kimi_k2_6",
            batch_size=2,
            resume=False,
            max_cost_usd=1.0,
        )

    assert [row["id"] for row in run_labeling.load_jsonl(output_path)] == ["2"]
