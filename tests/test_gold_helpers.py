from evaluation.gold_helper import extract_hard_filters, search_candidates
from labeling.gold_helper import HYBRID_FACT_FIELDS, suggest_hybrid_facts, suggest_visual_fields
from llm.gold_benchmark import (
    FACTS_GOLD_FIELDS,
    STRUCTURED_FACT_FIELDS,
    VISUAL_GOLD_FIELDS,
    build_prefilled_visual_gold,
    score_against_gold,
)


def test_new_gold_schema_field_counts_are_stable():
    assert len(FACTS_GOLD_FIELDS) == 22
    assert len(STRUCTURED_FACT_FIELDS) == 15
    assert len(HYBRID_FACT_FIELDS) == 7
    assert len(VISUAL_GOLD_FIELDS) == 12
    assert "near_metrobus" in FACTS_GOLD_FIELDS
    assert "has_aircon" in FACTS_GOLD_FIELDS
    assert "heating_type" in STRUCTURED_FACT_FIELDS
    assert "is_furnished" in STRUCTURED_FACT_FIELDS
    assert "depolama_gomme" in VISUAL_GOLD_FIELDS


def test_suggest_hybrid_facts_detects_common_keywords():
    suggested = suggest_hybrid_facts(
        title="Metroya yakin klimali 1+1",
        description="Metrobuse yakin acik otoparkli balkonlu daire.",
        property_features=["Amerikan Mutfak", "Asansor"],
    )

    # 7 hybrid alan suggested dict'te olmalı
    assert set(suggested.keys()) == set(HYBRID_FACT_FIELDS)
    # Structured alanlar artık önerilmiyor
    assert "heating_type" not in suggested
    assert "is_furnished" not in suggested
    assert "in_gated_complex" not in suggested
    # Detections
    assert suggested["near_metro"] is True
    assert suggested["near_metrobus"] is True
    assert suggested["kitchen_type"] == "amerikan_acik"
    assert suggested["has_balcony"] is True
    assert suggested["has_elevator"] is True
    assert suggested["has_parking"] is True
    assert suggested["has_aircon"] is True


def test_suggest_hybrid_facts_handles_turkish_uppercase_normalization():
    # "KLİMALI" (Türkçe kapital İ) doğru tespit edilmeli — Python lower() bug'ı için regression guard
    suggested = suggest_hybrid_facts(
        title="CIHANGIR EŞYALI KLİMALI STÜDYO DAİRE",
        description="",
        property_features=[],
    )
    assert suggested["has_aircon"] is True


def test_build_prefilled_visual_gold_returns_correct_structure():
    prefilled = build_prefilled_visual_gold()
    assert set(prefilled.keys()) == set(VISUAL_GOLD_FIELDS)
    # Multi-select alanlar array, single-select '|'li string
    assert isinstance(prefilled["manzara"], list)
    assert "deniz" in prefilled["manzara"]
    assert isinstance(prefilled["balkon_tipi"], str)
    assert "cam_balkon" in prefilled["balkon_tipi"]
    assert " | " in prefilled["balkon_tipi"]


def test_suggest_visual_fields_cross_validates_property_features():
    suggested = suggest_visual_fields(["Somine", "Kapali Otopark", "Havuz", "Gomme Dolap"])

    assert suggested["salon_ozellikleri"] == ["somine"]
    assert suggested["site_imkanlari"] == ["havuz", "kapali_otopark"]
    assert suggested["depolama_gomme"] == ["gomme_dolap_yatak"]


def test_score_against_gold_skips_null_gold_fields_and_scores_jaccard():
    scored = score_against_gold(
        {"has_elevator": True, "site_imkanlari": ["havuz", "guvenlik_kabini"]},
        {"has_elevator": True, "near_metro": None, "site_imkanlari": ["havuz", "acik_otopark"]},
        ("has_elevator", "near_metro", "site_imkanlari"),
    )

    assert scored["scored_fields"] == 2
    assert scored["correct_fields"] == 1
    assert scored["accuracy"] == 2 / 3
    assert scored["per_field"]["near_metro"]["match"] is None
    assert scored["per_field"]["site_imkanlari"]["field_score"] == 1 / 3


def test_extract_hard_filters_from_turkish_query():
    filters = extract_hard_filters("Kadikoy'de 45 bin TL alti esyali 2+1")

    assert filters["rooms"] == "2+1"
    assert filters["max_price_tl"] == 45000
    assert any("kad" in district for district in filters["districts"])


def test_search_candidates_returns_ranked_rows():
    records = [
        {
            "id": "1",
            "title": "Kadikoy 2+1",
            "text": "Kadikoy merkez 2+1 daire 40000 TL",
            "price": "40000 TL",
            "district": "Kadikoy",
            "attributes": {"roomCount": "2+1"},
        },
        {
            "id": "2",
            "title": "Pendik 3+1",
            "text": "Pendik 3+1 daire 50000 TL",
            "price": "50000 TL",
            "district": "Pendik",
            "attributes": {"roomCount": "3+1"},
        },
    ]

    results = search_candidates("kadikoy 2+1 45 bin alti", records, top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "1"
