from schema.emlakjet_filters import (
    EMLAKJET_FILTERS,
    PROPERTY_FEATURE_SPECS,
    empty_filter_values,
    extract_scraper_filter_facts,
    normalize_label,
    spec_for_info_label,
    spec_for_property_feature,
)


def test_registry_covers_representative_live_rental_panel_fields():
    by_slug = {spec.slug: spec for spec in EMLAKJET_FILTERS}

    assert by_slug["trade_type"].values == {"Kiralık": "kiralik"}
    assert by_slug["property_category"].values == {"Konut": "konut"}
    assert by_slug["property_type"].values["Yalı Dairesi"] == "yali_dairesi"
    assert by_slug["price_amount"].group == "structured"
    assert by_slug["price_currency"].values == {"TL": "TL", "USD": "USD", "EUR": "EUR", "GBP": "GBP"}
    assert by_slug["balcony_type"].values["Kapalı Teras"] == "kapali_teras"
    assert by_slug["has_fiber"].group == "ic_ozellikler.altyapi"
    assert by_slug["has_shower_cabin"].group == "ic_ozellikler.banyo"
    assert by_slug["has_aircon"].group == "ic_ozellikler.dekorasyon"
    assert by_slug["has_satin_paint"].group == "ic_ozellikler.dekorasyon"
    assert by_slug["has_american_kitchen"].group == "ic_ozellikler.mutfak"
    assert by_slug["has_elevator"].group == "dis_ozellikler.bina"
    assert by_slug["has_closed_parking"].group == "dis_ozellikler.sosyal_imkanlar"
    assert by_slug["has_bosphorus_view"].group == "konum_ozellikleri.manzara"
    assert by_slug["near_metro"].group == "konum_ozellikleri.ulasim"
    assert len(PROPERTY_FEATURE_SPECS) == 124


def test_registry_slugs_and_source_labels_are_unique():
    slugs = [spec.slug for spec in EMLAKJET_FILTERS]
    info_labels = [
        normalize_label(label)
        for spec in EMLAKJET_FILTERS
        if "listing_info" in spec.sources
        for label in spec.labels
    ]
    feature_labels = [
        normalize_label(label)
        for spec in PROPERTY_FEATURE_SPECS
        for label in spec.labels
    ]

    assert len(slugs) == len(set(slugs))
    assert len(info_labels) == len(set(info_labels))
    assert len(feature_labels) == len(set(feature_labels))


def test_registry_lookup_normalizes_turkish_labels_and_empty_values_are_null():
    assert spec_for_info_label("Balkon Durumu").slug == "has_balcony"
    assert spec_for_info_label("Banyo Sayısı").slug == "bathroom_count"
    assert spec_for_property_feature("Asansör").slug == "has_elevator"
    assert spec_for_property_feature("Metrobüs").slug == "near_metrobus"
    assert spec_for_property_feature("bilinmeyen") is None

    values = empty_filter_values()

    assert set(values) == {spec.slug for spec in EMLAKJET_FILTERS}
    assert all(value is None for value in values.values())


def test_removed_salon_ozellikleri_is_not_a_canonical_filter():
    assert "salon_ozellikleri" not in {spec.slug for spec in EMLAKJET_FILTERS}


def test_info_table_fills_canvas_by_label_including_previously_unmapped_filters():
    """İlan Bilgileri'nin tüm satırları etikete göre kanonik filtreleri doldurmalı.

    has_virtual_tour ('Görüntülü Gezilebilir mi?') registry'de listing_info kaynaklı
    ama _ATTRIBUTE_FILTER_SLUGS'ta yok; eski yol onu asla dolduramıyordu. infoTableAll
    üzerinden etiket çözümü bu boşluğu kapatmalı. Eşlenen alanlar (bathroom_count) da
    sadece tablodan, canonical key olmadan dolabilmeli.
    """
    attrs = {"infoTableAll": {
        "goruntulu gezilebilir mi?": "Evet",
        "banyo sayisi": "2",
    }}

    values, sources = extract_scraper_filter_facts(attrs)

    assert values["has_virtual_tour"] is True
    assert sources["has_virtual_tour"] == "scraper_info"
    assert values["bathroom_count"] == 2
    assert sources["bathroom_count"] == "scraper_info"


def test_info_table_label_resolution_does_not_overwrite_existing_value():
    """Monotonik provenance: etiket döngüsü daha önce set edilmiş değeri ezmemeli."""
    attrs = {
        "bathroomCount": "3",
        "infoTableAll": {"banyo sayisi": "1"},
    }

    values, _ = extract_scraper_filter_facts(attrs)

    assert values["bathroom_count"] == 3
