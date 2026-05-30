"""Offline extraction tests using saved Emlakjet detail HTML (no network)."""

from pathlib import Path

from scraper.playwright_scraper import (
    extract_images_from_html,
    build_listing_record,
    is_template_description,
    parse_description_from_dom_html,
    parse_listing_info_table,
    parse_property_features,
)

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "18675059.html"
LISTING_ID = "18675059"
FIXTURE_NO_FEATURES = Path(__file__).parent / "fixtures" / "19387277.html"


def _load_fixture() -> str:
    assert FIXTURE_HTML.exists(), (
        f"Missing {FIXTURE_HTML}. Run: python -m scraper._inspect "
        "--url https://www.emlakjet.com/ilan/tarcanlar-dan-satilik-genis-metrekareye-sahip-emsalsiz-31-daire-18675059"
    )
    return FIXTURE_HTML.read_text(encoding="utf-8")


def test_description_from_dom_is_real_listing_text_not_json_ld_template():
    html = _load_fixture()
    description = parse_description_from_dom_html(html)

    assert len(description) > 200
    assert "TARCANLARDAN" in description.upper() or "prestij" in description.lower()
    assert not is_template_description(description)
    assert "Emlakjet - #" not in description


def test_images_filtered_to_current_listing_id_only():
    html = _load_fixture()
    urls = extract_images_from_html(html, LISTING_ID)

    assert len(urls) >= 5
    assert all(f"/listing/{LISTING_ID}/" in u for u in urls)
    assert not any("/listing/19251034/" in u for u in urls)


def test_ilan_bilgileri_table_fills_missing_data_layer_fields():
    html = _load_fixture()
    info = parse_listing_info_table(html)

    assert info.get("heating") == "Kombi Doğalgaz"
    assert info.get("netSize") == "140 m²"
    assert info.get("grossSize") == "150 m²"
    assert info.get("floor") == "4.Kat"
    assert info.get("buildingAge") == "11-15"
    assert info.get("roomCount") == "3+1"


def test_description_parses_when_next_section_is_konum_not_ozellikler():
    """Bazı ilanlarda İlan Özellikleri yok; açıklama doğrudan Konum Bilgisi'ne bağlanır."""
    if not FIXTURE_NO_FEATURES.exists():
        return
    html = FIXTURE_NO_FEATURES.read_text(encoding="utf-8")
    description = parse_description_from_dom_html(html)

    assert len(description) > 50
    assert "mehmet akif" in description.lower() or "dairemiz" in description.lower()


def test_ilan_bilgileri_table_maps_extended_fact_fields():
    html = """
    <h2>Ä°lan Bilgileri</h2>
    <li><span>Site Ä°Ã§erisinde</span><span>Evet</span></li>
    <li><span>KullanÄ±m Durumu</span><span>BoÅŸ</span></li>
    <li><span>Aidat</span><span>1.250 TL</span></li>
    <li><span>Depozito</span><span>60.000 TL</span></li>
    <li><span>Tapu Durumu</span><span>Kat MÃ¼lkiyeti</span></li>
    <li><span>Takas</span><span>HayÄ±r</span></li>
    <li><span>Banyo SayÄ±sÄ±</span><span>2</span></li>
    <li><span>Balkon Durumu</span><span>Yok</span></li>
    <li><span>Balkon Tipi</span><span>KapalÄ± Teras</span></li>
    <li><span>Fiyat Durumu</span><span>Genel Fiyat</span></li>
    <h2>Ä°lan Ã–zellikleri</h2>
    """

    info = parse_listing_info_table(html)

    assert info["inGatedComplex"] == "Evet"
    assert info["occupancy"] == "BoÅŸ"
    assert info["maintenanceFee"] == "1.250 TL"
    assert info["deposit"] == "60.000 TL"
    assert info["titleDeedStatus"] == "Kat MÃ¼lkiyeti"
    assert info["tradeAccepted"] == "HayÄ±r"
    assert info["bathroomCount"] == "2"
    assert info["balconyStatus"] == "Yok"
    assert info["balconyType"] == "KapalÄ± Teras"
    assert info["priceStatus"] == "Genel Fiyat"


def test_ilan_ozellikleri_bullet_features_extracted():
    html = _load_fixture()
    features = parse_property_features(html)

    assert "Küvet" in features
    assert "ADSL" in features
    assert "Parke" in features
    assert len(features) >= 8


def test_json_listing_record_captures_category_and_multicurrency_price_filters():
    record = build_listing_record({
        "id": "1",
        "title": "USD kiralik daire",
        "price": {"value": "1250", "currency": "USD"},
        "location": {"cityName": "İstanbul", "districtName": "Kadıköy", "neighborhoodName": "Moda"},
        "tradeType": "Kiralık",
        "category": "Konut",
        "propertyType": "Daire",
    })

    assert record["filter_values"]["trade_type"] == "kiralik"
    assert record["filter_values"]["property_category"] == "konut"
    assert record["filter_values"]["property_type"] == "daire"
    assert record["filter_values"]["price_amount"] == 1250
    assert record["filter_values"]["price_currency"] == "USD"
    assert record["filter_sources"]["price_amount"] == "scraper_info"
