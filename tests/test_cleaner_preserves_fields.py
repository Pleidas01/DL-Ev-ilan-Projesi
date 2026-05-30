from scraper.cleaner import clean_record, normalize_text

FACTS_FIELDS = (
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
    "heating_type",
    "has_balcony",
    "has_elevator",
    "has_aircon",
    "is_furnished",
    "deposit_tl",
    "has_parking",
    "in_gated_complex",
    "near_metro",
    "near_metrobus",
    "title_deed_status",
)


def test_normalize_text_strips_truncated_html_tag_at_end():
    assert normalize_text('Aciklama metni <div class="x"') == "Aciklama metni"


def test_clean_record_keeps_m3_source_fields_separate_from_short_text():
    raw = {
        "id": "19189018",
        "url": "https://www.emlakjet.com/ilan/example-19189018",
        "title": "Test ilan",
        "price": "60000 TL",
        "district": "Istanbul - Esenyurt - Necip Fazil",
        "description": "Havuzlu site icinde ferah 3+1 daire. " * 80,
        "attributes": {
            "heating": "Merkezi (Pay Olcer)",
            "netSize": "145 m2",
            "grossSize": "148 m2",
            "floor": "6.Kat",
            "buildingAge": "11-15",
            "totalFloors": "7",
            "roomCount": "3+1",
            "deposit": "60.000 TL",
            "inGatedComplex": "Evet",
            "bathroomCount": "2",
            "occupancy": "Bos",
            "balconyStatus": "Yok",
            "balconyType": "Kapali Teras",
            "titleDeedStatus": "Kat Mulkiyeti",
            "propertyFeatures": ["Kuvet", "Sauna", "Amerikan Mutfak", "Asansor", "Kapali Otopark"],
        },
        "scraped_at": "2026-05-24T00:00:00",
    }

    record = clean_record(
        raw,
        image_path="data/images/19189018/01.jpg",
        all_image_paths=["data/images/19189018/01.jpg"],
    )

    for field in FACTS_FIELDS:
        assert field in record, f"missing field: {field}"

    assert len(record["description"]) > 500
    assert record["city"] == "Istanbul"
    assert record["district"] == "Esenyurt"
    assert record["neighborhood"] == "Necip Fazil"
    assert record["price_tl"] == 60000
    assert record["gross_size_m2"] == 148
    assert record["net_size_m2"] == 145
    assert record["total_floors"] == 7
    assert record["room_count"] == "3+1"
    assert record["deposit_tl"] == 60000
    assert record["in_gated_complex"] is True
    assert record["title_deed_status"] == "Kat Mulkiyeti"
    assert record["heating_type"] == "merkezi"
    assert record["has_balcony"] is False
    assert record["has_elevator"] is True
    assert record["has_parking"] is True
    assert record["has_aircon"] is None  # fixture'da klima feature yok
    assert record["near_metro"] is None
    assert record["near_metrobus"] is None
    assert record["property_features"] == ["Kuvet", "Sauna", "Amerikan Mutfak", "Asansor", "Kapali Otopark"]
    assert record["attributes"] == raw["attributes"]
    assert record["filter_values"]["has_balcony"] is False
    assert record["filter_values"]["balcony_type"] == ["kapali_teras"]
    assert record["filter_values"]["bathroom_count"] == 2
    assert record["filter_values"]["occupancy"] == "bos"
    assert record["filter_values"]["has_elevator"] is True
    assert record["filter_values"]["has_closed_parking"] is True
    assert record["filter_values"]["has_fiber"] is None
    assert record["filter_sources"]["has_balcony"] == "scraper_info"
    assert record["filter_sources"]["has_elevator"] == "scraper_property_feature"
    assert "has_fiber" not in record["filter_sources"]
    assert record["visual_qualities"] == {}
    assert len(record["text"]) <= 256
    assert record["text"] != record["description"]
