"""UI kart fact'leri: retriever'ın Gen3 `filters` çıktısını okunabilir satırlara çevir.

WHY: Eski kart `result['facts']` okuyordu — retriever Gen3'te `filters` döndürdüğü
için kart hep boştu (bug). Ayrıca CARD_FACTS'te `has_parking` vardı; bu canonical
registry'de YOK (`has_open_parking`/`has_closed_parking`). Enum değerler slug
(`kombi_dogalgaz`) — kullanıcı Türkçe label görmeli.
"""

import inspect

import ui.app as app
from ui.app import card_fact_lines


def test_streamlit_width_api_uses_current_parameter():
    source = inspect.getsource(app)

    assert "use_container_width" not in source
    assert 'width="stretch"' in source


def test_card_reads_filters_and_renders_enum_value_as_turkish_label():
    result = {
        "filters": {
            "district": "Kadikoy",
            "room_count": "2+1",
            "gross_size_m2": 90,
            "heating_type": "kombi_dogalgaz",
        }
    }

    lines = card_fact_lines(result)

    assert "İlçe: Kadikoy" in lines
    assert "Oda: 2+1" in lines
    assert "Brüt m²: 90" in lines
    # Slug değil okunabilir Türkçe label:
    assert "Isıtma: Kombi Doğalgaz" in lines
    assert "kombi_dogalgaz" not in " ".join(lines)


def test_card_renders_boolean_fact_as_var():
    result = {"filters": {"has_balcony": True, "is_furnished": True}}

    lines = card_fact_lines(result)

    assert "Balkon: Var" in lines
    assert "Eşyalı: Var" in lines


def test_card_omits_facts_absent_from_filters():
    result = {"filters": {"district": "Kadikoy"}}

    lines = card_fact_lines(result)

    assert lines == ["İlçe: Kadikoy"]


def test_card_uses_canonical_parking_slug_not_dead_has_parking():
    # has_parking canonical değil; eski kart onu hiç gösteremezdi.
    result = {"filters": {"has_closed_parking": True}}

    lines = card_fact_lines(result)

    assert any("Otopark" in line for line in lines)


def test_card_reads_nothing_from_legacy_facts_key():
    # Geriye dönük: eski `facts` anahtarı artık okunmamalı (göç kanıtı).
    result = {"facts": {"district": "Eski"}, "filters": {}}

    lines = card_fact_lines(result)

    assert lines == []
