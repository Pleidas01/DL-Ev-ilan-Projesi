import json
import sys
from types import SimpleNamespace

import pytest

from indexing.composer import embedding_text, to_metadata


def _record():
    return {
        "id": "19387277",
        "title": "Moda'da balkonlu daire",
        "filter_values": {
            "district": "Kadikoy",
            "price_amount": 30000,
            "price_currency": "TL",
            "room_count": "2+1",
            "gross_size_m2": 90,
            "in_gated_complex": False,
            "near_metro": True,
            "has_aircon": None,
            "balcony_type": ["acik_teras", "acik_balkon"],
        },
        "facts_gold": {
            "city": "Istanbul",
            "district": "Kadikoy",
            "neighborhood": "Moda",
            "price_tl": 30000,
            "room_count": "2+1",
            "gross_size_m2": 90,
            "net_size_m2": 75,
            "building_age": None,
            "floor": "3.Kat",
            "total_floors": 6,
            "deposit_tl": 30000,
            "in_gated_complex": False,
            "title_deed_status": None,
            "heating_type": "kombi",
            "is_furnished": True,
            "has_balcony": True,
            "has_elevator": True,
            "has_parking": False,
            "has_aircon": None,
            "near_metro": True,
            "near_metrobus": False,
        },
        "visual_qualities": {
            "aggregated": {
                "balkon_ozellikleri": ["teras", "acik_balkon"],
                "manzara": ["sehir_panorama", "deniz"],
                "mutfak_tipi": "kapali_ayri",
                "banyo_ozellikleri": None,
                "imkanlar": ["spor_alani", "kapali_otopark"],
            }
        },
        "enriched_doc": "Baslik: Moda'da balkonlu daire\n\nAciklama: Ferah.",
    }


def test_embedding_text_uses_m3_enriched_doc_without_recomposition():
    record = _record()

    assert embedding_text(record) == record["enriched_doc"]


def test_embedding_text_fails_loud_when_m3_contract_is_missing():
    with pytest.raises(ValueError, match="enriched_doc"):
        embedding_text({"id": "missing"})


def test_to_metadata_extracts_scalar_filter_fields_and_drops_none_values():
    metadata = to_metadata(_record())

    assert metadata["district"] == "Kadikoy"
    assert metadata["price_amount"] == 30000
    assert metadata["price_currency"] == "TL"
    assert metadata["in_gated_complex"] is False
    assert metadata["near_metro"] is True
    assert "building_age" not in metadata
    assert "has_aircon" not in metadata
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())


def test_to_metadata_joins_visual_lists_for_substring_post_filtering():
    metadata = to_metadata(_record())

    assert metadata["balcony_type"] == "acik_balkon|acik_teras"
    assert metadata["balcony_type__acik_balkon"] is True
    assert metadata["balcony_type__acik_teras"] is True
    assert "salon_ozellikleri" not in metadata


def test_to_metadata_fails_loud_for_registry_outside_multi_enum_option():
    record = _record()
    record["filter_values"]["balcony_type"] = ["acik_balkon_acik_teras"]

    with pytest.raises(ValueError, match="balcony_type"):
        to_metadata(record)


def test_build_index_resume_skips_existing_ids_without_reembedding(tmp_path, monkeypatch):
    from indexing import build_chroma

    existing = _record()
    new = {**_record(), "id": "new", "enriched_doc": "Baslik: Yeni"}
    input_path = tmp_path / "labeled.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in (existing, new)), encoding="utf-8")

    class FakeEmbeddings(list):
        def tolist(self):
            return list(self)

    class FakeEmbedder:
        def __init__(self):
            self.encoded = []
            self.batch_sizes = []

        def encode(self, documents, **kwargs):
            self.encoded.extend(documents)
            self.batch_sizes.append(kwargs["batch_size"])
            return FakeEmbeddings([[1.0] for _document in documents])

    class FakeCollection:
        def __init__(self):
            self.added = []

        def get(self, *, include):
            return {"ids": [existing["id"]]}

        def add(self, **kwargs):
            self.added.append(kwargs)

    collection = FakeCollection()
    client = SimpleNamespace(get_or_create_collection=lambda _name: collection)
    embedder = FakeEmbedder()
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **_kwargs: client))
    monkeypatch.setattr(build_chroma, "_load_embedder", lambda _model_name: embedder)

    written = build_chroma.build_index(input_path=input_path, persist_dir=tmp_path / "chroma")

    assert written == 1
    assert embedder.encoded == ["Baslik: Yeni"]
    assert embedder.batch_sizes == [8]
    assert collection.added[0]["ids"] == ["new"]
