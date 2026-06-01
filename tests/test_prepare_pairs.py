import json

from finetune.text_embed.prepare_pairs import build_pair_rows, write_pair_dataset


def _records():
    return [
        {"id": "3", "enriched_doc": "doc 3", "filter_values": {"district": "Sisli"}},
        {"id": "1", "enriched_doc": "doc 1", "filter_values": {"district": "Kadikoy"}},
        {"id": "4", "enriched_doc": "doc 4", "filter_values": {"district": "Besiktas"}},
        {"id": "2", "enriched_doc": "doc 2", "filter_values": {"district": "Uskudar"}},
    ]


def test_build_pair_rows_is_deterministic_and_splits_by_listing_id():
    first = build_pair_rows(_records(), validation_ratio=0.5, seed=42)
    second = build_pair_rows(list(reversed(_records())), validation_ratio=0.5, seed=42)

    assert first == second
    train_ids = {row["source_listing_id"] for row in first["train"]}
    validation_ids = {row["source_listing_id"] for row in first["validation"]}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids == set(first["manifest"]["train_listing_ids"])
    assert validation_ids == set(first["manifest"]["validation_listing_ids"])


def test_build_pair_rows_reuses_synthetic_query_and_uses_different_listing_negative():
    dataset = build_pair_rows(_records(), validation_ratio=0.5, seed=42)

    rows = dataset["train"] + dataset["validation"]
    assert {row["source_listing_id"] for row in rows} == {"1", "2", "3", "4"}
    assert next(row for row in rows if row["source_listing_id"] == "1")["query"] == (
        "Kadikoy kiralik daire"
    )
    assert all(row["positive"] == f"doc {row['source_listing_id']}" for row in rows)
    assert all(row["negative_listing_id"] != row["source_listing_id"] for row in rows)


def test_write_pair_dataset_creates_train_validation_and_manifest_json(tmp_path):
    dataset = build_pair_rows(_records(), validation_ratio=0.5, seed=42)

    write_pair_dataset(dataset, tmp_path)

    assert len((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((tmp_path / "validation.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    manifest = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["negative_strategy"] == "deterministic_next_split_listing_id"
