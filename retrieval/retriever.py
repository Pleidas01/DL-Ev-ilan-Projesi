from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llm.clients import candidate_by_id, complete_json
from llm.shootout import SLOT_SYSTEM_PROMPT, build_slot_prompt, flatten_slots
from schema.emlakjet_filters import EMLAKJET_FILTERS, label_for, normalize_label, spec_for_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "llm" / "selected.json"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_COLLECTION = "listings"


def extract_query_slots(query: str, selected_path: Path = DEFAULT_SELECTED_PATH) -> dict[str, Any]:
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    candidate = candidate_by_id(selected["text_model"])
    raw_response = complete_json(candidate, SLOT_SYSTEM_PROMPT, build_slot_prompt(query))
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Query parser must return a JSON object")
    return parsed


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# Konum proper-noun'ları index'te ASCII title-case saklanır (örn. 'Kadikoy').
# search_keyword serbest metindir; fold edilmez.
_LOCATION_SLUGS = frozenset({"city", "district", "neighborhood"})


def _canonical_location(value: Any) -> Any:
    """Konum string'ini index-zamanı kanonik formuna çevir: Türkçe→ASCII fold + title-case.

    Slot extractor 'Kadıköy' (Türkçe) veya 'kadıköy' (karışık case) dönebilir; index
    'Kadikoy' saklar. Bu deterministik dönüşüm (Rule 5) hard-filter'ın eşleşmesini sağlar.
    """
    if not isinstance(value, str):
        return value
    return normalize_label(value).title()


def _fold_location(slug: str, value: Any) -> Any:
    if slug not in _LOCATION_SLUGS:
        return value
    if isinstance(value, list):
        return [_canonical_location(item) for item in value]
    return _canonical_location(value)


def slots_to_where(slots: dict[str, Any]) -> dict[str, Any] | None:
    hard = slots.get("hard_filters") or {}
    conditions: list[dict[str, Any]] = []

    def add_filter(target: list[dict[str, Any]], slug: str, value: Any) -> None:
        spec = spec_for_slug(slug)
        if spec is None or value is None:
            return
        if spec.value_type == "multi_enum":
            for option in _list_value(value):
                if option in spec.values.values():
                    target.append({f"{slug}__{option}": True})
            return
        if spec.value_type == "enum":
            options = [option for option in _list_value(value) if option in spec.values.values()]
            if not options:
                return
            target.append({slug: {"$in": options}} if isinstance(value, list) else {slug: options[0]})
            return
        value = _fold_location(slug, value)
        if isinstance(value, list):
            target.append({slug: {"$in": value}})
            return
        target.append({slug: value})

    for slug, value in (hard.get("filters") or {}).items():
        add_filter(conditions, slug, value)
    for any_of in hard.get("any_of") or []:
        if not isinstance(any_of, dict):
            continue
        alternatives: list[dict[str, Any]] = []
        for slug, value in any_of.items():
            add_filter(alternatives, slug, value)
        if len(alternatives) == 1:
            conditions.append(alternatives[0])
        elif alternatives:
            conditions.append({"$or": alternatives})
    if hard.get("max_price_amount") is not None:
        conditions.append({"price_amount": {"$lte": hard["max_price_amount"]}})
    if hard.get("min_price_amount") is not None:
        conditions.append({"price_amount": {"$gte": hard["min_price_amount"]}})
    if hard.get("max_gross_size_m2") is not None:
        conditions.append({"gross_size_m2": {"$lte": hard["max_gross_size_m2"]}})
    if hard.get("min_gross_size_m2") is not None:
        conditions.append({"gross_size_m2": {"$gte": hard["min_gross_size_m2"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# Bu bir KİRALIK arama uygulaması. Dataset'te satılık ilanlar da bulunabilir;
# satılık fiyatı (toplam) ile kira (aylık) aynı eksende olduğundan satılık ilanlar
# kira sonuçlarına sızmamalı. trade_type registry'de yalnız 'kiralik' enum'u içerir.
_RENTAL_SCOPE = {"trade_type": "kiralik"}


def _scope_to_rental(where: dict[str, Any] | None) -> dict[str, Any]:
    if where is None:
        return dict(_RENTAL_SCOPE)
    if "$and" in where:
        return {"$and": [*where["$and"], dict(_RENTAL_SCOPE)]}
    return {"$and": [where, dict(_RENTAL_SCOPE)]}


def matched_filter_labels(slots: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Sorgunun istediği VE ilanın metadata'sında karşılanan filtreleri Türkçe çip olarak döndür.

    Deterministik match reason (Rule 5: kod cevaplıyor). where-clause her sonucu zaten
    elese de bu fonksiyon metadata'yı gerçekten kontrol eder; böylece çip asla
    karşılanmamış bir filtreyi 'neden eşleşti' diye iddia etmez.
    """
    hard = slots.get("hard_filters") or {}
    labels: list[str] = []

    def collect(slug: str, value: Any) -> None:
        spec = spec_for_slug(slug)
        if spec is None or value is None:
            return
        if spec.value_type == "multi_enum":
            for option in _list_value(value):
                if metadata.get(f"{slug}__{option}") is True:
                    label = label_for(slug, option)
                    if label:
                        labels.append(label)
            return
        if spec.value_type == "bool":
            if value is True and metadata.get(slug) is True:
                label = label_for(slug, True)
                if label:
                    labels.append(label)
            return
        value = _fold_location(slug, value)
        actual = metadata.get(slug)
        if actual is None:
            return
        if isinstance(value, list):
            if actual in value:
                label = label_for(slug, actual)
                if label:
                    labels.append(label)
        elif actual == value:
            label = label_for(slug, value)
            if label:
                labels.append(label)

    for slug, value in (hard.get("filters") or {}).items():
        collect(slug, value)
    for group in hard.get("any_of") or []:
        for slug, value in group.items():
            collect(slug, value)
    return labels


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_device())


def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=_device())


def _listify(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


class Retriever:
    def __init__(
        self,
        *,
        collection=None,
        embedder=None,
        reranker=None,
        slot_extractor: Callable[[str], dict[str, Any]] = extract_query_slots,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        if collection is None:
            import chromadb

            collection = chromadb.PersistentClient(path=str(persist_dir)).get_collection(collection_name)
        self.collection = collection
        self.embedder = embedder or _load_embedder(embedding_model)
        self.reranker = reranker or _load_reranker(reranker_model)
        self.slot_extractor = slot_extractor

    def retrieve(self, query: str, *, top_n: int = 50, top_k: int = 10) -> list[dict[str, Any]]:
        slots = self.slot_extractor(query)
        where = _scope_to_rental(slots_to_where(slots))
        query_embedding = _listify(self.embedder.encode([query], convert_to_numpy=True))
        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_n,
            where=where,
            include=["documents", "metadatas"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        if not ids:
            return []

        scores = self.reranker.predict([(query, document) for document in documents])
        ranked = sorted(
            zip(ids, documents, metadatas, scores),
            key=lambda item: float(item[3]),
            reverse=True,
        )[:top_k]
        return [
            {
                "id": listing_id,
                "score": float(score),
                "title": metadata.get("title"),
                "price_amount": metadata.get("price_amount"),
                "price_currency": metadata.get("price_currency"),
                "price_tl": metadata.get("price_amount") if metadata.get("price_currency") == "TL" else None,
                "filters": {spec.slug: metadata[spec.slug] for spec in EMLAKJET_FILTERS if spec.slug in metadata},
                "matched_filters": matched_filter_labels(slots, metadata),
                "enriched_doc": document,
            }
            for listing_id, document, metadata, score in ranked
        ]
