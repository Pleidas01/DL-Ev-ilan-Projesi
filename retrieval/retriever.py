from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llm.clients import candidate_by_id, complete_json
from llm.shootout import SLOT_SYSTEM_PROMPT, build_slot_prompt, flatten_slots
from schema.emlakjet_filters import EMLAKJET_FILTERS, spec_for_slug


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
        if isinstance(value, list):
            target.append({slug: {"$in": value}})
            return
        target.append({slug: value})

    for slug, value in (hard.get("filters") or {}).items():
        add_filter(conditions, slug, value)
    for any_of in hard.get("any_of") or []:
        alternatives: list[dict[str, Any]] = []
        for slug, value in any_of.items():
            add_filter(alternatives, slug, value)
        if alternatives:
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


def _device() -> str:
    import torch

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
        where = slots_to_where(slots)
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
                "enriched_doc": document,
            }
            for listing_id, document, metadata, score in ranked
        ]
