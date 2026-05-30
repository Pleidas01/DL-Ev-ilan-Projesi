from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llm.clients import candidate_by_id, complete_json
from llm.gold_benchmark import FACTS_GOLD_FIELDS
from llm.shootout import SLOT_SYSTEM_PROMPT, build_slot_prompt, flatten_slots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "llm" / "selected.json"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_COLLECTION = "listings"
BOOLEAN_FILTER_FIELDS = (
    "in_gated_complex",
    "is_furnished",
    "has_balcony",
    "has_elevator",
    "has_parking",
    "has_aircon",
    "near_metro",
    "near_metrobus",
)


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
    flat = flatten_slots(slots)
    conditions: list[dict[str, Any]] = []

    rooms = _list_value(hard.get("rooms"))
    if rooms:
        conditions.append({"room_count": {"$in": rooms}})
    districts = _list_value(hard.get("districts"))
    if districts:
        conditions.append({"district": {"$in": districts}})
    if hard.get("max_price_tl") is not None:
        conditions.append({"price_tl": {"$lte": hard["max_price_tl"]}})
    if hard.get("min_size_m2") is not None:
        conditions.append({"gross_size_m2": {"$gte": hard["min_size_m2"]}})
    for field_name in BOOLEAN_FILTER_FIELDS:
        value = flat.get(field_name)
        if isinstance(value, bool):
            conditions.append({field_name: value})

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
                "price_tl": metadata.get("price_tl"),
                "facts": {field: metadata[field] for field in FACTS_GOLD_FIELDS if field in metadata},
                "enriched_doc": document,
            }
            for listing_id, document, metadata, score in ranked
        ]
