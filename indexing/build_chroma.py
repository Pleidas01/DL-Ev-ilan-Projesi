from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from indexing.composer import embedding_text, to_metadata


DEFAULT_INPUT = Path("data/processed/labeled.jsonl")
DEFAULT_PERSIST_DIR = Path("data/processed/chroma")
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION = "listings"
DEFAULT_BATCH_SIZE = 8


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_device())


def build_index(
    *,
    input_path: Path,
    persist_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    import chromadb

    records = _load_jsonl(input_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    collection = chromadb.PersistentClient(path=str(persist_dir)).get_or_create_collection(collection_name)
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [record for record in records if str(record.get("id")) not in existing_ids]
    if not pending:
        return 0

    embedder = _load_embedder(model_name)
    written = 0
    for batch in _batches(pending, batch_size):
        ids = [str(record["id"]) for record in batch]
        documents = [embedding_text(record) for record in batch]
        embeddings = embedder.encode(documents, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[to_metadata(record) for record in batch],
        )
        written += len(batch)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent Chroma index from M3 labeled listings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    written = build_index(
        input_path=args.input,
        persist_dir=args.persist_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(f"Indexed {written} new listings")


if __name__ == "__main__":
    main()
