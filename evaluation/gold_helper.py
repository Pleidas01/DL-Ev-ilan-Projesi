from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from llm.gold_benchmark import DEFAULT_DATASET_PATH, load_jsonl

ISTANBUL_DISTRICTS = (
    "kadikoy",
    "kadıköy",
    "besiktas",
    "beşiktaş",
    "sisli",
    "şişli",
    "uskudar",
    "üsküdar",
    "pendik",
    "kartal",
    "maltepe",
    "atasehir",
    "ataşehir",
    "basaksehir",
    "başakşehir",
    "avcilar",
    "avcılar",
    "bostanci",
    "bostancı",
    "bakirkoy",
    "bakırköy",
    "sariyer",
    "sarıyer",
    "tuzla",
    "umraniye",
    "ümraniye",
    "beylikduzu",
    "beylikdüzü",
    "bahcesehir",
    "bahçeşehir",
    "kurtkoy",
    "kurtköy",
    "moda",
    "suadiye",
    "tarabya",
    "ayazaga",
    "ayazağa",
)


def tok(text: str) -> list[str]:
    normalized = text.lower()
    normalized = normalized.translate(str.maketrans("çğıöşü", "cgiosu"))
    return re.findall(r"[a-z0-9+]+", normalized)


def parse_price_tl(text: str) -> int | None:
    normalized = text.lower().replace(".", "").replace(",", "")
    match = re.search(r"(\d+)\s*bin", normalized)
    if match:
        return int(match.group(1)) * 1000
    match = re.search(r"(\d{4,6})\s*tl", normalized)
    if match:
        return int(match.group(1))
    return None


def parse_rooms(text: str) -> str | None:
    match = re.search(r"(\d\+\d)", text.lower())
    return match.group(1) if match else None


def parse_districts(text: str) -> list[str]:
    normalized = _normalize_blob(text)
    found: list[str] = []
    for district in ISTANBUL_DISTRICTS:
        district_norm = _normalize_blob(district)
        if district_norm in normalized and district_norm not in found:
            found.append(district_norm)
    return found


def extract_hard_filters(query: str) -> dict[str, Any]:
    return {
        "rooms": parse_rooms(query),
        "max_price_tl": parse_price_tl(query),
        "districts": parse_districts(query),
    }


def listing_price_tl(price: str) -> int | None:
    digits = re.sub(r"[^\d]", "", price or "")
    return int(digits) if digits else None


def passes_hard_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters.get("rooms"):
        room_count = (record.get("attributes") or {}).get("roomCount", "")
        haystack = f"{record.get('title', '')} {record.get('text', '')} {room_count}".lower()
        if filters["rooms"] not in haystack:
            return False

    if filters.get("max_price_tl") is not None:
        price = listing_price_tl(str(record.get("price", "")))
        if price is None or price > filters["max_price_tl"]:
            return False

    if filters.get("districts"):
        district_blob = _normalize_blob(record.get("district", "") + " " + record.get("text", ""))
        if not any(district in district_blob for district in filters["districts"]):
            return False

    return True


def _normalize_blob(text: str) -> str:
    return text.lower().translate(str.maketrans("çğıöşü", "cgiosu"))


def search_candidates(
    query: str,
    records: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    filters = extract_hard_filters(query)
    filtered = [record for record in records if passes_hard_filters(record, filters)]
    pool = filtered if filtered else records

    corpus_tokens = [tok((record.get("text", "") + " " + record.get("district", ""))) for record in pool]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tok(query))
    ranked_indices = sorted(range(len(scores)), key=lambda index: -scores[index])[:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        record = pool[index]
        results.append({
            "rank": rank,
            "id": record["id"],
            "title": record.get("title", ""),
            "price": record.get("price", ""),
            "district": record.get("district", ""),
            "url": record.get("url", ""),
            "bm25_score": float(scores[index]),
        })
    return results


def format_results(query: str, results: list[dict[str, Any]], filters: dict[str, Any]) -> str:
    lines = [
        f'query: "{query}"',
        f"hard_filters: {json.dumps(filters, ensure_ascii=False)}",
        "",
        "top_candidates:",
    ]
    for row in results:
        lines.append(
            f"  [{row['rank']}] id={row['id']:<10} price={row['price']:<12} score={row['bm25_score']:.3f} | {row['title']}"
        )
        if row.get("url"):
            lines.append(f"        {row['url']}")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="BM25 + hard-filter retrieval helper for gold query labeling")
    parser.add_argument("--query", required=True, help="Turkish search query")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    records = load_jsonl(Path(args.dataset))
    filters = extract_hard_filters(args.query)
    results = search_candidates(args.query, records, top_k=args.top_k)
    print(format_results(args.query, results, filters))


if __name__ == "__main__":
    main()
