from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llm.clients import candidate_by_id, complete_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "llm" / "selected.json"
EMPTY_RESULTS_ANSWER = "Aramanıza uygun ilan bulunamadı."
RAG_SYSTEM_PROMPT = """Sen Türkçe emlak ilanı önerileri hazırlayan bir asistansın.
Yalnızca verilen ilanları kullan. Bilgi uydurma.
Kısa ve doğrudan cevap ver.
Önerdiğin her ilanı şu formatta belirt:
[ilan:<id>] <title> - <price_tl> TL: <neden uygun>
Sadece geçerli JSON döndür:
{"answer": "<Türkçe cevap>"}"""

LLMFunction = Callable[[Any, str, str], str]


def _selected_candidate(selected_path: Path) -> Any:
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    return candidate_by_id(selected["text_model"])


def _build_user_prompt(query: str, results: list[dict[str, Any]]) -> str:
    listings = [
        {
            "id": result.get("id"),
            "title": result.get("title"),
            "price_tl": result.get("price_tl"),
            "filters": result.get("filters") or {},
            "matched_filters": result.get("matched_filters") or [],
            "enriched_doc": result.get("enriched_doc") or "",
        }
        for result in results
    ]
    return (
        f"Kullanıcı sorusu:\n{query}\n\n"
        "Retrieved ilanlar:\n"
        f"{json.dumps(listings, ensure_ascii=False, indent=2)}"
    )


def compose_answer(
    query: str,
    results: list[dict[str, Any]],
    *,
    llm_fn: LLMFunction = complete_json,
    candidate: Any | None = None,
    selected_path: Path = DEFAULT_SELECTED_PATH,
) -> str:
    if not results:
        return EMPTY_RESULTS_ANSWER

    selected_candidate = candidate if candidate is not None else _selected_candidate(selected_path)
    raw_response = llm_fn(selected_candidate, RAG_SYSTEM_PROMPT, _build_user_prompt(query, results))
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("RAG response LLM returned invalid JSON") from exc
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("RAG response LLM must return a non-empty answer")
    return answer.strip()
