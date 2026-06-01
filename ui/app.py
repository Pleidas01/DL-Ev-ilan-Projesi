from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from schema.emlakjet_filters import label_for, spec_for_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"
DEMO_QUERIES = (
    "Kadıköy'de 30 bin TL altı balkonlu 2+1 daire",
    "Metroya yakın eşyalı 1+1 kiralık",
    "Site içinde otoparklı aileye uygun 3+1",
)
# (canonical slug, kart etiketi). has_parking canonical DEĞİL; has_closed_parking kullanılır.
CARD_FACTS = (
    ("district", "İlçe"),
    ("room_count", "Oda"),
    ("gross_size_m2", "Brüt m²"),
    ("heating_type", "Isıtma"),
    ("is_furnished", "Eşyalı"),
    ("has_balcony", "Balkon"),
    ("has_closed_parking", "Otopark"),
)


def _card_value(slug: str, value: Any) -> str | None:
    spec = spec_for_slug(slug)
    if spec is not None and spec.value_type == "bool":
        return "Var" if value is True else None
    if spec is not None and spec.value_type in ("enum", "multi_enum"):
        return label_for(slug, value)
    return str(value)


def card_fact_lines(result: dict[str, Any]) -> list[str]:
    """Retriever'ın `filters` çıktısını karta yazılacak okunabilir satırlara çevir.

    Yalnız filters'ta dolu canonical slug'lar gösterilir; enum slug'lar Türkçe label'a,
    bool True 'Var'a çevrilir. Eski `facts` anahtarı okunmaz (Gen3 göçü).
    """
    filters = result.get("filters") or {}
    lines: list[str] = []
    for slug, label in CARD_FACTS:
        if slug not in filters:
            continue
        value = _card_value(slug, filters[slug])
        if value is not None:
            lines.append(f"{label}: {value}")
    return lines


def _create_retriever():
    from retrieval.retriever import Retriever

    return Retriever()


def _compose_answer(query: str, results: list[dict[str, Any]]) -> str:
    from chat.rag_response import compose_answer

    return compose_answer(query, results)


def _run_search(
    query: str,
    retriever_factory: Callable[[], Any],
    compose_fn: Callable[[str, list[dict[str, Any]]], str] = _compose_answer,
) -> tuple[Any | None, list[dict[str, Any]], str | None, str | None]:
    try:
        retriever = retriever_factory()
        results = retriever.retrieve(query)
        return retriever, results, compose_fn(query, results), None
    except Exception as exc:
        return None, [], None, f"Arama başlatılamadı: {exc}"


def _first_image(listing_id: Any) -> Path | None:
    listing_dir = IMAGE_ROOT / str(listing_id)
    if not listing_dir.is_dir():
        return None
    return next((path for path in sorted(listing_dir.iterdir()) if path.is_file()), None)


def _format_price(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", ".") + " TL"
    return f"{value} TL" if value is not None else "Fiyat belirtilmemiş"


def _render_listing_card(st, result: dict[str, Any]) -> None:
    with st.container(border=True):
        image_column, detail_column = st.columns([1, 3])
        image_path = _first_image(result.get("id"))
        if image_path:
            image_column.image(str(image_path), width="stretch")
        with detail_column:
            st.subheader(result.get("title") or f"İlan {result.get('id')}")
            st.write(_format_price(result.get("price_tl")))
            st.caption(f"İlan ID: {result.get('id')}")
            matched = result.get("matched_filters") or []
            if matched:
                st.markdown("Neden uygun: " + " ".join(f"`{label}`" for label in matched))
            fact_lines = card_fact_lines(result)
            if fact_lines:
                st.markdown(" · ".join(f"**{line}**" for line in fact_lines))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Türkçe Emlak RAG", layout="wide")
    st.title("Türkçe Emlak RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []

    st.write("Demo sorgular")
    demo_query = None
    for column, query in zip(st.columns(len(DEMO_QUERIES)), DEMO_QUERIES):
        if column.button(query, width="stretch"):
            demo_query = query

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = demo_query or st.chat_input("Nasıl bir ev arıyorsunuz?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("İlanlar aranıyor..."):
            retriever, results, answer, error = _run_search(
                query,
                lambda: st.session_state.get("retriever") or _create_retriever(),
            )
        if retriever is not None:
            st.session_state.retriever = retriever
        st.session_state.last_results = results
        if error:
            st.error(error)
        else:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

    if st.session_state.last_results:
        st.subheader("İlanlar")
        for result in st.session_state.last_results:
            _render_listing_card(st, result)


if __name__ == "__main__":
    main()
