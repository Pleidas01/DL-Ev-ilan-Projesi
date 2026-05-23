"""
mCLIP Gayrimenkul Arama — Streamlit UI
=======================================
Kullanım:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.retriever import Retriever

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emlak Arama · mCLIP",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Türkçe Gayrimenkul Semantic Arama")
st.caption("M-CLIP XLM-Roberta-Large-Vit-B-32 · ChromaDB · MMR")

# ── Retriever (önbellekli) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Model yükleniyor…")
def get_retriever(checkpoint: str | None = None) -> Retriever:
    return Retriever(checkpoint=checkpoint)


# ── Yan panel — ayarlar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Arama Ayarları")

    k = st.slider("Sonuç sayısı (K)", min_value=1, max_value=20, value=5)
    lam = st.slider(
        "MMR λ  (0 = çeşitlilik, 1 = alaka)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    )
    alpha = st.slider(
        "Metin ağırlığı α  (çok kipli modda)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="α × metin + (1-α) × görsel",
    )
    checkpoint = st.text_input(
        "Fine-tuned checkpoint (opsiyonel)",
        placeholder="model/checkpoints/best.pt",
    ) or None

    st.divider()
    st.caption("Arama modları:\n- Sadece metin\n- Sadece görsel\n- Metin + görsel (çok kipli)")

retriever = get_retriever(checkpoint)

# ── Ana içerik — sorgular ───────────────────────────────────────────────────────
col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Sorgu")
    text_query = st.text_area(
        "Metin sorgusu",
        placeholder="Örn: deniz manzaralı 2+1 daire, ferah salon, merkezi konum…",
        height=120,
    )
    image_query = st.file_uploader(
        "Görsel sorgusu (opsiyonel)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if image_query:
        st.image(image_query, caption="Yüklenen sorgu görseli", use_container_width=True)

    search_btn = st.button("🔍 Ara", type="primary", use_container_width=True)

# ── Sonuçlar ────────────────────────────────────────────────────────────────────
with col_results:
    st.subheader("Sonuçlar")

    if search_btn:
        if not text_query and not image_query:
            st.warning("Lütfen bir metin veya görsel sorgusu girin.")
        else:
            with st.spinner("Aranıyor…"):
                try:
                    if text_query and image_query:
                        pil_img = Image.open(image_query).convert("RGB")
                        results = retriever.query_multimodal(
                            text_query, pil_img, k=k, lam=lam, alpha=alpha
                        )
                        mode_label = "Çok Kipli"
                    elif text_query:
                        results = retriever.query_text(text_query, k=k, lam=lam)
                        mode_label = "Metin"
                    else:
                        pil_img = Image.open(image_query).convert("RGB")
                        results = retriever.query_image(pil_img, k=k, lam=lam)
                        mode_label = "Görsel"

                    st.success(f"{mode_label} araması · {len(results)} sonuç")

                    for i, res in enumerate(results, 1):
                        m = res["metadata"]
                        with st.container():
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                img_path = m.get("image_path", "")
                                if img_path and Path(img_path).exists():
                                    st.image(img_path, use_container_width=True)
                                else:
                                    st.markdown("🖼️ *Görsel yok*")
                            with c2:
                                score_pct = int(res["score"] * 100)
                                st.markdown(f"**#{i} — {m['title']}**")
                                st.markdown(
                                    f"💰 `{m['price']}`  &nbsp;&nbsp;  "
                                    f"📍 {m['district']}  &nbsp;&nbsp;  "
                                    f"✅ Benzerlik: `{score_pct}%`"
                                )
                                st.markdown(
                                    f"[İlana git →]({m['url']})",
                                    unsafe_allow_html=False,
                                )
                                with st.expander("Metin önizleme"):
                                    st.caption(m.get("text", "—"))
                            st.divider()

                except Exception as e:
                    st.error(f"Arama hatası: {e}")
    else:
        st.info("Sol tarafta bir metin veya görsel girin, ardından 'Ara' butonuna tıklayın.")
