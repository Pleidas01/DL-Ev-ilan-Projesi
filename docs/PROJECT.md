# PROJECT.md — Domain & Architecture (stable reference)

Bu dosya proje vizyonunu, mimariyi ve "neden böyle" sorularını yanıtlar. Milestone'lar değiştikçe değişmez. Güncel ilerleme için `STATUS.md`, sıradaki somut iş için `HANDOFF.md` oku.

---

## 1. Ödev bağlamı

- **Ders:** END439E — Neural Networks for Industrial Systems (İTÜ İşletme Fak., Prof. Dr. Muammer Altan Çakır)
- **Takım:** Damla Hatice Selçuk, Elif Kamalak, Mertcan El
- **İki teslim var:**
  1. Ana proje: Türkçe gayrimenkul ilanı arama (multimodal RAG)
  2. Time series ek ödevi: "design time series neural network models for your project. In case of lack of data, create synthetic data generation."
- **Yasak modeller:** GPT-5.5, Claude Opus 4.7 (TR erişim + maliyet). Hoca hem **feasibility** (yüzbinlerce ilana scale eder mi) hem **kendi eğitilmiş NN** görmek istiyor.

## 2. Kullanıcı senaryosu

Kullanıcı serbest Türkçe text yazar:
> "Kadıköy'de 30 bin altı, geniş salonlu, asansörlü, denize yakın 2+1 öğrenciye uygun daire"

Sistem ilgili ilanları döndürür: kapak fotoğrafı + başlık + fiyat + neden eşleştiği (match reason).

Sorgu **hem hard filter** (oda, ilçe, max fiyat) **hem soft özellik** (image: `salon_size=genis`, `view=deniz`; text: `elevator=true`, `near_transit=true`) içerebilir.

## 3. Neden CLIP'i terk ettik

Önceki HW6 yaklaşımı M-CLIP fine-tuning idi (`END439E_HW6.ipynb`). İki temel problem:

1. **Çoklu foto problemi:** Bir ilanın 5-15 fotoğrafı var (salon, mutfak, balkon, vs). CLIP tek bir embedding'e nasıl sığar?
2. **Modality gap:** Türkçe ilan metni ↔ ev fotoğrafı semantik mesafesi büyük; R@10 = 0.037 ile başarısız oldu.

**Ders asistanının önerisi:** "LLM/VLM ile fotoğrafları JSON özelliklere sınıflandır, image-text bulmayı text-text bulmaya çevir."

## 4. Yeni mimari

```
SCRAPING (Playwright)
  → data/raw/listings.jsonl + data/raw/images/<id>/*.jpg
CLEANING (scraper/cleaner.py)
  → data/processed/dataset.jsonl

LABELING PIPELINE (Milestone 3)
  Per listing:
    - VLM (Gemma4 / Kimi / Gemini) → her foto için image_json
    - LLM (DeepSeek / Gemma4) → description'dan text_json
    - merge → enriched_doc (title + attrs + image_tags + text_tags)
  → data/processed/labeled.jsonl

INDEXING (Milestone 3-4)
  - BGE-M3 multilingual embedding (LoRA fine-tune ile domain adapt)
  - ChromaDB persistent collection
  → data/chroma/

QUERY TIME (Milestone 4-5)
  user query
    → LLM slot extraction (hard_filters + soft_features JSON)
    → Chroma vector search (with metadata filter)
    → Cross-encoder reranker
    → top-K listings + match reason
  → Streamlit chat UI

TIME SERIES (parallel ödev)
  - Scraped fiyat istatistikleri ile sentetik kira zaman serisi
  - LSTM / GRU forecaster
  → notebooks/timeseries_report.ipynb
```

## 5. Model seçimleri ve neden

### 5a. LLM/VLM aday seti (feasibility-first)

Tüm adaylar `llm/clients.py:CANDIDATES` içinde. Pricing $/M token.

| id | rol | input $ | output $ | TR erişim | not |
|---|---|---|---|---|---|
| `deepseek_v4_flash` | text only | 0.14 | 0.28 | OK | en ucuz text aday |
| `kimi_k2_6` | text + vision | 0.95 | 4.00 | OK | Moonshot, vision adayı |
| `gemini_3_5_flash` | text + vision | 1.50 | 9.00 | OK | Google AI Studio free tier |
| `glm_4_6` | text only | 0.30 | 0.50 | OpenRouter üzerinden | opsiyonel |
| `gemma_4_local` | text + vision | 0.0 | 0.0 | local Ollama | "scale eder" argümanı |

**Karar kuralı:** Gold benchmark accuracy ≥ %75 olan en ucuz modeli seç. Gemma4 yeterince iyiyse production'da Gemma4 (sıfır maliyet); değilse vision için Gemini/Kimi.

### 5b. Embedding modeli

- **BGE-M3** (`BAAI/bge-m3`) — multilingual, 1024-dim, Türkçe destekli.
- Domain adaptation: **LoRA fine-tune** (kendi-eğitilmiş NN #1 — hoca isteği).
- Alternatif (yedek): `multilingual-e5-large-instruct`.

### 5c. Reranker

- `BAAI/bge-reranker-v2-m3` cross-encoder.

### 5d. Kendi eğitilmiş NN'ler (akademik gereksinim)

1. **BGE-M3 LoRA** (text embedding, domain adapt)
2. **ResNet-18 / ViT-small classifier** (image özellikleri için — VLM ile karşılaştırma noktası)
3. **LSTM / GRU forecaster** (time series ödevi)
4. (Opsiyonel) Slot extraction için küçük distilled model

Bu 4 model **"hoca, sadece API çağırmadık, kendi de eğittik"** argümanını taşır.

## 6. Veri kaynağı

- **Şu an:** Emlakjet, sadece İstanbul kiralık, **1239 ilan** (`data/processed/dataset.jsonl`, 2026-05-26 re-scrape sonrası).
- Ham scrape 1483 benzersiz ilan; cleaner duplicate URL/görsel + image-yok ilanları eledi → 1239 net (retention %42.3 ham 2928'e göre, ama scrape sırasında resume duplicate satırları biriktiği için anlamlı retention 1239/1483 ≈ %83.5).
- **Plan:** Hoca demo'su için 1239 yeterli. M3 sonrası retrieval metric'lerine bakılır, gerekirse ek 1500-2000 ilan incremental scrape edilir.
- Scraper (`scraper/playwright_scraper.py`) yeni `INFO_FIELD_MAP` ile 21 facts alanının structured 13'ünü otomatik çıkarıyor.

## 7. Şema tanımları (single source of truth)

### 7a. Katman A — `facts_gold` (21 alan)

Priority chain: structured scraper alanı varsa otomatik dolar; hybrid alanlarda önce features/title, sonra description LLM; visual destekli alanlarda gerekirse VLM fallback kullanılır.

**STRUCTURED (15 — scraper "İlan Bilgileri" tablosundan kesin):**
1. `city` — district split
2. `district` — district split
3. `neighborhood` — district split
4. `price_tl` — price → int
5. `room_count` — `attributes.roomCount`
6. `gross_size_m2` — `attributes.grossSize` → int
7. `net_size_m2` — `attributes.netSize` → int
8. `building_age` — `attributes.buildingAge`, string
9. `floor` — `attributes.floor`
10. `total_floors` — `attributes.totalFloors` → int
11. `deposit_tl` — İlan Bilgileri "Depozito" → int|null
12. `in_gated_complex` — İlan Bilgileri "Site İçerisinde" Evet/Hayır → bool|null
13. `title_deed_status` — İlan Bilgileri "Tapu Durumu" → string|null
14. `heating_type` — `attributes.heating` normalize; enum `kombi|dogalgaz|merkezi|klima|yerden_isitma|null`
15. `is_furnished` — İlan Bilgileri "Eşya Durumu" Eşyalı/Eşyasız → bool|null (mevcut raw'da attribute yok; cleaner fallback ile title/features keyword)

**HYBRID (4 — features/title + description LLM + image VLM fallback):**
16. `has_balcony` — features → desc LLM → image VLM; bool|null
17. `has_elevator` — features "Asansör" → title regex → desc LLM; bool|null
18. `has_parking` — features → desc LLM; bool|null
19. `has_aircon` — features "Klima" → title/desc LLM → image VLM; bool|null

**DESCRIPTION-ONLY (2 — LLM extraction):**
20. `near_metro` — desc LLM, bool|null, `<15dk yürüme`
21. `near_metrobus` — desc LLM, bool|null

> Not: `kitchen_type` kaldırıldı; mutfak bilgisi tek alanda — `visual_gold.mutfak_tipi` (sadece `amerikan_acik | kapali_ayri`).

### 7b. Katman B — `visual_gold` (7 alan)

Primary VLM, fallback/cross-check `property_features`. single-select: `mutfak_tipi`, `zemin_tipi`. Diğerleri multi-select (JSON array).

```json
{
  "balkon_ozellikleri": ["cam_balkon", "acik_balkon", "fransiz_balkon", "cikma_balkon", "teras"],
  "manzara": ["deniz", "bogaz", "orman_yesil", "park", "sehir_panorama", "dag", "ic_avlu", "komsu_duvari"],
  "mutfak_tipi": "amerikan_acik | kapali_ayri",
  "banyo_ozellikleri": ["dusakabin", "kuvet", "jakuzi", "banyoda_pencere", "birden_fazla_banyo"],
  "zemin_tipi": "parke | laminat | seramik | granit | mermer | hali | karma",
  "salon_ozellikleri": ["somine", "nis", "acik_plan_genis", "ayri_yemek_alani"],
  "imkanlar": ["havuz", "yesil_alan_peyzaj", "guvenlik_kabini", "kapali_otopark", "acik_otopark", "cocuk_parki", "spor_alani"]
}
```

### 7c. Gold scoring kuralları

- Fotoğrafta görünmeyen visual alan `null` kalır. `score_against_gold` gold tarafı `null` olan satırı SKIP eder.
- Multi-select alanlar için Jaccard score kullanılır: `|intersect| / |union|`. Hem gold hem pred boşsa skor `1`.
- Confidence M3'e ertelendi; shootout sade accuracy ölçer.

### 7d. Slot extraction çıktısı

```json
{
  "hard_filters": {"rooms": ["2+1"], "districts": ["Kadıköy"], "max_price_tl": 30000, "min_size_m2": null},
  "soft_features": {
    "facts_gold": { "...": "21 facts_gold alan adları" },
    "visual_gold": { "...": "7 visual_gold alan adları" }
  },
  "free_form_tr": "<orijinal sorgu>"
}
```

Resmi tanımlar `llm/gold_benchmark.py:FACTS_GOLD_FIELDS` ve `VISUAL_GOLD_FIELDS` ile `llm/shootout.py:FEW_SHOT_SLOT_EXAMPLES` içindedir. Yeni alan eklenirse bu dosyalar birlikte güncellenir.

## 8. Değerlendirme metrikleri

| Aşama | Metrik | Hedef | Gold kaynağı |
|---|---|---|---|
| Slot extraction | JSON-valid + slot accuracy | quality_score ≥ 0.85 | `llm/shootout.py:BENCHMARK_QUERIES` |
| Description extraction | per-field accuracy (7 hybrid+desc alanı) | ≥ 0.75 | `labeling/gold_listings_manual_todo.jsonl:facts_gold` |
| Image labeling | per-field Jaccard / accuracy (7 visual alan) | ≥ 0.70 | `labeling/gold_listings_manual_todo.jsonl:visual_gold` |
| Retrieval | R@5, R@10, MRR | R@10 ≥ 0.60 | `evaluation/gold_queries_manual_todo.jsonl` |
| Time series | MAE / sMAPE | rapor (baseline'a göre) | sentetik gold |

## 9. Proje dizini (mevcut)

```
DL-Ev-ilan-Projesi/
├── scraper/           # playwright_scraper.py, cleaner.py, image_downloader.py
├── llm/               # clients.py, shootout.py, shootout_vision.py,
│                      # shootout_description.py, gold_benchmark.py
├── labeling/          # gold_helper.py, gold_listings_manual_todo.jsonl
├── evaluation/        # gold_helper.py, gold_queries_manual_todo.jsonl
├── retrieval/         # (boş) — indexer.py + retriever.py burada olacak
├── model/             # (boş) — fine-tune scriptleri
├── finetune/          # (boş) — BGE-M3 LoRA + ResNet classifier
├── indexing/          # (boş) — composer + chroma builder
├── chat/              # (boş) — RAG response composition
├── timeseries/        # (boş) — LSTM/GRU
├── ui/                # (boş) — Streamlit app
├── tests/             # test_llm_shootout, test_gold_helpers, test_scraper_description_limit
├── data/
│   ├── raw/           # Playwright çıktısı (listings.jsonl 1483 unique)
│   ├── processed/dataset.jsonl  (1239 satır, yeni 21 facts şeması)
│   ├── chroma/        # vector store (henüz yok)
│   └── eval/manual_queries.jsonl  (eski auto-BM25 çıktısı, kullanma)
├── docs/              # llm_setup_tr.md, manual_gold_todo_tr.md, + bu dosya
├── archive/           # eski CLIP işleri
└── END439E_HW6.ipynb  # eski CLIP fine-tune (arşivlik)
```

## 10. Milestone haritası

| # | Ad | Çıktı | Durum |
|---|---|---|---|
| 0 | Scaffold + scraper + cleaner (yeni şema) | clean repo, 21 facts + visual_qualities | **DONE** (2026-05-26, 1239 ilan) |
| 1 | LLM slot shootout | `selected.json` text aday | **DONE** (kimi_k2_6 winner) |
| 2 | Manual gold sets | facts_gold + visual_gold + query gold | **AKTİF** (template hazır, kullanıcı dolduruyor) |
| 1.5 | Vision + Description shootout | `vision_model`+`text_model` kararı | **BLOKE** (M2 bitince) |
| 3 | Labeling pipeline | `labeled.jsonl` | PENDING |
| 4 | Indexing + retrieval | chroma + retriever | PENDING |
| 5 | RAG chat + UI | Streamlit demo | PENDING |
| 6 | Fine-tune NN'ler | BGE-M3 LoRA + ResNet | PENDING |
| 7 | Time series | LSTM/GRU + notebook | PENDING |
| 8 | Final report | rapor notebook + sunum | PENDING |

Detaylı durum için `STATUS.md`.

## 11. Kullanıcı kuralları (rules) — özet

`.cursor/rules` kapsamı dışında, kullanıcının bu projedeki çalışma kuralları:

- **Bias: caution over speed** non-trivial işlerde.
- **Soru sor, tahmin etme** belirsizlik varken.
- **Surgical changes** — sadece istenen yere dokun, yan refactor yok.
- **Sessizce başarısız olma** — "tests pass" derken hiçbir test skip edilmemiş olmalı.
- **Token budget:** task başına ~4K, session ~30K. Aşıyorsa özetle ve yeniden başla.
- **Kod konvansiyonuna uy**, kendi tarzını dayatma.
- **Komut çalıştırmadan önce oku.** Caller'ı, util'i, schema'yı bil.

Tam metin için kullanıcının `user_rules` bölümüne bak.
