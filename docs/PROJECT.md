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

Sorgudaki Emlakjet paneliyle karşılığı olan açık istekler **hard filter** olur (oda, ilçe, max fiyat, asansör, denize yakınlık vb.). Registry dışındaki serbest nitelemeler embedding + reranker sıralamasına kalır. Kanıtlanmamış `null` değer explicit filtreyle eşleşmez.

## 3. Neden CLIP'i terk ettik

Önceki HW6 yaklaşımı M-CLIP fine-tuning idi (`END439E_HW6.ipynb`). İki temel problem:

1. **Çoklu foto problemi:** Bir ilanın 5-15 fotoğrafı var (salon, mutfak, balkon, vs). CLIP tek bir embedding'e nasıl sığar?
2. **Modality gap:** Türkçe ilan metni ↔ ev fotoğrafı semantik mesafesi büyük; R@10 = 0.037 ile başarısız oldu.

**Ders asistanının önerisi:** "LLM/VLM ile fotoğrafları JSON özelliklere sınıflandır, image-text bulmayı text-text bulmaya çevir."

## 4. Yeni mimari

```
SCRAPING (Playwright)
  → data/raw/listings.jsonl + data/images/<id>/*.jpg
CLEANING (scraper/cleaner.py)
  → data/processed/dataset.jsonl

LABELING PIPELINE (Milestone 3)
  Per listing:
    - scraper → İlan Bilgileri + seçili İlan Özellikleri fact'leri
    - DeepSeek V4 Pro → yalnız title + description; null alanları açık metin kanıtıyla true/false doldurur
    - Kimi K2.6 → yalnız fotoğraflar; kalan null boolean alanları yalnız açık görsel kanıt varsa true doldurur
    - monotonic merge → filter_values + filter_sources + enriched_doc
  → data/processed/labeled.jsonl

INDEXING (Milestone 3-4)
  - BGE-M3 multilingual embedding (LoRA fine-tune ile domain adapt)
  - ChromaDB persistent collection (canonical scalar metadata + multi-enum option flags)
  → data/processed/chroma/

QUERY TIME (Milestone 4-5)
  user query
    → LLM slot extraction (canonical hard_filters JSON)
    → Chroma metadata hard filter
    → Chroma vector search
    → Cross-encoder reranker
    → top-K listings + match reason
  → Streamlit chat UI

TIME SERIES (parallel ödev) — KAPSAM DIŞI (2026-05-30, sınıf arkadaşı yaptı)
  - (bu projede yapılmayacak) sentetik kira zaman serisi + LSTM/GRU forecaster
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

> **Güncel seçim (`llm/selected.json`, 2026-05-30):** `text_model = deepseek_v4_pro`, `vision_model = kimi_k2_6`. Tabloya sonradan eklenen iki aday: `deepseek_v4_pro` (%75 indirimli $0.435/$0.87 text-only, gold A/B kazananı) ve `qwen3_vl_local` (qwen3-vl:8b, Ollama — ücretsiz/yerel vision aday, gold benchmark'ı henüz koşulmadı). Detaylı gerekçe için STATUS.md M1 revize.

### 5b. Embedding modeli

- **BGE-M3** (`BAAI/bge-m3`) — multilingual, 1024-dim, Türkçe destekli.
- Domain adaptation: **LoRA fine-tune** (kendi-eğitilmiş NN #1 — hoca isteği).
- Alternatif (yedek): `multilingual-e5-large-instruct`.

### 5c. Reranker

- `BAAI/bge-reranker-v2-m3` cross-encoder.

### 5d. Kendi eğitilmiş NN'ler (akademik gereksinim)

1. **BGE-M3 LoRA** (text embedding, domain adapt)
2. **ResNet-18 / ViT-small classifier** (image özellikleri için — VLM ile karşılaştırma noktası)
3. ~~**LSTM / GRU forecaster** (time series ödevi)~~ — **KAPSAM DIŞI (2026-05-30):** time series ödevini sınıf arkadaşı yaptı, bu projede yapılmayacak.
4. (Opsiyonel) Slot extraction için küçük distilled model

> **NN gereksinimi uyarısı (2026-05-30):** Hocanın "kendi eğittiğiniz NN olsun" şartı önceden #3 (LSTM/GRU) ile karşılanıyor sayılıyordu. #3 dışarı çıktığı için bu proje kendi öz-eğitimli NN'ini **#1 (BGE-M3 LoRA) veya #2 (ResNet/ViT)** üzerinden vermeli (M6). Kullanıcı kararı bekleniyor — sessizce atlanmasın.

Bu 4 model **"hoca, sadece API çağırmadık, kendi de eğittik"** argümanını taşır.

## 6. Veri kaynağı

- **Şu an (2026-05-31):** Emlakjet, sadece İstanbul kiralık, yeni canonical-canvas şemasıyla **305 ham ilan** (`data/raw/listings.jsonl`; 150-slug `filter_values`). `4804/4804` indirilebilir görsel `data/images/` altına alındı. Cleaner `303` aktif ilan üretti (`data/processed/dataset.jsonl`; retention `%99.3`); 2 ilanda görsel URL'lerinin dosya adları başka ilan ID'si taşıdığı için cleaner bilinçli olarak eledi.
- Eski `1239` ilanlık dataset yalnız tarihsel arşivdir; yeni labeling/index build için kullanılmayacak.
- **Plan:** Hoca demo'su için ilk batch 305; M3 retrieval metric'lerine göre incremental büyütülebilir (kullanıcı "300 tane çekelim" dedi, ölçülen hız ~4s/ilan).
- Scraper (`scraper/playwright_scraper.py`) canonical registry lookup ile fact çıkarıyor. **Capture-all (2026-05-31):** `İlan Bilgileri`'nin TÜM satırları `infoTableAll`'a kayıpsız alınır (whitelist yok); `İlan Özellikleri`'nin 3 sekmesi de (İç/Dış/Konum) tıklanıp tüm maddeler toplanır. Önceki "seçili alanlar" davranışı kaldırıldı (örn. balkon gibi satırların sessizce düşmesini engeller). Dolum monotonik: scraper → DeepSeek (metin, null-only) → Kimi (görsel, null-only).

## 7. Şema tanımları (single source of truth)

### 7a. Canonical registry

Resmi şema kaynağı Emlakjet kiralık konut sol filtre panelidir: <https://www.emlakjet.com/kiralik-konut>. Kod tarafındaki tek merkezi tanım `schema/emlakjet_filters.py:EMLAKJET_FILTERS` listesidir. Registry şu anda `26` structured ve `124` checkbox alanı içerir.

Her `FilterSpec` stable slug, grup, değer tipi, izinli enum değerleri ve kanıt kaynaklarını taşır. Canonical kayıt biçimi:

```json
{
  "filter_values": {
    "price_amount": 30000,
    "price_currency": "TL",
    "room_count": "2+1",
    "has_elevator": true,
    "has_balcony": null
  },
  "filter_sources": {
    "price_amount": "scraper_info",
    "room_count": "scraper_info",
    "has_elevator": "scraper_property_feature"
  }
}
```

`price_tl`, eski top-level facts ve `visual_qualities.aggregated.imkanlar` geçici uyumluluk/output görünümü olarak kalabilir. Canonical arama yalnız registry typed alanlarıyla çalışır. `salon_ozellikleri` canonical ve aktif compatibility sözleşmesinden kaldırılmıştır.

### 7b. Kanıt zinciri

1. Scraper `İlan Bilgileri` explicit değerlerini ve seçili `İlan Özellikleri` checkbox'larını deterministik doldurur.
2. Explicit `Var/Evet -> true`, `Yok/Hayır -> false`, eksik structured değer `null` olur.
3. Checkbox fact'leri positive-only'dir: seçili `true`, eksik `null`.
4. DeepSeek yalnız title + description okur ve yalnız halen `null` kalan `description_llm` izinli alanları açık metin kanıtıyla `true` veya `false` doldurur.
5. Kimi yalnız fotoğrafları okur ve yalnız halen `null` kalan `image_vlm` izinli boolean alanları yalnız açık görsel kanıt varsa `true` doldurur.
6. Görsel boolean çıkarımı true-only'dir: açık kanıt `true`, görünmeyen değer `null`.

Sonraki aşama önceki kaynağın değerini overwrite edemez.

### 7c. Chroma metadata ve slot extraction

Canonical scalar alanlar registry slug'larıyla metadata olur. Multi-enum alanlar deterministik birleşik scalar değerin yanında seçenek başına exact flag üretir: örneğin `balcony_type__acik_balkon = true`.

```json
{
  "hard_filters": {
    "filters": {
      "district": ["Kadikoy"],
      "room_count": ["2+1"],
      "has_elevator": true,
      "balcony_type": ["acik_balkon"]
    },
    "max_price_amount": 30000,
    "price_currency": "TL"
  },
  "free_form_tr": "<orijinal sorgu>"
}
```

Explicit destekli kullanıcı istekleri metadata hard filter'a çevrilir. `null` metadata yazılmadığı için explicit istekle eşleşmez. BGE-M3 ve reranker yalnız bu elemeden sonra sıralama yapar.

### 7d. Gold compatibility

Eski `facts_gold` ve `visual_gold` benchmark'ları tarihsel ölçümleri açıklamak için korunur. Canonical registry için yeni gold template üretmek zaman maliyeti nedeniyle iptal edildi; yeni akış ilk etapta 10 ilanlık manuel göz kontrolüyle değerlendirilecek.

Göz kontrolü için `data/processed/clean_json.json` kullanılır. Bu okunabilir JSON listesi labeling output'una her row yazıldığında yeniden senkronlanır; yalnız `id`, `url`, `title`, `description`, dolu scraper fact'leri, DeepSeek boolean kararları ve Kimi positive-only boolean kararlarını içerir. Görsel yolları ve compatibility tekrarları yazılmaz.

## 8. Değerlendirme metrikleri

| Aşama | Metrik | Hedef | Gold kaynağı |
|---|---|---|---|
| Slot extraction | JSON-valid + slot accuracy | quality_score ≥ 0.85 | `llm/shootout.py:BENCHMARK_QUERIES` |
| Description extraction | per-field accuracy (7 hybrid+desc alanı) | ≥ 0.75 | `labeling/gold_listings_manual_todo.jsonl:facts_gold` |
| Image labeling | 10 yeni ilanda doğrudan göz kontrolü | kullanıcı onayı | `data/processed/labeled_benchmark_kimi_vision_10.jsonl` |
| Retrieval | R@5, R@10, MRR | R@10 ≥ 0.60 | `evaluation/gold_queries_manual_todo.jsonl` |
| Time series | MAE / sMAPE | rapor (baseline'a göre) | sentetik gold |

## 9. Proje dizini (mevcut)

```
DL-Ev-ilan-Projesi/
├── scraper/           # playwright_scraper.py, cleaner.py, image_downloader.py
├── schema/            # emlakjet_filters.py — canonical registry
├── llm/               # clients.py, shootout.py, shootout_vision.py,
│                      # shootout_description.py, gold_benchmark.py
├── labeling/          # gold_helper.py, gold_listings_manual_todo.jsonl
├── evaluation/        # gold_helper.py, gold_queries_manual_todo.jsonl
├── labeling/run_labeling.py  # M3 labeling pipeline (VLM+LLM merge, pre-flight gate)
├── indexing/          # composer.py + build_chroma.py (M4, Codex)
├── retrieval/         # retriever.py (M4, Codex) — slot extract + filter + rerank
├── model/             # (boş) — fine-tune scriptleri
├── finetune/          # (boş) — BGE-M3 LoRA + ResNet classifier (M6)
├── chat/              # eski mimari M5 iskeleti; canonical entegrasyon bekliyor
├── ui/                # eski mimari Streamlit iskeleti; canonical entegrasyon bekliyor
│                      # timeseries/ KALDIRILDI — M7 kapsam dışı (arkadaş yaptı)
├── tests/             # 68 test: registry, scraper, cleaner, labeling, composer, retriever, RAG
├── data/
│   ├── raw/           # Playwright çıktısı (listings.jsonl 305 ilan, 150 canonical slug)
│   ├── processed/dataset.jsonl  (303 aktif satır, canonical filter_values)
│   ├── processed/labeled.jsonl  (M3 full build çıktısı — HENÜZ YOK; sadece labeled_preflight* smoke var)
│   ├── processed/chroma/  # vector store (M4 build sonrası; henüz yok)
│   └── eval/manual_queries.jsonl  (eski auto-BM25 çıktısı, kullanma)
├── docs/              # llm_setup_tr.md, manual_gold_todo_tr.md, + bu dosya
├── archive/           # eski CLIP işleri
└── END439E_HW6.ipynb  # eski CLIP fine-tune (arşivlik)
```

## 10. Milestone haritası

| # | Ad | Çıktı | Durum |
|---|---|---|---|
| 0 | Scraper + cleaner (canonical) | 305 raw → 303 aktif, 150 slug | **DONE** (2026-05-31) |
| 1 | LLM slot shootout | `selected.json` text aday | **DONE** — text=deepseek_v4_pro (A/B revize), vision=kimi_k2_6 |
| 2 | Manual kontrol | 10 ilanlık göz kontrolü | **AKTİF** — yeni gold template rebuild iptal |
| 1.5 | Vision shootout | `vision_model` kararı | **DONE** — Kimi multi-image winner |
| 3 | Labeling pipeline | `labeled.jsonl` | **DONE iskele + iki geçişli CLI**; full 303 build PENDING |
| 4 | Indexing + retrieval | chroma + retriever | **ESKİ MİMARİ İSKELET** — canonical uyarlama + gerçek build PENDING |
| 5 | RAG chat + UI | Streamlit demo | **ESKİ MİMARİ İSKELET** — canonical uyarlama PENDING |
| 6 | Fine-tune NN'ler | BGE-M3 LoRA + ResNet | PENDING — **NN gereksinimi için kritik** (M7 dışarı çıktı) |
| 7 | Time series | LSTM/GRU + notebook | ⛔ **KAPSAM DIŞI** (sınıf arkadaşı yaptı) |
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
