# STATUS.md — Mevcut Durum (snapshot)

> Son güncelleme: 2026-05-26 13:00, schema 21→22 alan refactor (heating_type+is_furnished STRUCTURED'a taşındı, has_aircon eklendi), cleaner `_normalize_blob` Türkçe `İ` bug fix, gold template prefilled visual ile regen edildi.
> Bu dosyayı sıradaki agent her milestone bitince güncellemeli.

---

## TL;DR

- **M0 (scraper + cleaner + yeni şema)** — **DONE** (2026-05-26 13:00 schema refactor sonrası): 1483 ham → **1239** temizlenmiş ilan, **22 facts** top-level (15 structured + 5 hybrid + 2 desc-only) + boş `visual_qualities` object.
- **M1 (slot shootout)** — **DONE** (2026-05-25): Kazanan **`kimi_k2_6`** (quality 0.897). `llm/selected.json` text+vision = Kimi (vision provisional, M1.5 ile kesinleşecek).
- **M2.0 (helper + schema refactor)** — **DONE** (2026-05-26 13:00): heating_type + is_furnished STRUCTURED'a taşındı, `has_aircon` hybrid alan eklendi, `_normalize_blob` Türkçe büyük harf bug'ı fix, visual gold prefilled regen.
- **M2 (manuel gold)** — **AKTİF**: yeni schema'da 30 listing template hazır (`labeling/gold_listings_manual_todo.jsonl` — facts 7 user-fill + visual prefilled), 30 sorgu template hazır. Kullanıcı doldurmaya başlayabilir. Şu an 0/30 dolu.
- **M1.5 (vision + description shootout)** — BLOKE: M2 dolmasını bekliyor.
- **M3–M8** — M2 + M1.5 bitince başlar.

Veri yedeği: `archive/pre_schema_refactor/` (önceki 1430 ilanlık dataset) + `archive/pre_scraper_fix/` (en eski yedek) + `data/raw/listings_PRE_FIX.jsonl`.

**⚠️ Composer güvenilirlik uyarısı:** Bu projede Cursor Composer 2.5 ajanı 4 kere yalan rapor verdi (fake heating value, fake gold regen, fake satır sayısı, fake regen done mesajı). Her milestone'da raporu DOSYADAN doğrulayın, sadece ajan iddiasına güvenmeyin.

---

## Milestone-by-milestone

### M0 — Scraper + cleaner + yeni şema (**DONE**, 2026-05-26)

**Tarihçe:**
- 2026-05-24: İlk re-scrape (1430 ilan). Eski şema (`text_gold` 10 alan + `image_gold` 11 alan).
- 2026-05-25 22:30: Şema redesign — `facts_gold` (21 alan) + `visual_gold` (12 alan, başta 18 önerildi, "Orta" güvenlikler kesildi).
- 2026-05-26 03:36 başlangıç → 05:12 bitiş: Yeni şema ile full re-scrape.
- 2026-05-26 11:25: Cleaner çalıştırıldı, gold template regen edildi (Python tek satır script, Composer yalan söylediği için kullanıcı manuel çalıştırdı).

**Scraper değişiklikleri:**
- `INFO_FIELD_MAP` genişletildi: `inGatedComplex`, `deposit`, `titleDeedStatus`, `maintenanceFee`, `occupancy`, `tradeAccepted`, `bathroomCount`, `priceStatus`.
- `MAX_DESCRIPTION_LEN = 4000`.
- Unicode arrow karakteri (`→`) `->` ile değiştirildi (Windows cp1254 console patlamasın diye).
- DOM selektörleri: "İlan Bilgileri" tablo, "İlan Özellikleri" bullet listesi, "Açıklaması" başlığı + "Daha Fazla Gör" expand.

**Cleaner değişiklikleri:**
- 21 facts alanı top-level (`city`, `district`, ..., `near_metrobus`, `title_deed_status`).
- `visual_qualities: {}` boş object (M3 dolduracak).
- `_parse_tl`, `_parse_int` helper'ları.
- `_normalize_heating` enum normalize (`kombi|dogalgaz|merkezi|klima|yerden_isitma|null`).
- HTML leakage regex (`<[^<]*$`).

**Re-scrape sonuçları (2026-05-26):**
- Ham: `data/raw/listings.jsonl` — **1483 unique** ilan (toplam 2928 satır, resume duplicate dahil)
- Görseller: `data/images/` — ~25k foto
- İşlenmiş: `data/processed/dataset.jsonl` — **1239** ilan (`cleaning_report.json`)

**Sağlık metrikleri (2026-05-26, n=1239):**

| Metrik | Sayı | % | Eşik | Durum |
|---|---|---|---|---|
| empty_desc | 0 | 0.0% | <10% | ✅ |
| has_features | 918 | 74.1% | >70% | ✅ |
| has_heating (normalize) | 1222 | **98.6%** | >80% | ✅ |
| has_in_gated_complex | 1239 | 100.0% | — | ✅ yeni alan |
| has_title_deed | 1239 | 100.0% | — | ✅ yeni alan |
| has_deposit | 376 | 30.3% | — | ⚠ ilanların 30%'inde explicit depozito |
| has_kitchen_type (features keyword) | 134 | 10.8% | — | ⚠ düşük → LLM extract edecek |
| has_balcony (features keyword) | 112 | 9.0% | — | ⚠ düşük → LLM extract edecek |
| has_elevator (features keyword) | 58 | 4.7% | — | ⚠ düşük → LLM extract edecek |
| has_aircon (features+title) | 218 | 17.6% | — | ⚠ kalan kısmı LLM/VLM extract edecek |
| is_furnished (features+title) | 229 | 18.5% | — | ⚠ kalan kısmı LLM extract edecek |
| has_parking (features keyword) | 35 | 2.8% | — | ⚠ düşük → LLM extract edecek |

> "Düşük" yüzdeler **bug değil** — bunlar property_features keyword tarama ile **True olarak işaretlenmiş** ilan sayısı. `False` veya `null` (bilinmiyor) olan ilanlar bu sayıya dahil değil. LLM extraction (M3) bu boşlukları title + description'dan dolduracak.

Cleaning report:
```json
{
  "total_raw": 2928, "skipped_no_image": 256, "skipped_bad_image": 0,
  "skipped_duplicate_url": 1429, "skipped_duplicate_image": 4,
  "skipped_no_text": 0, "saved": 1239, "retention_rate": 42.3
}
```
Anlamlı retention 1239 / 1483 unique = **%83.5**.

### M1 — LLM slot shootout (**DONE**, 2026-05-25)

Final 4-model run (`llm/shootout_report.md`, eski dataset 1430 ilan, 30 query benchmark, 3 few-shot örnek):

| Model | Status | quality | json | slot | 100K $ |
|---|---|---|---|---|---|
| **kimi_k2_6** | ok | **0.8967** | 1.0000 | **0.8278** | $146.50 |
| gemma_4_local | ok | 0.8833 | 1.0000 | 0.8056 | $0.00 |
| deepseek_v4_flash | ok | 0.8733 | 1.0000 | 0.7889 | $15.40 |
| gemini_3_5_flash | error 429 | — | — | — | $285.00 |

**Karar:** `llm/selected.json` → `text_model = vision_model = kimi_k2_6`. Akademik tartışma için ikinci aday Gemma4 (cost 0, %1.4 quality drop).

Gemini free tier (20 req/day per model) yetmedi; billing açılırsa veya günlük reset sonrası tekrar koşulabilir.

> **Not:** Shootout eski dataset üzerinde (1430 ilan, eski şema) yapıldı. Yeni dataset slot extraction benchmark sonuçlarını değiştirmemeli — query/slot tarafı dataset-bağımsız.

Artifacts: `llm/selected.json`, `llm/shootout_rows.json`, `llm/shootout_rows_v2.json`, `llm/shootout_report.md`.

### M1.5 — Vision + Description shootout (BLOKE — gold bekliyor)

`llm/shootout_vision.py` ve `llm/shootout_description.py` kod hazır.

Gold dolduktan sonra koşulacak:
- **Vision** (10 listing × ~5 foto): adaylar `gemma_4_local`, `kimi_k2_6` (Gemini 429 sebebiyle dışlandı). `visual_gold` 12 alan ile per-field Jaccard accuracy.
- **Description** (10 listing × 1 desc): adaylar `deepseek_v4_flash`, `kimi_k2_6`, `gemma_4_local`. `facts_gold`'un 8 hybrid+desc-only alanı ile per-field accuracy.

Karar kuralı:
- Vision: accuracy ≥ 0.70 olan en ucuz aday → Gemma4 yeterliyse $0
- Description: accuracy ≥ 0.75 olan en ucuz aday → M1 winner Kimi zaten ileride ama Gemma4 yeterliyse cost avantajı

Eğer Gemma4 her ikisinde de yeterliyse M3'te tüm labeling'de Gemma kullanılabilir → 1239 listing × ~6 call = $0.

### M2 — Manual gold sets (HAZIR — kullanıcı doldurmaya başlayabilir)

**2026-05-26 13:00 regenerate** (schema refactor sonrası yeni `dataset.jsonl`'den 30 listing, mevcut ID'ler korundu):

- `labeling/gold_listings_manual_todo.jsonl` — **30 satır**, yeni 22-facts şema:
  - `facts_gold` (22 alan): 15 structured otomatik dolu (city, district, ..., title_deed_status, **heating_type, is_furnished**), 7 user-fill null (kitchen_type, has_balcony, has_elevator, has_parking, **has_aircon**, near_metro, near_metrobus)
  - `visual_gold` (12 alan): **prefilled** — multi-select alanlar tüm enum array içinde, single-select alanlar '|'li string. Kullanıcı görülmeyen değerleri siler, single-select için tek değer bırakır.
- `evaluation/gold_queries_manual_todo.jsonl` — **30 sorgu**, `expected_listing_ids=[]` (kullanıcı 1-3 ID seçecek)

**Toplam manuel iş:** 30 listing × ~19 değerlendirme + 30 query expected_ids = **~3-4 saat**.

Helper'lar (yeni schema'ya göre çalışıyor):
- `python -m labeling.gold_helper --listing <ID>` — listing detayı + 15 structured facts (otomatik dolu) + 7 suggested hybrid facts + suggested visual fields cross-validation + manual TODO listesi
- `python -m evaluation.gold_helper --query "..."` — query için BM25 + hard-filter top-K candidate

### M3 — Labeling pipeline (PENDING)

- `labeling/run_labeling.py` (henüz yok): her ilan için VLM + LLM çağırıp `data/processed/labeled.jsonl` üretir.
- Batch boyutu, retry, cost cap, resume desteği gerekli.
- Output şeması: dataset.jsonl + 21 facts (boş hybrid+desc alanları LLM ile dolar) + visual_qualities (12 alan VLM ile dolar) + enriched_doc (embedding metni) + confidence skorları.

### M4 — Indexing + retrieval (PENDING)

- `indexing/composer.py`: enriched_doc → BGE-M3 embedding metni.
- `indexing/build_chroma.py`: ChromaDB persistent collection.
- `retrieval/retriever.py`: query → slot extract → metadata filter + vector search + reranker.

### M5 — RAG chat + Streamlit (PENDING)

- `chat/rag_response.py`: retrieved listings + soru → LLM ile cevap kompozisyonu.
- `ui/app.py`: chat box + listing cards + 3 demo senaryo.

### M6 — Fine-tune (PENDING)

- `finetune/bge_lora.py`: BGE-M3 LoRA on listing pairs.
- `finetune/image_classifier.py`: ResNet-18 / ViT-small on `visual_gold` labels (3-4 alan: balkon_tipi, manzara, zemin_tipi).

### M7 — Time series (PENDING)

- Sentetik kira fiyat serisi (district x oda) + LSTM/GRU + notebook.
- Scraped fiyat istatistikleri ile calibrate edilmeli.

### M8 — Final report (PENDING)

- Rapor notebook'u + README rewrite + sunum akışı.

---

## Açık sorular / kararlar

1. **`is_furnished` re-scrape gerekli mi?** Cleaner şu an attribute-yoksa-keyword fallback ile çalışıyor (`furnishedStatus` raw'da yok). Mevcut 229 True/1010 None. Gelecek bir scrape'te scraper `furnishedStatus` çekecek (INFO_FIELD_MAP güncel). Re-scrape M3 öncesi yapılırsa is_furnished gerçek structured olur.
2. **`heating_type` enum dışı değerler:** 17 ilanda heating_type=null (Sobalı, VRV, Fueloil, vs). Bunlar enum'a eklenmeli mi yoksa "diger" kategorisi mi açılmalı?
3. **Scrape genişletme:** 1239 yeterli mi, ek 1500-2000 incremental scrape edilsin mi? — M3 + M4 retrieval sonuçlarına bakılmalı.
4. **Gemini free tier:** 429 sebebiyle shootout'larda kullanılamadı. Billing açma kararı? — şu an Kimi+Gemma yeterli.

---

## Bilinen riskler

| Risk | Etki | Mitigation |
|---|---|---|
| Composer yalan rapor verir | Tamamlanmamış işi tamamlanmış sanma | Her milestone'da DOSYADAN doğrula |
| Gemma4 vision Türkçe ev fotoğrafında zayıf çıkabilir | M3 labeling kalitesiz | Gold ile ölç, gerekirse Kimi'ye geç ($25-30 ek maliyet) |
| 1239 ilan fine-tune için az olabilir | LoRA underfit/overfit | Synthetic pair generation veya hard-negative mining; pre/post karşılaştırma |
| Time series için gerçek veri yok | Tek snapshot'tan trend yok | Sentetik + scraped ortalamayı calibration anchor |
| Hocaya teslim deadline yaklaşıyor | Zaman baskısı | M6'yı atla, M5 minimal yap (önceki notlar) |

---

## Cost tracker (yaklaşık)

| Adım | Harcanan | Tahmini ek | Notlar |
|---|---|---|---|
| M1 4-model shootout | ~$2 | — | Kimi+Gemma+DeepSeek+Gemini (kısmî) |
| M1.5 vision+desc shootout | $0 | $1-2 | 10 listing × 2-3 model |
| M3 labeling 1239 ilan | $0 | $0-25 | Gemma4 ise $0, Kimi ise ~$20-25 |
| **TOPLAM** | ~$2 | $1-27 | $100 bütçenin çok altında |

---

## Test durumu

```
tests/test_llm_shootout.py
tests/test_gold_helpers.py
tests/test_scraper_description_limit.py
tests/test_scraper_extraction.py
tests/test_cleaner_preserves_fields.py
```

Komut: `.\.venv\Scripts\python.exe -m pytest -q` — **27 passed** (2026-05-26, schema refactor + Türkçe normalize bug fix sonrası).

---

## Kalan iş + zaman tahmini (2026-05-26 12:20 itibariyle)

| # | İş | Kim | Wall-clock | Bağımlılık |
|---|---|---|---|---|
| M2 | Manuel gold doldurma (30 listing + 30 query) | Sen | ~3-4 saat | — |
| M1.5 | Vision + description shootout (2-3 model) | Agent | ~15 dk | M2 |
| M3 | Labeling pipeline (run_labeling.py + 1239 ilan batch) | Agent | ~3-4 saat | M1.5 |
| M4 | Indexing (BGE-M3 + Chroma) + retriever | Agent | ~2.5 saat | M3 |
| M5 | RAG chat + Streamlit UI + 3 demo | Agent | ~3 saat | M4 |
| M7 | Time series (sentetik + LSTM/GRU + notebook) | Agent | ~4 saat | — (paralel) |
| M6 | Fine-tune (BGE-M3 LoRA + ResNet classifier) | Agent | ~6 saat (training dahil) | M3 |
| M8 | Final rapor notebook + README + sunum taslağı | Agent + Sen | ~4 saat | hepsi |

**Toplam agent wall-clock:** ~20-25 saat (paralelleştirilirse ~14-17 saat).
**Sen:** ~3-4 saat gold + ~2-3 saat review/iteration = ~5-7 saat.

### Acil teslim için scope kesme önerileri
- **M6 fine-tune'u atla** → "demonstrated feasibility, future work" diye not düş. Hocanın NN gereksinimi zaten M7 LSTM/GRU ile karşılanıyor. ~6 saat tasarruf.
- **M5 UI'ı minimal yap** (chat box + 3 demo senaryo, kart süslemesi yok). ~2 saat tasarruf.
- **Minimum viable demo:** M2+M1.5+M3+M4+M5(minimal)+M7+M8 ≈ 13-15 saat agent + 5-7 saat sen.
