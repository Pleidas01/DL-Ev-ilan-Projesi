# STATUS.md — Mevcut Durum (snapshot)

> Son güncelleme: 2026-05-29, **vision multi-image refactor**: `complete_vision_json` çoklu image (tek çağrı/ilan), `shootout_vision` per-image aggregate kaldırıldı + listing-başına resilience. Kimi multi-image gold testi geçti (bkz. M1.5). Önceki (2026-05-28): schema sadeleştirme `kitchen_type` kaldırıldı, visual_gold 6 alan (zemin_tipi de çıktı), `site_imkanlari`→`imkanlar`, listing gold 10/30.
> Bu dosyayı sıradaki agent her milestone bitince güncellemeli.

---

## TL;DR

- **M0 (scraper + cleaner + yeni şema)** — **DONE** (2026-05-26 13:00 schema refactor sonrası): 1483 ham → **1239** temizlenmiş ilan, **21 facts** top-level (15 structured + 4 hybrid + 2 desc-only) + boş `visual_qualities` object.
- **M1 (slot shootout)** — **DONE** (2026-05-25): Kazanan **`kimi_k2_6`** (quality 0.897). `llm/selected.json` text+vision = Kimi (vision provisional, M1.5 ile kesinleşecek).
- **M2.0 (helper + schema refactor)** — **DONE** (2026-05-26 13:00): heating_type + is_furnished STRUCTURED'a taşındı, `has_aircon` hybrid alan eklendi, `_normalize_blob` Türkçe büyük harf bug'ı fix, visual gold prefilled regen.
- **M2 (manuel gold)** — **AKTİF**: sadeleştirilmiş schema (21 facts + 7 visual). Listing gold **10/30 dolu** (M1.5 eşiği ≥10 geçildi). Query gold (30 sorgu) M4'e ertelendi.
- **M1.5 (vision shootout)** — **VISION DONE** (2026-05-29, multi-image refactor): Kimi multi-image (tek çağrı/ilan, tüm foto) **10/10 ilan tamamladı**, accuracy **0.748** (10 ilan); per-image ile ortak 3 ilanda **0.931 ≥ 0.917** → dilution yok, robust, **~3x ucuz ($0.036/ilan)**. Karar: M3 multi-image+Kimi. (Description shootout kısmı hâlâ açık — heating_type haksızlığı düzeltilmeli.)
- **M3–M8** — M3 sıradaki: `run_labeling.py` (multi-image + Kimi, **eşzamanlı** çağrı, resume, cost cap).

Veri yedeği: `archive/pre_schema_refactor/` (önceki 1430 ilanlık dataset). Not: `archive/hw6` + `pre_scraper_fix` + `data/_*` temizlikte silindi (bkz. checkpoint 6b019f7).

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

### M1.5 — Vision + Description shootout (KISMİ — 2026-05-28)

**Vision sonucu (visual_gold 6 alan, per-image extract + aggregate, tüm foto):**
| Model | Accuracy | Not |
|---|---|---|
| **kimi_k2_6** | **0.917** | mutfak_tipi 1.0, banyo 1.0 — ama sadece 3 listing (timeout kesti) |
| gemma_4_local | 0.135 | yetersiz; kararsız, enum-dışı uyduruyor |

Kimi kalitesi net üstün AMA per-image tasarım **kırılgan + pahalı + yavaş** çıktı:
- 161 ayrı çağrı → 4. listing'de bir foto 90s timeout → tüm run "error" (sadece 3 listing tamamlandı). 51 dk sürdü.
- **Ölçülen maliyet: foto başına ~$0.007** (Kimi image ~5000 token). 1500×16 foto = ~$168 (imkansız), 1500×4 = ~$42, 500×4 = ~$14.
- Leak kontrolü yapıldı: **leak YOK** (prompt'ta gold yok, manzara'yı yanlış bildi, imkanlar farklı sırada).

**Bug'lar bulundu + düzeltildi:** ollama base64 (vision hiç çalışmıyordu), aggregate_model_scores "samples" çakışması (detay kaybı), Türkçe→ASCII normalize eksik (boğaz≠bogaz), kimi vision timeout yok.

**Şema sadeleştirme:** `kitchen_type` çıktı (mutfak_tipi'nde), visual 12→6 (zemin_tipi parke/laminat ayırt edilemiyordu → çıktı), `site_imkanlari`→`imkanlar`, `deepseek_v4_pro` eklendi (%75 indirimli $0.435/$0.87).

**Description sonucu (eski, heating_type structured haksızlığıyla):** deepseek_flash 0.28, kimi 0.28, gemma 0.32 — heating_type/is_furnished structured alanlar haksız ceza veriyor, çıkarılmalı.

**M3 önerisi (per-image sonrası):** multi-image (161→10 çağrı: hız+ucuz+az timeout) + Kimi. Multi-image kalite riski (attention dilution) gold'da test edilmeli.

#### Multi-image refactor + test (2026-05-29) — **VISION DONE**

Per-image → **tek çağrı/ilan** (tüm fotoğraflar tek pakette, model tek bütünleşik JSON). `clients.complete_vision_json` artık `image_paths: list[str]`; `shootout_vision` listing-başına try/except ile resilient. Foto cap konmadı (per-image ile adil karşılaştırma için tüm foto).

Kimi multi-image (10 gold listing, tüm foto = 161, `llm/shootout_vision_multi_rows.json`):
| Metrik | Multi-image | Per-image (önceki) |
|---|---|---|
| Tamamlanan | **10/10, hata yok** | 3/10 (timeout kesti) |
| Ham accuracy | **0.748** (10 ilan) | 0.917 (sadece 3 ilan) |
| Ortak 3 ilanda | **0.931** | 0.917 |
| Süre | 19 dk (seri) | 51 dk (seri, kısmi) |

- **0.748 < 0.917 elma-armut**: per-image yalnızca kolay ilk 3 ilanı bitirdi. Adil karşılaştırma (her ikisinin de tamamladığı 3 ilan) → **multi 0.931 ≥ per 0.917**; 19-fotolu ilanda bile multi (0.792) > per (0.750) ⇒ **attention dilution YOK**. 0.748, per-image'ın hiç ulaşamadığı 7 zor ilandan (0.33–0.63) geliyor; yine de M3 eşiği ≥0.70'in üstünde.
- Per-field (multi): mutfak_tipi 0.900 (n=10), banyo 0.738 (n=7) güçlü; balkon 0.375 / imkanlar 0.458 / salon 0.333 zayıf ama **düşük-n** (gürültülü).
- **Ölçülen maliyet** (Moonshot bakiye farkı $1.26385→$0.90311): **$0.3607** = **$0.036/ilan, $0.00224/foto** → per-image'ın $0.007/foto'suna göre ~3x ucuz. M3 ekstrapolasyon ~1239 ilan ≈ **~$45**.
- **Zamanlama:** 19 dk seri (CPU 1.2s; tamamı ağ-bound). M3'te eşzamanlı (20-30 worker) → ~1-2 saat. Seri DEĞİL.

**Karar:** M3 labeling **multi-image + Kimi** ile yapılır. Caveat: 10 gold küçük örneklem, zayıf per-field'lar düşük-n; M3 pre-flight'ta prompt tune + gold 10→30 önerilir.

### M2 — Manual gold sets (AKTİF — 10/30 dolu)

**2026-05-28 schema sadeleştirme** (kitchen_type çıktı, visual 12→7, imkanlar rename; 30 listing ID korundu, template regen):

- `labeling/gold_listings_manual_todo.jsonl` — **30 satır**, sadeleştirilmiş 21-facts şema:
  - `facts_gold` (21 alan): 15 structured otomatik dolu, 6 user-fill (has_balcony, has_elevator, has_parking, has_aircon, near_metro, near_metrobus). `kitchen_type` kaldırıldı.
  - `visual_gold` (7 alan): balkon_ozellikleri, manzara, mutfak_tipi, banyo_ozellikleri, zemin_tipi, salon_ozellikleri, imkanlar. **10/30 dolduruldu**, kalan prefilled.
  - Doldurma kuralı: gördüğün/emin olduğun → işaretle; negatif kanıtla emin "yok" → false/`[]`; emin değil/görünmüyor → null (benchmark'ta skip).
- `evaluation/gold_queries_manual_todo.jsonl` — **30 sorgu**, `expected_listing_ids=[]`. **M4'e ertelendi** (retriever olmadan test edilemez).

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
| M1.5 vision per-image (önceki) | ~$0.11 | — | 3 listing tamamlandı (timeout) |
| M1.5 vision multi-image (2026-05-29) | **$0.36** | — | Kimi, 10 ilan, 161 foto, bakiye farkıyla ölçüldü |
| M3 labeling 1239 ilan | $0 | ~$45 | Kimi multi-image ekstrapolasyon ($0.036/ilan) |
| **TOPLAM** | ~$2.5 | ~$45 | $100 bütçenin altında |

---

## Test durumu

```
tests/test_llm_shootout.py
tests/test_gold_helpers.py
tests/test_scraper_description_limit.py
tests/test_scraper_extraction.py
tests/test_cleaner_preserves_fields.py
```

Komut: `.\.venv\Scripts\python.exe -m pytest -q` — **27 passed** (2026-05-28, schema sadeleştirme + imkanlar rename + shootout fix sonrası).

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
