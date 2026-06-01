# STATUS.md — Mevcut Durum (snapshot)

> Son güncelleme: 2026-06-01, **M6 MacBook hazırlığı DONE**: pair dataset, dense-only evaluator ve CUDA-gated PEFT LoRA CLI hazır. Full suite `112 passed`, skip yok. Sıradaki: RTX 5070 Ti makinede ölçüm kapılı BGE-M3 LoRA deneyi + sunum.
> Önceki (2026-05-30): canonical Emlakjet filter enrichment Task 6 pre-scrape checkpoint, `62 passed`, registry-first mimari belgeleri senkronize.
> Önceki (2026-05-29): vision multi-image refactor (tek çağrı/ilan), Kimi multi-image gold testi geçti (bkz. M1.5).
> Bu dosyayı sıradaki agent her milestone bitince güncellemeli.
> **Doküman-hijyeni:** Üç nesil (Gen1 CLIP → Gen2 el-yapımı şema → Gen3 canonical) ve "neden terk ettik" için `MIMARI_EVRIMI.md`. 2026-06-01 itibariyle M4 ve M5 Gen3 baseline tamamlandı.

---

## TL;DR

- **M0 (scraper + cleaner)** — **DONE**: aktif canonical batch 305 ham → **303** temiz ilan, 150 slug. Aşağıdaki 1239-ilan bölümü yalnız tarihsel arşiv bilgisidir.
- **M1 (slot shootout)** — **DONE** (2026-05-25, 2026-05-30 revize): M1'de kazanan `kimi_k2_6` (quality 0.897) idi; **2026-05-30 gold A/B** ile `text_model` **`deepseek_v4_pro`**'ya çevrildi (aynı koşullu head-to-head: slot 0.815 > kimi 0.792, ~3.5x ucuz, ~5x hızlı). `llm/selected.json`: **text=deepseek_v4_pro, vision=kimi_k2_6**.
- **M2.0 (helper + schema refactor)** — **DONE** (2026-05-26 13:00): heating_type + is_furnished STRUCTURED'a taşındı, `has_aircon` hybrid alan eklendi, `_normalize_blob` Türkçe büyük harf bug'ı fix, visual gold prefilled regen.
- **M2 (manuel kontrol)** — **AKTİF**: önceki gold tarihsel referans olarak korunuyor. Canonical registry için yeni template rebuild yapılmayacak; zaman maliyeti yerine yeni 10 ilanlık benchmark çıktısı kullanıcı tarafından gözle incelenecek.
- **M1.5 (vision shootout)** — **VISION DONE** (2026-05-29, multi-image refactor): Kimi multi-image (tek çağrı/ilan, tüm foto) **10/10 ilan tamamladı**, accuracy **0.748** (10 ilan); per-image ile ortak 3 ilanda **0.931 ≥ 0.917** → dilution yok, robust, **~3x ucuz ($0.036/ilan)**. Karar: M3 multi-image+Kimi. (Description shootout kısmı hâlâ açık — heating_type haksızlığı düzeltilmeli.)
- **M3 (labeling pipeline)** — **DONE**: iki geçişli CLI + full **303-ilan** build tamamlandı. `labeled.jsonl`: scraper `10443`, DeepSeek `2140`, Kimi `638` kaynak damgası; Kimi kaynaklı değerlerin tamamı positive-only `true`.
- **M4 (indexing + retrieval)** — **BASELINE DONE (2026-06-01)**: Gen3 index gerçek `labeled.jsonl` ile kuruldu (`303` collection row). Metadata kontratı ve birleşik multi-enum flag düzeltmesi doğrulandı. MacBook Air M4 üzerinde bounded eval: synthetic known-item `n=4`, `R@1/R@5/R@10/MRR=1.0`; canlı benchmark `n=3`, coverage `1/3`, filter satisfaction `8/8=1.0`. Bu full relevance gold değildir; kontrollü proxy örneklemidir.
- **M5 (RAG + UI)** — **BASELINE DONE (2026-06-01)**: `chat/rag_response.py` + `ui/app.py` Gen3 `filters` çıktısını okuyor. Canonical label ve deterministik match çipleri render oluyor. Browser smoke'ta tek canlı sorgu `8` kart döndürdü; RAG cevap yalnız retrieved ilanları kullandı. M6/M8 pending; M7 kapsam dışı.
- **Canonical filter enrichment** — **ilk batch doğrulaması DONE (2026-05-31)**: 305 raw → 303 cleaned. Raw'da ortalama 34.47 dolu canonical filtre/ilan; cleaner sonrası ortalama 15.81 yerel görsel/ilan. DeepSeek 10 ilanda 38, Kimi 10 ilanda 15 ek alan doldurdu.

Veri yedeği: `archive/pre_schema_refactor/` (önceki 1430 ilanlık dataset). Not: `archive/hw6` + `pre_scraper_fix` + `data/_*` temizlikte silindi (bkz. checkpoint 6b019f7).

**⚠️ Composer güvenilirlik uyarısı:** Bu projede Cursor Composer 2.5 ajanı 4 kere yalan rapor verdi (fake heating value, fake gold regen, fake satır sayısı, fake regen done mesajı). Her milestone'da raporu DOSYADAN doğrulayın, sadece ajan iddiasına güvenmeyin.

---

## Milestone-by-milestone

### M0 — Scraper + cleaner tarihçesi (**ARŞİV**, 2026-05-26)

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

**Arşiv re-scrape sonuçları (2026-05-26; aktif dataset değil):**
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

**Karar (M1, 2026-05-25):** `llm/selected.json` → `text_model = vision_model = kimi_k2_6`. Akademik tartışma için ikinci aday Gemma4 (cost 0, %1.4 quality drop).

**Revize (2026-05-30) — text_model → deepseek_v4_pro:** M1 skoru eski dataset + `temperature=1` varyansıyla alınmıştı. **Aynı koşullu gold A/B** (16 ilan labeling + 30 query slot, ThreadPoolExecutor):
- Query slot quality **deepseek 0.815 > kimi 0.792** (slot 0.815 vs 0.792, JSON ikisi de 1.0).
- Labeling text-fact ~berabere (0.315 vs 0.333 — ikisi de null-vs-false artefaktıyla düşük, bkz. Açık sorular).
- Maliyet ~3.5x ucuz ($0.0088 vs $0.0310 / 14 ilan), latency ~5x hızlı (deepseek temp=0 ~13s vs kimi temp=1 ~30s/çağrı).
- **Vision tarafı değişmedi**: `vision_model = kimi_k2_6` (M1.5 kazananı). Geri almak için `text_model`'i `kimi_k2_6` yap.

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

### M2 — Tarihsel manual gold seti (REFERANS — yeni rebuild yok)

**2026-05-28 schema sadeleştirme** (kitchen_type çıktı, visual 12→7, imkanlar rename; 30 listing ID korundu, template regen):

- `labeling/gold_listings_manual_todo.jsonl` — **30 satır**, sadeleştirilmiş 21-facts şema:
  - `facts_gold` (21 alan): 15 structured otomatik dolu, 6 user-fill (has_balcony, has_elevator, has_parking, has_aircon, near_metro, near_metrobus). `kitchen_type` kaldırıldı.
  - Eski manuel gold dosyası `salon_ozellikleri` kalıntıları içeriyor. Tarihsel ölçümleri açıklamak için korunuyor; canonical registry için yeniden üretilmeyecek.
  - Doldurma kuralı: gördüğün/emin olduğun → işaretle; negatif kanıtla emin "yok" → false/`[]`; emin değil/görünmüyor → null (benchmark'ta skip).
- `evaluation/gold_queries_manual_todo.jsonl` — **30 sorgu**, `expected_listing_ids=[]`. **M4'e ertelendi** (retriever olmadan test edilemez).

**Toplam manuel iş:** 30 listing × ~19 değerlendirme + 30 query expected_ids = **~3-4 saat**.

Helper'lar (yeni schema'ya göre çalışıyor):
- `python -m labeling.gold_helper --listing <ID>` — listing detayı + 15 structured facts (otomatik dolu) + 7 suggested hybrid facts + suggested visual fields cross-validation + manual TODO listesi
- `python -m evaluation.gold_helper --query "..."` — query için BM25 + hard-filter top-K candidate

### M3 — Labeling pipeline (DONE — full 303 build hazır)

- `labeling/run_labeling.py` **mevcut**: `--phase text` yalnız DeepSeek'i, `--phase vision` yalnız Kimi'yi, varsayılan `--phase combined` ikisini sırayla çalıştırır. Eşzamanlı çağrı, resume, cost cap ve pre-flight gold gate korunur.
- Kanıt kuralı: scraper fact'leri ezilmez. DeepSeek yalnız başlık+açıklamadan kalan null alanlara açık kanıtla `true` veya `false` yazabilir. Kimi yalnız görsellerden kalan null boolean alanlara açık kanıtla `true` yazabilir; görünmeyen özellik `false` olmaz.
- Tarihsel pre-flight kapısı `facts_all` idi. Yeni canonical batch için gold rebuild yapılmayacak; kullanıcı iki 10-ilan JSONL çıktısını doğrudan gözle inceleyecek.
- Output şeması: dataset.jsonl + 21 facts (boş hybrid+desc alanları LLM ile dolar) + visual_qualities (7 alan VLM ile dolar) + enriched_doc (embedding metni) + labeling_metadata.
- **Full 303-ilan build tamamlandı:** `data/processed/labeled.jsonl`. İlk benchmark 10 ilan ile kalan 293 ilan aktif dataset sırasına göre birleştirildi; duplicate ve missing ID yok.
- Göz kontrolü için `data/processed/clean_json.json` üretildi. `run_labeling.py`, asıl output JSONL'ye her row append edildiğinde bu okunabilir JSON listesini yeniden senkronlar. Listede görsel yolu yoktur; yalnız bağlam alanları ile source bazında `scraper`, `deepseek`, `kimi` fact grupları bulunur.

**10 ilan ölçümü (2026-05-31):**

| Koşu | Ayar | Wall-clock | Tahmini maliyet | Ek dolu alan |
|---|---|---:|---:|---:|
| DeepSeek text-only | 3 worker | 192.12s | $0.007736 | 38 |
| Kimi baseline | 512px, chunk=0, 1 worker, 3 ilan | 142.02s | — | — |
| Kimi chunk deneyi | 512px, chunk=5, 3 worker, 3 ilan | 259.62s | — | — |
| Kimi seçilen ayar | 512px, chunk=0, 10 worker, 10 ilan | 144.34s | $0.040671 | 15 |

Karar: DeepSeek text geçişi `--batch-size 3` ile koşuldu. Kimi benchmark'ında 10 worker hızlıydı ancak full batch'te timeout baskısı yarattı; operasyonel ayar `VISION_MAX_IMAGE_EDGE=512`, `--vision-chunk-size 0`, `--batch-size 5`. Yalnız timeout veren 20-görselli `19416208` ilanı `--vision-chunk-size 5`, `--batch-size 1` fallback ile tamamlandı.

### M4 — Indexing + retrieval (BASELINE DONE — 2026-06-01)

> **Tasarım:** `docs/superpowers/specs/2026-05-31-m4-m5-canonical-design.md` (komponent C1–C6). Gold doldurulmadığı için final relevance gold yerine kontrollü proxy + filtre regresyon ölçümü kullanıldı.

- `indexing/composer.py`: `embedding_text(record)` enriched_doc'u aynen döner (eksikse hata verir); `to_metadata(record)` canonical `filter_values` + title'ı **skalar** Chroma metadata'ya düzleştirir. List alanlar deterministik sıralı `|` ile birleşir; multi-enum alanlar ayrıca seçenek başına exact-match flag üretir.
- `indexing/build_chroma.py`: ChromaDB persistent collection builder (CLI). MacBook Air CPU-only ortamında varsayılan batch `8`; daha büyük batch bellek baskısı oluşturdu.
- `retrieval/retriever.py`: query → slot extract (`selected.json` text_model, read-only) → registry-backed metadata hard filter (`filters`, `any_of`, fiyat/m² range) → vektör arama (BGE-M3) → CrossEncoder reranker (bge-reranker-v2-m3). `null` metadata yazılmadığı için explicit filtreyi karşılamaz. DI ile test edilebilir (collection/embedder/reranker/slot_extractor enjekte).
- Gerçek build: `data/processed/labeled.jsonl` üzerinden collection count `303`; metadata skaler-only; birleşik `acik_balkon_acik_teras` flag kalmadı (`0` invalid row).
- Gold-free eval: `evaluation/run_retrieval_eval.py` limit, progress ve sorgu-başı checkpoint destekli. MacBook Air'de ağır full koşu yapılmadı. Kaydedilen hafif örneklem: synthetic `n=4`, `R@1/R@5/R@10/MRR=1.0`; canlı benchmark `n=3`, coverage `1/3`, dönen kısıtlı sonuç `8`, filter satisfaction `8/8=1.0`.
- Not: `coverage=1/3` retrieval bug'ı değil. İlk ve üçüncü sorgunun tüm hard filter'larını sağlayan ilan aktif `303` satırda yok. Ayrıca `"denize yakın"` semantiği benchmark'ta yanlışlıkla `has_sea_view` idi; canonical `near_sea` olarak TDD ile düzeltildi.

### M5 — RAG chat + Streamlit (BASELINE DONE — 2026-06-01)

**Yapıldı (2026-05-31, TDD):**
- `schema/emlakjet_filters.py::label_for(slug, value)`: enum slug→Türkçe label, bool True→özellik adı, False/None→None. (`tests/test_filter_labels.py`, 6 test)
- `retrieval/retriever.py::matched_filter_labels(slots, metadata)` + sonuç-dict'e `matched_filters`: sorgunun istediği VE ilanın karşıladığı filtreler, deterministik Türkçe çip (Rule 5). (`tests/test_retriever.py`, +3 test)
- `chat/rag_response.py`: `_build_user_prompt` artık `result["filters"]` + `matched_filters` okuyor (eski `facts` boş kalıyordu — bug). (`tests/test_rag_response.py`)
- `ui/app.py::card_fact_lines()`: `filters`'tan okunabilir kart satırları; `CARD_FACTS` `has_parking`→`has_closed_parking`; enum slug→label; match çipleri render. (`tests/test_ui_app.py`, 5 test)
- Browser QA: `python3 -m streamlit run ui/app.py` ile canlı demo açıldı. `"geniş salonlu denize yakın 2+1 asansörlü ev"` sorgusu `8` kart render etti; kartlarda `2+1`, `Asansör`, `Denize Yakın` çipleri ve canonical detaylar görüldü. RAG cevap yalnız retrieved ilanlardan üretildi.
- Streamlit güncel API uyumu: `use_container_width=True` kaldırıldı, `width="stretch"` kullanılıyor.
- Full suite: **104 passed**, skip yok.

- `chat/rag_response.py`: retrieved listings + soru → LLM ile cevap kompozisyonu.
- `ui/app.py`: chat box + listing cards + 3 demo senaryo.

### M6 — Fine-tune (SIRADAKİ — RTX 5070 Ti)

- MacBook Air hazırlık kodu tamamlandı: `finetune/text_embed/prepare_pairs.py`, `evaluate_dense.py`, `train_bge_m3_lora.py`.
- Pair çıktısı üretildi: `242` train, `61` validation, listing-ID overlap `0`, `seed=42`. Hafif negatif fallback aynı split içindeki sıradaki farklı listing ID'dir; ağır mining MacBook'ta çalıştırılmadı.
- Dense-only evaluator metadata hard filter ve reranker kullanmaz; sahte embedder DI ile test edildi. LoRA CLI adapter-only PEFT kullanır, target modüllerini explicit doğrular ve CUDA yoksa model yüklemeden önce açık hata verir.
- MacBook doğrulaması: `cuda_available=False`, `mps_available=False`; yeni hedefli unit testler `8 passed`, tam paket `112 passed`, skip yok.
- Eğitim MacBook Air'de koşulmaz. Adapter yalnız held-out R@K/MRR artarsa demo adayı olur; artmazsa dürüst negatif deney sonucu raporlanır.

### M7 — Time series (**KAPSAM DIŞI** — sınıf arkadaşı yaptı, 2026-05-30)

- Sentetik kira serisi + LSTM/GRU notebook bu projede **yapılmayacak**; bağımsız/paralel bir ödevdi ve sınıf arkadaşı tamamladı.
- **Etki:** Bu projenin teslimi artık kendi NN'ini içermeli mi? — M7 NN gereksinimini karşılıyordu varsayımı düştü (bkz. M6 + Açık sorular #6).

### M8 — Final report (PENDING)

- Rapor notebook'u + README rewrite + sunum akışı.

---

## Açık sorular / kararlar

1. **`is_furnished` re-scrape gerekli mi?** Cleaner şu an attribute-yoksa-keyword fallback ile çalışıyor (`furnishedStatus` raw'da yok). Mevcut 229 True/1010 None. Gelecek bir scrape'te scraper `furnishedStatus` çekecek (INFO_FIELD_MAP güncel). Re-scrape M3 öncesi yapılırsa is_furnished gerçek structured olur.
2. **`heating_type` enum dışı değerler:** 17 ilanda heating_type=null (Sobalı, VRV, Fueloil, vs). Bunlar enum'a eklenmeli mi yoksa "diger" kategorisi mi açılmalı?
3. **Scrape genişletme:** 303 aktif ilan demo için yeterli mi? — Full M3 + canonical M4 retrieval sonuçlarına göre incremental büyütme kararı verilecek.
4. **Gemini free tier:** 429 sebebiyle shootout'larda kullanılamadı. Billing açma kararı? — şu an Kimi+Gemma yeterli.
5. **DeepSeek negatif kanıt — KARAR (2026-05-31):** DeepSeek başlık+açıklamada açık negatif kanıt varsa kalan null alanı `false` yapabilir; kanıt yoksa `null` bırakır. Kimi için bu geçerli değildir: görselde görünmeyen özellik `false` olmaz.
6. **NN gereksinimi — KARAR (2026-06-01):** Baharat olarak tek NN deneyi yapılacak: BGE-M3 LoRA adapter. Eğitim MacBook Air'de değil, RTX 5070 Ti makinede koşacak. Adapter yalnız held-out synthetic R@K/MRR baseline'ı iyileştirirse demo adayı olacak; aksi halde baseline korunacak.
7. **Qwen3-VL local vision testi — ERTELENDİ (2026-05-30):** `qwen3_vl_local` (qwen3-vl:8b) MacBook Air M4 16GB'ı darlıyor (mouse takılıyor); probe: tek 8-fotolu çağrı **62.5s** + boş `{}` çıktı (format=json olsa da). 16GB çok-fotoğraflı vision'a yetmiyor. Kullanıcının asıl makinesi (ev kasası: 32GB DDR5 + RTX 5070 Ti) ev wifi'si gelince kullanılacak; test oraya **ertelendi**, rapora "future work / dipnot". **Sonuç: vision_model = kimi_k2_6 ile devam (zorunlu).** Yeni canonical 10-ilan Kimi ölçümü `$0.040671`; full 303 koşusu maliyet limitiyle başlatılacak.
8. **`imkanlar` modalite uyuşmazlığı — TARİHSEL NOT (2026-05-29):** Eski gold hata analizi `imkanlar` alanında image+text karışımı olduğunu gösterdi. Canonical akış bu karışımı kaldırır: scraper `İlan Özellikleri` bullet'larını deterministik fact olarak yazar; DeepSeek yalnız başlık+açıklamayı okur; Kimi yalnız görsellerden kalan null boolean alanlara positive-only `true` ekler.

9. **Bağımsız iki-geçişli labeling — DONE (2026-05-31):** `--phase text|vision|combined` eklendi. Text ara çıktısı açıklama ve görsel yollarını koruyor; vision geçişi bu JSONL üzerine yalnız kalan null alanları dolduruyor.

10. **JSON çıktı formatı — KARAR (2026-05-31): `filter_values` DÜZ kalır (150 slug), canvas hiyerarşisine GÖMÜLMEZ.** Gerekçe: labeling (`run_labeling.py`) düz slug→değer okuyup monotonik null-only merge yapıyor; indexing (`composer.to_metadata`) Chroma için yalnızca düz skaler metadata yazabiliyor. Canvas (Ana/İç/Dış/Konum) bir *sunum* meselesi — `spec.group` zaten taşıyor, gerekirse görüntüleyici üretilir. Raw aşamada her dolu slot'un kaynağının `scraper_info`/`scraper_property_feature` olması hata değil: yalnız scraper koştu; DeepSeek/Kimi labeling'de null'ları doldurup kendi damgalarını (`deepseek_description`/`kimi_image`) basar. Dolum monotonik ve null-only (önceden dolu değer ezilmez).

11. **Kimi hız ayarı — KARAR (2026-05-31):** Görsel sayısını kısıtlama. `512px`, varsayılan chunk kapalı (`--vision-chunk-size 0`) ve `5 worker` kullan. Benchmark'ta chunk kapalı 10 worker hızlıydı (`10 ilan: 144.34s`) fakat full batch'te timeout baskısı yarattı. Tekil 20-görselli `19416208` timeout verdiğinde yalnız o ilan `chunk=5`, `1 worker` fallback ile tamamlandı.

12. **Timeout dayanıklılığı — DONE (2026-05-31):** Provider timeout transient retry kapsamına alındı. Paralel batch'te tek ilan hata verirse diğer başarılı row'lar yazılmaya devam eder; batch sonunda başarısız ID'ler açıkça raporlanır ve `--resume` yalnız eksikleri yeniden işler.

13. **MacBook Air eval sınırı — KARAR (2026-06-01):** BGE-M3 + reranker CPU-only çalışıyor (`mps_available=False`). Full `303` sentetik sorgu eval bu cihazda tekrar koşulmayacak. Yerel doğrulama `--known-limit`, `--benchmark-limit` ve sorgu-başı checkpoint ile küçük örneklem; uzun koşu RTX makineye taşınacak.

---

## Bilinen riskler

| Risk | Etki | Mitigation |
|---|---|---|
| Composer yalan rapor verir | Tamamlanmamış işi tamamlanmış sanma | Her milestone'da DOSYADAN doğrula |
| Gemma4 vision Türkçe ev fotoğrafında zayıf çıkabilir | M3 labeling kalitesiz | Gold ile ölç, gerekirse Kimi'ye geç ($25-30 ek maliyet) |
| 303 ilan fine-tune için az olabilir | LoRA underfit/overfit | Synthetic pair generation veya hard-negative mining; pre/post karşılaştırma |
| Time series için gerçek veri yok | Tek snapshot'tan trend yok | Sentetik + scraped ortalamayı calibration anchor |
| Hocaya teslim deadline yaklaşıyor | Zaman baskısı | M5 baseline tamamlandı. **M6'yı tek deney tut:** yalnız BGE-M3 LoRA; eğitim RTX makinede |

---

## Cost tracker (yaklaşık)

| Adım | Harcanan | Tahmini ek | Notlar |
|---|---|---|---|
| M1 4-model shootout | ~$2 | — | Kimi+Gemma+DeepSeek+Gemini (kısmî) |
| M1.5 vision per-image (önceki) | ~$0.11 | — | 3 listing tamamlandı (timeout) |
| M1.5 vision multi-image (2026-05-29) | **$0.36** | — | Kimi, 10 ilan, 161 foto, bakiye farkıyla ölçüldü |
| M3 yeni batch 10 ilan DeepSeek | ~$0.0077 tahmini | — | text-only, 3 worker |
| M3 yeni batch 10 ilan Kimi | ~$0.0407 tahmini | — | 512px, chunk=0, 10 worker |
| M3 full 303 ilan | ölçülmedi | — | kesinti + resume nedeniyle tek wall-clock/maliyet ölçümü yok; çıktı tamamlandı |
| **TOPLAM** | ~$2.55 + önceki ölçümler | ölçülecek | $100 bütçenin altında |

---

## Test durumu

```
tests/test_llm_shootout.py
tests/test_emlakjet_filters.py    # canonical registry
tests/test_gold_helpers.py
tests/test_scraper_description_limit.py
tests/test_scraper_extraction.py
tests/test_cleaner_preserves_fields.py
tests/test_run_labeling.py        # M3 pre-flight + facts_all gate testi dahil
tests/test_composer.py            # M4 (Codex)
tests/test_retriever.py           # M4 (Codex)
tests/test_retrieval_eval.py      # M4 gold-free proxy + checkpoint
tests/test_rag_response.py        # M5 grounded prompt
tests/test_ui_app.py              # M5 Gen3 kartlar + güncel Streamlit API
tests/test_prepare_pairs.py       # M6 deterministik listing-ID split + hafif negatif
tests/test_evaluate_dense.py      # M6 saf dense-only metrik + embedder DI
tests/test_train_bge_m3_lora.py   # M6 CUDA ve explicit target kapıları
```

Komut (macOS/zsh): `source .venv/bin/activate && python3 -m pytest -q` — **112 passed** (2026-06-01; skip yok).

---

## Sıradaki adımlar (2026-06-01 itibariyle)

Tamamlanan: M0 canonical scrape+clean, M1 (+text revize), M1.5 vision, M3 full 303 labeling, M4 baseline index + hafif eval, M5 Gen3 canlı demo. M2 yeni gold rebuild iptal.

| # | İş | Kim | Wall-clock | Bağımlılık | Durum |
|---|---|---|---|---|---|
| Canonical batch | 305 raw → görseller → cleaner → **303 aktif ilan DONE** | Agent | download 106s, cleaner 5s | — | **DONE** |
| 10 ilan göz kontrolü | `labeled_benchmark_deepseek_text_10.jsonl` + `labeled_benchmark_kimi_vision_10.jsonl` incele | Kullanıcı | — | — | **DONE** |
| M3-build | Full 303 ilan: DeepSeek `worker=3`; Kimi `512px`, varsayılan `chunk=0`, `worker=5`; tekil timeout fallback `chunk=5` | Agent | kesintili/resume | kullanıcı göz kontrolü + ücretli full batch onayı | **DONE** |
| Qwen | `qwen3_vl_local` gold visual benchmark | Agent (ev kasası) | ~30-60 dk | ev wifi + 32GB makine | **ERTELENDİ** (16GB MacBook darlıyor; future work) |
| M5-göç | `chat/rag_response.py` + `ui/app.py` Gen3 sözleşmesine göç (facts->filters, has_parking->canonical, slug->label, deterministik match çipleri) | Agent | ~1 saat | — | **DONE** |
| M4-build | Gerçek index build + retriever smoke + checkpoint'li gold-free hafif eval | Agent | — | M3-build | **DONE** |
| M5 | RAG chat + Streamlit UI canlı browser smoke | Agent | — | M4-build | **DONE** |
| M6 | BGE-M3 LoRA held-out deneyi | Agent (RTX 5070 Ti) | ~2-4 saat | M4 baseline | **SIRADAKİ** |
| M8 | Final rapor notebook + README + sunum taslağı | Agent + Sen | ~4 saat | hepsi | — |

**Not:** M7 (time series) tablodan çıkarıldı — kapsam dışı (arkadaş yaptı).

### Acil teslim için scope kesme önerileri
- **M6'yı hafif tut:** yalnız BGE-M3 LoRA; ResNet/ViT ekleme.
- **MacBook'ta ağır iş başlatma:** full eval ve eğitim RTX makineye taşınır.
- **Minimum kalan teslim:** M6 LoRA deneyi + M8 sunum.
