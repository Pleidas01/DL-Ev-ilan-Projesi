# STATUS.md — Mevcut Durum (snapshot)

> Son güncelleme: 2026-05-30, **M3 pre-flight kararları + M4 iskele + scope**: (1) pre-flight kapısı `facts_llm` → **`facts_all`** (tam birleştirilmiş pipeline; `facts_llm` text-only modeli null-vs-false yüzünden haksızca eler, sadece teşhis). (2) `text_model` **kimi_k2_6 → deepseek_v4_pro** (gold A/B: slot 0.815 > 0.792, ~3.5x ucuz, ~5x hızlı; vision_model Kimi kalıyor). (3) **M4 iskele DONE** (Codex: composer + build_chroma + retriever). (4) **M7 (time series) kapsam dışı** — sınıf arkadaşı yaptı; NN gereksinimi yeniden açık (bkz. Açık sorular #6). (5) Gold listing **16/30**. Test **47 passed**.
> Önceki (2026-05-29): vision multi-image refactor (tek çağrı/ilan), Kimi multi-image gold testi geçti (bkz. M1.5).
> Bu dosyayı sıradaki agent her milestone bitince güncellemeli.

---

## TL;DR

- **M0 (scraper + cleaner + yeni şema)** — **DONE** (2026-05-26 13:00 schema refactor sonrası): 1483 ham → **1239** temizlenmiş ilan, **21 facts** top-level (15 structured + 4 hybrid + 2 desc-only) + boş `visual_qualities` object.
- **M1 (slot shootout)** — **DONE** (2026-05-25, 2026-05-30 revize): M1'de kazanan `kimi_k2_6` (quality 0.897) idi; **2026-05-30 gold A/B** ile `text_model` **`deepseek_v4_pro`**'ya çevrildi (aynı koşullu head-to-head: slot 0.815 > kimi 0.792, ~3.5x ucuz, ~5x hızlı). `llm/selected.json`: **text=deepseek_v4_pro, vision=kimi_k2_6**.
- **M2.0 (helper + schema refactor)** — **DONE** (2026-05-26 13:00): heating_type + is_furnished STRUCTURED'a taşındı, `has_aircon` hybrid alan eklendi, `_normalize_blob` Türkçe büyük harf bug'ı fix, visual gold prefilled regen.
- **M2 (manuel gold)** — **AKTİF**: sadeleştirilmiş schema (21 facts + 7 visual). Listing gold **16/30 dolu** (`GOLD_BENCHMARK_LISTING_COUNT=16`). Query gold (30 sorgu) M4'e ertelendi (expected_ids hâlâ 0/30).
- **M1.5 (vision shootout)** — **VISION DONE** (2026-05-29, multi-image refactor): Kimi multi-image (tek çağrı/ilan, tüm foto) **10/10 ilan tamamladı**, accuracy **0.748** (10 ilan); per-image ile ortak 3 ilanda **0.931 ≥ 0.917** → dilution yok, robust, **~3x ucuz ($0.036/ilan)**. Karar: M3 multi-image+Kimi. (Description shootout kısmı hâlâ açık — heating_type haksızlığı düzeltilmeli.)
- **M3 (labeling pipeline)** — **DONE** (iskele + pre-flight): `labeling/run_labeling.py` mevcut (multi-image VLM + text LLM, eşzamanlı çağrı, resume, cost cap, pre-flight gold gate). Pre-flight kapısı **`facts_all`** (eşik facts ≥0.75, visual ≥0.70). **Full 1239-ilan build henüz çalıştırılmadı** (`data/processed/labeled.jsonl` yok; sadece pre-flight smoke çıktıları var).
- **M4 (indexing + retrieval)** — **DONE (iskele, Codex 2026-05-30)**: `indexing/composer.py`, `indexing/build_chroma.py`, `retrieval/retriever.py` + testler. Gerçek index build M3 `labeled.jsonl`'e bağlı (henüz koşulmadı).
- **M5–M8** — PENDING. **M7 kapsam dışı** (arkadaş yaptı). Sıradaki: M3 full labeling build → M4 gerçek index → M5 RAG/Streamlit.

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

### M2 — Manual gold sets (AKTİF — 16/30 dolu)

**2026-05-28 schema sadeleştirme** (kitchen_type çıktı, visual 12→7, imkanlar rename; 30 listing ID korundu, template regen):

- `labeling/gold_listings_manual_todo.jsonl` — **30 satır**, sadeleştirilmiş 21-facts şema:
  - `facts_gold` (21 alan): 15 structured otomatik dolu, 6 user-fill (has_balcony, has_elevator, has_parking, has_aircon, near_metro, near_metrobus). `kitchen_type` kaldırıldı.
  - `visual_gold` (7 alan): balkon_ozellikleri, manzara, mutfak_tipi, banyo_ozellikleri, zemin_tipi, salon_ozellikleri, imkanlar. **16/30 dolduruldu** (`GOLD_BENCHMARK_LISTING_COUNT=16`; benchmark ilk 16 satırı kullanır), kalan prefilled.
  - Doldurma kuralı: gördüğün/emin olduğun → işaretle; negatif kanıtla emin "yok" → false/`[]`; emin değil/görünmüyor → null (benchmark'ta skip).
- `evaluation/gold_queries_manual_todo.jsonl` — **30 sorgu**, `expected_listing_ids=[]`. **M4'e ertelendi** (retriever olmadan test edilemez).

**Toplam manuel iş:** 30 listing × ~19 değerlendirme + 30 query expected_ids = **~3-4 saat**.

Helper'lar (yeni schema'ya göre çalışıyor):
- `python -m labeling.gold_helper --listing <ID>` — listing detayı + 15 structured facts (otomatik dolu) + 7 suggested hybrid facts + suggested visual fields cross-validation + manual TODO listesi
- `python -m evaluation.gold_helper --query "..."` — query için BM25 + hard-filter top-K candidate

### M3 — Labeling pipeline (DONE iskele + pre-flight, full build PENDING)

- `labeling/run_labeling.py` **mevcut**: her ilan için multi-image VLM (Kimi) + text LLM (deepseek_v4_pro) çağırıp merge eder; eşzamanlı çağrı, resume, cost cap, pre-flight gold gate.
- **Pre-flight kapısı: `facts_all`** (eşik facts ≥0.75, visual ≥0.70). `facts_llm` (TEXT_FACT_FIELDS, 6 hybrid/desc alan) sadece teşhis amaçlı raporlanır — text-only model fotoğraftan dolan hybrid alanlarda null üretir, gold false der → null-vs-false yüzünden yapısal olarak 0 alır; kapı olarak kullanılırsa güçlü modeli haksızca eler. (bkz. `test_passes_thresholds_uses_facts_all_not_facts_llm`).
- Output şeması: dataset.jsonl + 21 facts (boş hybrid+desc alanları LLM ile dolar) + visual_qualities (7 alan VLM ile dolar) + enriched_doc (embedding metni) + labeling_metadata.
- **Full 1239-ilan build henüz çalıştırılmadı.** `data/processed/`'te yalnızca pre-flight smoke çıktıları var (`labeled_preflight*.jsonl`); `labeled.jsonl` yok. M4 gerçek index buna bağlı.

### M4 — Indexing + retrieval (DONE iskele, Codex 2026-05-30; gerçek build PENDING)

- `indexing/composer.py`: `embedding_text(record)` enriched_doc'u aynen döner (eksikse hata verir); `to_metadata(record)` facts_gold + visual.aggregated + title'ı **skalar** Chroma metadata'ya düzleştirir (list alanlar deterministik sıralı `|` ile birleşir).
- `indexing/build_chroma.py`: ChromaDB persistent collection builder (CLI).
- `retrieval/retriever.py`: query → slot extract (`selected.json` text_model, read-only) → metadata filter (`slots_to_where`: rooms→room_count `$in`, districts→district `$in`, max_price_tl→price_tl `$lte`, min_size_m2→gross_size_m2 `$gte`, bool alanlar eşitlik) + vektör arama (BGE-M3) + CrossEncoder reranker (bge-reranker-v2-m3). DI ile test edilebilir (collection/embedder/reranker/slot_extractor enjekte).
- Testler: `tests/test_composer.py`, `tests/test_retriever.py` (embedder/reranker/API mocklu). **Gerçek full index build M3 labeled.jsonl hazır olunca koşulacak.**

### M5 — RAG chat + Streamlit (PENDING)

- `chat/rag_response.py`: retrieved listings + soru → LLM ile cevap kompozisyonu.
- `ui/app.py`: chat box + listing cards + 3 demo senaryo.

### M6 — Fine-tune (PENDING — NN gereksinimi için yeniden kritik, bkz. Açık sorular #6)

- `finetune/bge_lora.py`: BGE-M3 LoRA on listing pairs.
- `finetune/image_classifier.py`: ResNet-18 / ViT-small on `visual_gold` labels (3-4 alan: balkon_tipi, manzara, zemin_tipi).
- **Önceki "M6 atlanabilir" notu artık geçersiz**: M7 LSTM/GRU dışarıdan geldiği için hocanın "kendi NN'iniz olsun" gereksinimi bu projede M6 (veya başka bir öz-eğitimli NN) ile karşılanmalı. Kullanıcı kararı bekleniyor.

### M7 — Time series (**KAPSAM DIŞI** — sınıf arkadaşı yaptı, 2026-05-30)

- Sentetik kira serisi + LSTM/GRU notebook bu projede **yapılmayacak**; bağımsız/paralel bir ödevdi ve sınıf arkadaşı tamamladı.
- **Etki:** Bu projenin teslimi artık kendi NN'ini içermeli mi? — M7 NN gereksinimini karşılıyordu varsayımı düştü (bkz. M6 + Açık sorular #6).

### M8 — Final report (PENDING)

- Rapor notebook'u + README rewrite + sunum akışı.

---

## Açık sorular / kararlar

1. **`is_furnished` re-scrape gerekli mi?** Cleaner şu an attribute-yoksa-keyword fallback ile çalışıyor (`furnishedStatus` raw'da yok). Mevcut 229 True/1010 None. Gelecek bir scrape'te scraper `furnishedStatus` çekecek (INFO_FIELD_MAP güncel). Re-scrape M3 öncesi yapılırsa is_furnished gerçek structured olur.
2. **`heating_type` enum dışı değerler:** 17 ilanda heating_type=null (Sobalı, VRV, Fueloil, vs). Bunlar enum'a eklenmeli mi yoksa "diger" kategorisi mi açılmalı?
3. **Scrape genişletme:** 1239 yeterli mi, ek 1500-2000 incremental scrape edilsin mi? — M3 + M4 retrieval sonuçlarına bakılmalı.
4. **Gemini free tier:** 429 sebebiyle shootout'larda kullanılamadı. Billing açma kararı? — şu an Kimi+Gemma yeterli.
5. **null-vs-false artefaktı (M3 pre-flight) — TEŞHİS edildi, KARAR verildi:** Text prompt hybrid alanlarda (has_balcony vb.) "kanıt varsa true, yoksa null" üretir, hiç `false` demez; gold fotoğraftan türetilen bu alanlara `false` yazar. Sonuç: text-only model `facts_llm`'de yapısal olarak ~0 alır. **Karar:** pre-flight kapısı `facts_all`'a bağlandı; `facts_llm` sadece teşhis. İki model de eşit etkilendiği için A/B göreli karşılaştırması adil kaldı. İstenirse text prompt "emin negatifte false" üretecek şekilde tune edilebilir (opsiyonel iyileştirme).
6. **NN gereksinimi — AÇIK (2026-05-30):** Hocanın "kendi eğittiğiniz NN olsun" şartı önceden M7 (LSTM/GRU) ile karşılanıyor sayılıyordu. M7 artık dışarıdan (arkadaş) geldiği için bu proje **kendi öz-eğitimli NN'ini içermeli**: ya M6 BGE-M3 LoRA, ya ResNet/ViT görsel sınıflandırıcı (M6 image_classifier). Hangi NN, ne zaman, hangi gold ile? — kullanıcı kararı gerekiyor. Sessizce atlama.
7. **Qwen3-VL local vision testi — ERTELENDİ (2026-05-30):** `qwen3_vl_local` (qwen3-vl:8b) MacBook Air M4 16GB'ı darlıyor (mouse takılıyor); probe: tek 8-fotolu çağrı **62.5s** + boş `{}` çıktı (format=json olsa da). 16GB çok-fotoğraflı vision'a yetmiyor. Kullanıcının asıl makinesi (ev kasası: 32GB DDR5 + RTX 5070 Ti) ev wifi'si gelince kullanılacak; test oraya **ertelendi**, rapora "future work / dipnot". **Sonuç: vision_model = kimi_k2_6 ile devam (zorunlu).** 500 ilan ≈ ~$18, 1239 ilan ≈ ~$45 (bütçe $100, sorun yok).
8. **`imkanlar` modalite uyuşmazlığı — KARAR (2026-05-29):** Vision multi-image hata analizi (kullanıcı teyitli) gösterdi ki gold `imkanlar` **image+text birlikte** dolduruldu; site özellikleri (guvenlik_kabini, kapali_otopark, cocuk_parki, spor_alani) genelde iç-mekan fotoğraflarında değil, ilan **açıklaması/property_features metninde**. Kanıt: 19382188 gold `[guvenlik_kabini]` ama 7 fotonun hepsi iç mekan + açıklamada amenity yok + `in_gated_complex=False` (muhtemel gold hatası); 19392370 & 19376408 amenity'leri doğrudan description kelimelerinden (site/güvenlik/havuz/otopark/fitness/çocuk). Ayrıca gold bazı görünür özellikleri eksik etiketlemiş (19382188 banyoda_pencere foto 05'te net, model doğru bildi, gold atlamış). **Karar:** M3'te `imkanlar` vision'dan DEĞİL, description+property_features'tan LLM ile (gerekirse vision union) etiketlenecek; vision-prompt'u imkanlar için zorlama (halüsinasyon riski). Vision gerçek doğruluğu imkanlar hariç **0.789** (mutfak 0.900, banyo 0.738). Gold 10→30 genişletmesinde 19382188 gözden geçirilsin.

---

## Bilinen riskler

| Risk | Etki | Mitigation |
|---|---|---|
| Composer yalan rapor verir | Tamamlanmamış işi tamamlanmış sanma | Her milestone'da DOSYADAN doğrula |
| Gemma4 vision Türkçe ev fotoğrafında zayıf çıkabilir | M3 labeling kalitesiz | Gold ile ölç, gerekirse Kimi'ye geç ($25-30 ek maliyet) |
| 1239 ilan fine-tune için az olabilir | LoRA underfit/overfit | Synthetic pair generation veya hard-negative mining; pre/post karşılaştırma |
| Time series için gerçek veri yok | Tek snapshot'tan trend yok | Sentetik + scraped ortalamayı calibration anchor |
| Hocaya teslim deadline yaklaşıyor | Zaman baskısı | M5 minimal yap. **M6'yı atlama** — M7 dışarı çıktığı için NN gereksinimi M6'ya bağlı (bkz. Açık sorular #6); en hafif öz-eğitimli NN'i (BGE-M3 LoRA veya ResNet sınıflandırıcı) koru |

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
tests/test_run_labeling.py        # M3 pre-flight + facts_all gate testi dahil
tests/test_composer.py            # M4 (Codex)
tests/test_retriever.py           # M4 (Codex)
```

Komut (macOS/zsh): `source .venv/bin/activate && python3 -m pytest -q` — **47 passed** (2026-05-30; M3 pre-flight + M4 iskele testleri eklendi).

---

## Sıradaki adımlar (2026-05-30 itibariyle)

Tamamlanan: M0, M1 (+text revize), M1.5 vision, M2 gold 16/30, M3 iskele+pre-flight, M4 iskele.

| # | İş | Kim | Wall-clock | Bağımlılık | Durum |
|---|---|---|---|---|---|
| M3-build | Full labeling build (`run_labeling.py` → ilanlar → `labeled.jsonl`), **vision=Kimi** | Agent | ~1-2 saat (eşzamanlı, ağ-bound) | — | **sıradaki** |
| Qwen | `qwen3_vl_local` gold visual benchmark | Agent (ev kasası) | ~30-60 dk | ev wifi + 32GB makine | **ERTELENDİ** (16GB MacBook darlıyor; future work) |
| M5-iskele | `chat/rag_response.py` + `ui/app.py` (Streamlit) — Retriever'ı tüketir | Codex | ~1 saat | M4 retriever (var) | Codex'e verildi (paralel, çakışmasız) |
| M4-build | Gerçek index build (composer → BGE-M3 → Chroma) + retriever smoke + 30 query gold | Agent + Sen | ~2.5 saat | M3-build | — |
| M5 | RAG chat + Streamlit UI + 3 demo | Agent | ~3 saat | M4-build | — |
| M6 | Öz-eğitimli NN (BGE-M3 LoRA **veya** ResNet/ViT görsel sınıflandırıcı) | Agent | ~6 saat (training dahil) | M3-build | **NN gereksinimi için kritik (Açık sorular #6)** |
| M8 | Final rapor notebook + README + sunum taslağı | Agent + Sen | ~4 saat | hepsi | — |

**Not:** M7 (time series) tablodan çıkarıldı — kapsam dışı (arkadaş yaptı).

### Acil teslim için scope kesme önerileri
- **M5 UI'ı minimal yap** (chat box + 3 demo senaryo, kart süslemesi yok). ~2 saat tasarruf.
- **M6'yı atlama ama hafiflet:** En küçük öz-eğitimli NN'i koru (örn. sadece BGE-M3 LoRA *veya* sadece ResNet görsel sınıflandırıcı, ikisi birden değil). M7 dışarı çıktığı için NN gereksinimini bu karşılayacak — tamamen atlanırsa hocanın "kendi NN'iniz" şartı boşta kalır.
- **Minimum viable demo:** M3-build + M4-build + M5(minimal) + M6(hafif) + M8.
