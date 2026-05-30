# HANDOFF.md — Sıradaki agent için somut talimatlar

> Bu dosya "şu an ne yapılacak"ı söyler. Bağlam için önce `PROJECT.md`, durum için `STATUS.md` oku.
> Dosya hiyerarşisi: PROJECT (neden) → STATUS (ne durumda) → HANDOFF (şimdi ne yap).

---

## 0. Cold start — yeni session'a katıldıysan

**Sırayla yap:**

1. `docs/PROJECT.md` oku (vizyon, mimari, şema)
2. `docs/STATUS.md` oku (milestone durumu, sağlık metrikleri)
3. Bu dosyayı (`HANDOFF.md`) oku (somut sıradaki iş)
4. Aşağıdaki "İlk 30 saniye sağlık kontrolü"nü koş

### İlk 30 saniye sağlık kontrolü

```bash
# macOS / zsh. (Eski PowerShell komutları 2026-05-30'da bash'e çevrildi.)
source .venv/bin/activate
python3 -m pytest -q   # 47 passed beklenir

# .env kontrolü — Read tool'una GÜVENME, gerçek key doluluğu için:
python3 -c "from dotenv import dotenv_values; v=dotenv_values('.env'); [print(f'{k}: {\"FILLED\" if val else \"EMPTY\"}') for k,val in v.items() if 'API_KEY' in k]"
# Beklenen: DEEPSEEK_API_KEY=FILLED, MOONSHOT_API_KEY=FILLED, GEMINI_API_KEY=FILLED
# OPENROUTER_API_KEY EMPTY OK (GLM-4.6 skip ediliyor)

# Dataset sağlık kontrolü
python3 -c "import json; rows=[json.loads(l) for l in open('data/processed/dataset.jsonl', encoding='utf-8')]; print(f'total={len(rows)}, has_facts={sum(1 for r in rows if r.get(\"city\"))}, has_heating={sum(1 for r in rows if r.get(\"heating_type\"))}, has_in_gated={sum(1 for r in rows if r.get(\"in_gated_complex\") is not None)}')"
# Beklenen: total=1239, has_facts=1239, has_heating~1222, has_in_gated=1239

# Gold template doğrulama (YENİ ŞEMA)
python3 -c "import json; rows=[json.loads(l) for l in open('labeling/gold_listings_manual_todo.jsonl', encoding='utf-8')]; print(f'total={len(rows)}, has_facts_gold={sum(1 for r in rows if \"facts_gold\" in r)}, has_visual_gold={sum(1 for r in rows if \"visual_gold\" in r)}, has_old_text_gold={sum(1 for r in rows if \"text_gold\" in r)}')"
# Beklenen: total=30, has_facts_gold=30, has_visual_gold=30, has_old_text_gold=0
```

**Test kırmızıysa, key EMPTY ise, dataset eksikse, gold eski şemadaysa: durup kullanıcıya sor.**

### ⚠️ Composer güvenilirlik uyarısı

Bu projede Cursor Composer 2.5 **4 ayrı kerede yalan rapor verdi**:
1. Heating value `null` dedi, gerçekte `"dogalgaz"` idi
2. Vision shootout sonucu fake satır
3. "30 satır gold regen" dedi, dosyaya yazmadı (hâlâ eski şemaydı)
4. "regen done, 30 satır" dedi, dosya hâlâ 42 satır eski şemadaydı

**Sıradaki agent için kural:** Bir agent "X tamamlandı" derse, **DOSYAYI KENDİN OKU** ve doğrula. Sadece raporuna güvenme. Composer artık projeden çıkarıldı, ana ajan Claude (sen).

---

## 1. Mevcut durum özeti

```
M0 (scrape + clean)   ✅ DONE — 1239 ilan, 21 facts şeması (15 structured + 4 hybrid + 2 desc-only)
M1 (slot shootout)    ✅ DONE — text=deepseek_v4_pro (A/B revize), vision=kimi_k2_6
M2.0 (schema refactor)✅ DONE — schema sadeleştirme: kitchen_type çıktı, visual 12→7, imkanlar rename
M2 (manuel gold)      🔄 AKTİF — listing gold 16/30 dolu (query gold 0/30, M4'e ertelendi)
M1.5 (vision shoot)   ✅ DONE — Kimi multi-image winner; description shootout açık
M3 (labeling)         ✅ DONE iskele+pre-flight — run_labeling.py var; full build PENDING
M4 (indexing+retr.)   ✅ DONE iskele (Codex) — composer/build_chroma/retriever; gerçek build PENDING
M5, M6, M8            ⏳ PENDING        M7 (time series) ⛔ KAPSAM DIŞI (arkadaş yaptı)
```

**Acil sıradaki iş:** `docs/superpowers/plans/2026-05-30-emlakjet-filter-enrichment.md` planı **Task 2 revizyonu**. Task 1 authoritative tam listeye göre düzeltildi: kategori zinciri, `price_amount` / `price_currency`, `Saten Boya` eklendi; canonical registry `27` structured + `124` checkbox filtre içeriyor. Şimdi scraper ve cleaner canonical kategori ve çoklu para birimi alanlarını deterministic doldurmalı; mevcut top-level `price_tl` yalnız TL ilanlar için compatibility output olarak kalmalı. Full re-scrape ve ücretli API labeling kullanıcı checkpoint onayı olmadan başlatılmayacak.

> **Karar verilebilir önce-işler (opsiyonel/paralel):** (1) `qwen3_vl_local` gold visual benchmark — yerel/ücretsiz vision alternatifi (Kimi ≥0.70 mu) — M3 kararını etkilemez. (2) NN gereksinimi (M6) — M7 dışarı çıktı, hocanın "kendi NN'iniz" şartı yeniden açık (STATUS Açık sorular #6). İkisi de kullanıcı kararı bekler.

---

## 2. KULLANICI — Manuel gold doldurma (AKTİF — 16/30 dolu)

**Workflow:** Kullanıcı fotoğraflara (`data/images/<ID>/`) bakar, her ilan için özellikleri seçer, supervisor'a `ID + özellik listesi` verir; supervisor JSON'a çevirip `labeling/gold_listings_manual_todo.jsonl`'a yazar.

### 2a. Tek listing inceleme (opsiyonel)

```bash
python3 -m labeling.gold_helper --listing <ID>
```

Çıktı: 15 structured facts (otomatik dolu) + suggested hybrid + suggested visual + manuel TODO listesi.

**Doldurma kuralı (gold = ground truth):**
- Gördüğün / emin olduğun → işaretle
- Negatif kanıtla **emin** "yok" → `false` (bool) veya `[]` (multi-select)
- Olabilir ama kanıt yok / görünmüyor → `null` (benchmark'ta skip edilir)

**`facts_gold` user-fill (6 bool):** `has_balcony`, `has_elevator`, `has_parking`, `has_aircon`, `near_metro`, `near_metrobus`. (`kitchen_type` kaldırıldı; structured 15 alan otomatik dolu.)

### 2b. Visual gold doldurma rehberi

Her listing'in `data/images/<listing_id>/` klasöründeki fotoğraflara bak. **7 alan** (single-select: mutfak_tipi, zemin_tipi; diğerleri multi-select):

| Alan | Tip | Değerler |
|---|---|---|
| balkon_ozellikleri | multi | cam_balkon, acik_balkon, fransiz_balkon, cikma_balkon, teras |
| manzara | multi | deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari |
| mutfak_tipi | single | amerikan_acik \| kapali_ayri |
| banyo_ozellikleri | multi | dusakabin, kuvet, jakuzi, banyoda_pencere, birden_fazla_banyo |
| zemin_tipi | single | parke \| laminat \| seramik \| granit \| mermer \| hali \| karma |
| salon_ozellikleri | multi | somine, nis, acik_plan_genis, ayri_yemek_alani |
| imkanlar | multi | havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani |

**Multi-select** alanlar JSON array, **single** alanlar string. `imkanlar` site dışı binalarda da geçerli (güvenlik/otopark site olmadan da olabilir).

### 2c. Query için doğru ilan ID'leri bulma

```bash
python3 -m evaluation.gold_helper --query "Kadıköy'de 45 bin TL altı eşyalı 2+1"
```

Çıktı: top-10 candidate ID + title + price + URL. Kullanıcı en alakalı 1-3 ID'yi seçer.

Kullanıcı:
1. `evaluation/gold_queries_manual_todo.jsonl`'da satırı düzenle, `expected_listing_ids` doldur.

### 2d. Doluluk kontrolü

```bash
python3 -c "import json; from llm.gold_benchmark import build_prefilled_visual_gold, build_prefilled_hybrid_facts; rows=[json.loads(l) for l in open('labeling/gold_listings_manual_todo.jsonl', encoding='utf-8')]; ph=build_prefilled_hybrid_facts(); pv=build_prefilled_visual_gold(); touched_facts=sum(1 for r in rows if any(r['facts_gold'].get(k)!=ph[k] for k in ph)); touched_visual=sum(1 for r in rows if r['visual_gold']!=pv); print(f'facts düzenlenmiş: {touched_facts}/30, visual düzenlenmiş: {touched_visual}/30')"
```

> Bir alan "düzenlenmiş" sayılır eğer prefilled string'den farklıysa (tek değer, null, boolean, veya kısaltılmış array). Hâlâ `"true | false"` veya `"amerikan_acik | kapali_ayri | ..."` ise dokunulmamış sayılır.

**Tahmini süre:** Listing başına ~6-8 dakika × 30 = ~3-4 saat. Query'ler için ~30 dk.

---

## 3. AGENT — Vision + Description shootout (M2 dolduktan SONRA)

> **Vision kısmı BİTTİ (2026-05-29):** multi-image refactor + Kimi gold testi tamamlandı (bkz. STATUS.md M1.5). Vision modeli = **kimi_k2_6 multi-image**. Aşağıdaki vision shootout komutu artık `--max-photos` opsiyonu alır (varsayılan: tüm foto) ve sonucu `llm/shootout_vision_multi_rows.json`'a yazılır.
> **Text/description kararı da BİTTİ (2026-05-30):** `text_model = deepseek_v4_pro` — ayrı bir description shootout yerine **gold A/B** ile seçildi (slot 0.815 > kimi 0.792, ~3.5x ucuz, ~5x hızlı; STATUS M1 revize). Aşağıdaki description shootout komutu yalnızca yeniden-doğrulamak istersen referans; `selected.json` zaten güncel.

Kullanıcı M2'yi bitirdiğini söyleyene kadar başlama. Gold dolduğunu doğrulamak için yukarıdaki 2d (doluluk kontrolü) komutu. En az 10/30 listing dolu olmalı.

Sonra:

```bash
# Description (text) shootout — 3 model x 10 listing
python3 -m llm.shootout_description \
  --models deepseek_v4_flash kimi_k2_6 gemma_4_local \
  --out llm/shootout_description_rows.json

# Vision shootout — 2 multimodal model x 10 listing x ~5 foto
python3 -m llm.shootout_vision \
  --models gemma_4_local kimi_k2_6 \
  --out llm/shootout_vision_rows.json
```

> Gemini hariç tutuldu (429 free-tier sorunu). Eğer billing açılırsa eklenebilir.

Sonra `llm/selected.json`'u güncelle:
- `text_model`: description shootout kazananı (slot benchmark ile cross-check; aynıysa süper)
- `vision_model`: vision shootout kazananı

`llm/shootout_report.md`'ye yeni bölüm ekle: per-field accuracy, final seçim gerekçesi.

**Bütçe alarmı:** Toplam ~$1-2. $5'i geçerse durdur, kullanıcıya bildir.

---

## 4. M3 — Labeling pipeline (iskele + pre-flight DONE; sıradaki = full build)

`labeling/run_labeling.py` **yazıldı ve pre-flight smoke koşuldu** (`data/processed/labeled_preflight*.jsonl`). Önkoşullar sağlandı: `llm/selected.json` final (text=deepseek_v4_pro, vision=kimi_k2_6), gold 16/30 dolu. **Kalan: full 1239-ilan build** → `data/processed/labeled.jsonl` (henüz yok).

CLI (gerçek argümanlar için `python3 -m labeling.run_labeling --help`):

```
labeling/run_labeling.py
  --input data/processed/dataset.jsonl
  --output data/processed/labeled.jsonl
  --resume                    # idempotent, ID bazlı
  --max-cost-usd <cap>        # cost cap, aşılınca dur
  # vision/text model llm/selected.json'dan okunur
```

Output şeması (her satır):
```json
{
  "id": "19364542",
  "title": "...", "url": "...",
  "facts_gold": { ...21 alan, hybrid+desc-only LLM ile dolar... },
  "visual_qualities": {
    "per_image": [{"path": "...", "fields": {...}, "confidence": 0.85}],
    "aggregated": { ...6 alan, multi-select için union... }
  },
  "enriched_doc": "<embedding için TR composed string>"
}
```

**Pre-flight test (kodda mevcut):** Gold'daki ilk 16 listing'i etiketle, `facts_gold` + `visual_gold` ile karşılaştır, accuracy raporla. Kapı:
- **`facts_all` (tam birleştirilmiş pipeline, 21 alan): ≥ 0.75** — gate budur.
- visual (Jaccard ortalaması): ≥ 0.70
- `facts_llm` (TEXT_FACT_FIELDS, 6 hybrid/desc alan): **sadece teşhis, kapı DEĞİL.** Text-only model fotoğraftan dolan alanlarda null üretir, gold false der → null-vs-false yüzünden yapısal 0; kapı olsa güçlü modeli haksız eler (bkz. `tests/test_run_labeling.py::test_passes_thresholds_uses_facts_all_not_facts_llm`).

> **`imkanlar` vision'dan ÇIKARMA — text ile etiketle (karar 2026-05-29, STATUS Açık Soru #5).** Vision hata analizi: gold `imkanlar` kullanıcı tarafından image+text birlikte dolduruldu; site özellikleri (guvenlik_kabini, kapali_otopark, cocuk_parki, spor_alani) iç-mekan fotolarında yok, ilan açıklaması/property_features metninde. M3'te `imkanlar`'ı description+property_features'tan LLM ile çıkar (gerekirse vision union); vision-prompt'u imkanlar için zorlama → halüsinasyon. Vision'ın gerçek görsel alan doğruluğu imkanlar hariç **0.789** (mutfak 0.900, banyo 0.738). İkincil vision-tunable zayıflık (küçük-n, M3 pre-flight gold=30'da doğrula): `fransiz_balkon` 0/2 (model acik_balkon/null diyor), `manzara` hafif aşırı-tahmin.

Tutmazsa M3 batch'i koşma; prompt'u tune et veya supervisor'a dön.

**Confidence:** M3'te eklensin (önceki kararda ertelenmişti). İki opsiyon:
- VLM'i K=3 kez çağırıp agreement ratio
- VLM'in self-reported confidence

İkisini de dene, hangisi daha kalibre çıkarsa onu kullan. `visual_qualities.per_image[i].confidence` < 0.7 olanlar `aggregated`'a girmesin.

---

## 5. M4-M8 — Detaylı plan

PROJECT.md'nin section 4 (mimari) ve section 10 (milestone tablosu) ile STATUS.md'deki kalan iş tahminine bak.

---

## 6. Sık karşılaşılan tuzaklar

1. **`temperature=0` Moonshot'ı çöküyor.** `llm/clients.py:_openai_chat_temperature` Kimi için 1.0 dönüyor. Yeni provider eklersen güncelle.
2. **Gemini SDK'sı `google-genai`** (yeni), `google-generativeai` (eski) DEĞİL.
3. **Türkçe normalize:** `gold_benchmark.normalize_gold_value` lower+strip yapar, `gold_helper._normalize` Türkçe karakterleri ASCII'ye map'ler. İki yerde de tutarlı kal.
4. **`OLLAMA_HOST` env key check'i** atlanıyor (`missing_environment` içinde). Gemma local için Ollama service'in çalıştığını ayrıca kontrol et (`ollama list`).
5. **Image data URL** Gemini için bytes, OpenAI-compat için base64 data URL. `complete_vision_json` ikisini de hallediyor.
6. **`data/eval/manual_queries.jsonl` ESKİ** dosya, kullanma. Yeni `evaluation/gold_queries_manual_todo.jsonl` kullan.
7. **Ortam: macOS / zsh (2026-05-30'dan beri).** Konsol UTF-8 native; eski Windows cp1254 `→`/`…` crash sorunu artık geçerli değil. Venv: `source .venv/bin/activate`, yorumlayıcı `python3`. Bu dosyadaki eski PowerShell komutları bash'e çevrildi.
8. **`archive/`** klasörü yedek, dokunma. `pre_scraper_fix/` ve `pre_schema_refactor/` farklı yedek noktaları.
9. **Geçmiş ajan güvenilirlik dersi:** Cursor Composer 2.5 bu projede 4 kez yalan rapor verdi (yukarıda detay). Artık ana ajan Claude Code; yine de **bir ajan "X tamamlandı" derse dosyadan doğrula.**
10. **`data/processed/` smoke kalıntıları:** `labeled_preflight*.jsonl` + boş `*_stdout/stderr.log` dosyaları M3 pre-flight denemelerinden kaldı; `labeled.jsonl` (gerçek build) henüz yok. Temizlik kullanıcı onayıyla yapılır (silme = geri alınamaz, bkz. §7).

---

## 7. Acil durum protokolü

Bir agent şunlardan birini yaparsa **DUR ve kullanıcıya sor**:

- Cost cap aşımı (>$10 tek koşuda)
- `data/` altında geri alınamayan silme
- `git push` (kullanıcı izin vermedi)
- LLM çıktısı sürekli boş/None geliyor (model down olabilir)
- Test sayısı azalıyor (regression)
- Üst üste 3 prompt tune denemesi accuracy threshold'u tutmuyor
- Bir önceki agent raporuyla dosya içeriği uyuşmuyor → dosyaya güven, rapora güvenme
- Kullanıcı token bütçesi aşımı endişesi ifade ediyor

---

## 8. Bu dosyayı güncelleme kuralı

Sıradaki agent her milestone bitince:
1. `STATUS.md`'deki ilgili milestone'u "DONE" yap, sonuçları yaz.
2. `HANDOFF.md`'nin "Acil sıradaki iş" bölümünü bir sonraki adıma çevir.
3. Yeni karar/sorun varsa `STATUS.md` "Açık sorular" bölümüne ekle.
4. `PROJECT.md`'yi DEĞİŞTİRME (vision/şema değişmedikçe).
