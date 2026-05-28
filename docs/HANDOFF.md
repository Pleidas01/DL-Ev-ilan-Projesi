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

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pytest -q   # 27 passed beklenir

# .env kontrolü — Cursor Read tool'una GÜVENME, gerçek key dolulğu için:
.\.venv\Scripts\python.exe -c "from dotenv import dotenv_values; v=dotenv_values('.env'); [print(f'{k}: {\"FILLED\" if val else \"EMPTY\"}') for k,val in v.items() if 'API_KEY' in k]"
# Beklenen: DEEPSEEK_API_KEY=FILLED, MOONSHOT_API_KEY=FILLED, GEMINI_API_KEY=FILLED
# OPENROUTER_API_KEY EMPTY OK (GLM-4.6 skip ediliyor)

# Dataset sağlık kontrolü
.\.venv\Scripts\python.exe -c "import json; rows=[json.loads(l) for l in open('data/processed/dataset.jsonl', encoding='utf-8')]; print(f'total={len(rows)}, has_facts={sum(1 for r in rows if r.get(\"city\"))}, has_heating={sum(1 for r in rows if r.get(\"heating_type\"))}, has_in_gated={sum(1 for r in rows if r.get(\"in_gated_complex\") is not None)}')"
# Beklenen: total=1239, has_facts=1239, has_heating~1222, has_in_gated=1239

# Gold template doğrulama (YENİ ŞEMA)
.\.venv\Scripts\python.exe -c "import json; rows=[json.loads(l) for l in open('labeling/gold_listings_manual_todo.jsonl', encoding='utf-8')]; print(f'total={len(rows)}, has_facts_gold={sum(1 for r in rows if \"facts_gold\" in r)}, has_visual_gold={sum(1 for r in rows if \"visual_gold\" in r)}, has_old_text_gold={sum(1 for r in rows if \"text_gold\" in r)}')"
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
M1 (slot shootout)    ✅ DONE — kimi_k2_6 winner
M2.0 (schema refactor)✅ DONE — schema sadeleştirme: kitchen_type çıktı, visual 12→7, imkanlar rename
M2 (manuel gold)      🔄 AKTİF — listing gold 10/30 dolu (query gold M4'e ertelendi)
M1.5 (vis+desc shoot) ✅ HAZIR — 10/30 gold yeterli, shootout kodu güncel şemada
M3-M8                 ⏳ PENDING
```

**Acil sıradaki iş:** M3 — `labeling/run_labeling.py` yaz (multi-image + Kimi). Multi-image refactor + gold testi **BİTTİ** (2026-05-29): Kimi multi-image 10/10 ilan tamamladı, ortak 3 ilanda 0.931 ≥ per-image 0.917 (dilution yok), $0.036/ilan (~3x ucuz), robust. `complete_vision_json` artık `image_paths: list[str]`; `shootout_vision` tek çağrı/ilan + listing-başına resilience. **M3 kritik nokta:** seri run 1239 ilan ≈ 39 saat → `run_labeling.py` **eşzamanlı** olmalı (20-30 worker, Moonshot rate limit'e göre) → ~1-2 saat. Vision karar dosyası: `llm/shootout_vision_multi_rows.json`.

---

## 2. KULLANICI — Manuel gold doldurma (AKTİF — 10/30 dolu)

**Workflow:** Kullanıcı fotoğraflara (`data/images/<ID>/`) bakar, her ilan için özellikleri seçer, supervisor'a `ID + özellik listesi` verir; supervisor JSON'a çevirip `labeling/gold_listings_manual_todo.jsonl`'a yazar.

### 2a. Tek listing inceleme (opsiyonel)

```powershell
.\.venv\Scripts\python.exe -m labeling.gold_helper --listing <ID>
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

```powershell
.\.venv\Scripts\python.exe -m evaluation.gold_helper --query "Kadıköy'de 45 bin TL altı eşyalı 2+1"
```

Çıktı: top-10 candidate ID + title + price + URL. Kullanıcı en alakalı 1-3 ID'yi seçer.

Kullanıcı:
1. `evaluation/gold_queries_manual_todo.jsonl`'da satırı düzenle, `expected_listing_ids` doldur.

### 2d. Doluluk kontrolü

```powershell
.\.venv\Scripts\python.exe -c "import json; from llm.gold_benchmark import build_prefilled_visual_gold, build_prefilled_hybrid_facts; rows=[json.loads(l) for l in open('labeling/gold_listings_manual_todo.jsonl', encoding='utf-8')]; ph=build_prefilled_hybrid_facts(); pv=build_prefilled_visual_gold(); touched_facts=sum(1 for r in rows if any(r['facts_gold'].get(k)!=ph[k] for k in ph)); touched_visual=sum(1 for r in rows if r['visual_gold']!=pv); print(f'facts düzenlenmiş: {touched_facts}/30, visual düzenlenmiş: {touched_visual}/30')"
```

> Bir alan "düzenlenmiş" sayılır eğer prefilled string'den farklıysa (tek değer, null, boolean, veya kısaltılmış array). Hâlâ `"true | false"` veya `"amerikan_acik | kapali_ayri | ..."` ise dokunulmamış sayılır.

**Tahmini süre:** Listing başına ~6-8 dakika × 30 = ~3-4 saat. Query'ler için ~30 dk.

---

## 3. AGENT — Vision + Description shootout (M2 dolduktan SONRA)

> **Vision kısmı BİTTİ (2026-05-29):** multi-image refactor + Kimi gold testi tamamlandı (bkz. STATUS.md M1.5). Vision modeli = **kimi_k2_6 multi-image**. Aşağıdaki vision shootout komutu artık `--max-photos` opsiyonu alır (varsayılan: tüm foto) ve sonucu `llm/shootout_vision_multi_rows.json`'a yazılır. **Description shootout hâlâ açık** (heating_type/is_furnished structured haksızlığı düzeltilmeli).

Kullanıcı M2'yi bitirdiğini söyleyene kadar başlama. Gold dolduğunu doğrulamak için yukarıdaki 2d (doluluk kontrolü) komutu. En az 10/30 listing dolu olmalı.

Sonra:

```powershell
# Description (text) shootout — 3 model x 10 listing
.\.venv\Scripts\python.exe -m llm.shootout_description `
  --models deepseek_v4_flash kimi_k2_6 gemma_4_local `
  --out llm\shootout_description_rows.json

# Vision shootout — 2 multimodal model x 10 listing x ~5 foto
.\.venv\Scripts\python.exe -m llm.shootout_vision `
  --models gemma_4_local kimi_k2_6 `
  --out llm\shootout_vision_rows.json
```

> Gemini hariç tutuldu (429 free-tier sorunu). Eğer billing açılırsa eklenebilir.

Sonra `llm/selected.json`'u güncelle:
- `text_model`: description shootout kazananı (slot benchmark ile cross-check; aynıysa süper)
- `vision_model`: vision shootout kazananı

`llm/shootout_report.md`'ye yeni bölüm ekle: per-field accuracy, final seçim gerekçesi.

**Bütçe alarmı:** Toplam ~$1-2. $5'i geçerse durdur, kullanıcıya bildir.

---

## 4. M3 — Labeling pipeline (M1.5 bitince başla)

Önkoşul: `llm/selected.json` final, gold 10+/30 dolu.

İskele (henüz yazılmadı):

```
labeling/run_labeling.py
  --input data/processed/dataset.jsonl
  --output data/processed/labeled.jsonl
  --batch-size 20
  --resume                    # idempotent, ID bazlı
  --max-cost-usd 50           # cost cap, aşılınca dur
  --vision-model <selected>
  --text-model <selected>
```

Output şeması (her satır):
```json
{
  "id": "19364542",
  "title": "...", "url": "...",
  "facts_gold": { ...21 alan, hybrid+desc-only LLM ile dolar... },
  "visual_qualities": {
    "per_image": [{"path": "...", "fields": {...}, "confidence": 0.85}],
    "aggregated": { ...12 alan, multi-select için union... }
  },
  "enriched_doc": "<embedding için TR composed string>"
}
```

**Pre-flight test:** Önce gold'daki 10 listing'i etiketle, `facts_gold` + `visual_gold` ile karşılaştır, accuracy raporla. Threshold:
- facts (8 alan): ≥ 0.75
- visual (12 alan, Jaccard ortalaması): ≥ 0.70

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
7. **Windows encoding gotcha:** Console cp1254, dosyalar UTF-8. Print'lerde özel karakter (`→`, `…`) crash yaratabilir. Düzgün fix: `sys.stdout.reconfigure(encoding='utf-8')`.
8. **`archive/`** klasörü yedek, dokunma. `pre_scraper_fix/` ve `pre_schema_refactor/` farklı yedek noktaları.
9. **Composer 2.5 yalan söyler.** Her milestone'da dosyadan doğrula (yukarıda detay).
10. **Read tool stale cache:** Cursor Read tool bazen dosya yeni yazıldıktan sonra eski versiyonu döner. Şüphe varsa PowerShell `Get-Content` ile cross-check.

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
