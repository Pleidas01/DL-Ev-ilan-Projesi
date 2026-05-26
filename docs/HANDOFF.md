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
.\.venv\Scripts\python.exe -m pytest -q   # 25 passed beklenir

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
M0 (scrape + clean)   ✅ DONE — 1239 ilan, 22 facts şeması (15 structured + 5 hybrid + 2 desc-only)
M1 (slot shootout)    ✅ DONE — kimi_k2_6 winner
M2.0 (schema refactor)✅ DONE — heating_type+is_furnished STRUCTURED'a, has_aircon hybrid eklendi, Türkçe İ bug fix, visual prefilled
M2 (manuel gold)      🔄 AKTİF — kullanıcı doldurmaya başlayabilir
M1.5 (vis+desc shoot) ⏳ BLOKE — M2 bekliyor
M3-M8                 ⏳ PENDING
```

**Acil sıradaki iş:** Kullanıcı M2 manuel gold doldurmaya başlasın. Helper'lar hazır, dataset (1239) hazır, template (30 satır prefilled visual ile) hazır.

---

## 2. KULLANICI — Manuel gold doldurma (HAZIR — başlayabilirsin)

Kullanıcı bu workflow ile dolduracak:

### 2a. Tek listing inceleme

```powershell
.\.venv\Scripts\python.exe -m labeling.gold_helper --listing 19364542
```

Çıktı: 15 structured facts (otomatik dolu) + 7 suggested hybrid + suggested visual fields + manuel doldurulacak alan listesi.

Kullanıcı:
1. Önerilen değerleri doğrula (özellikle structured'lar — scraper otomatik doldurdu)
2. `labeling/gold_listings_manual_todo.jsonl` içindeki ilgili satırı düzenle. **Tüm user-fill alanlar prefilled string/array** — syntax'ı tek tek hatırlamaya gerek yok, sadece olmayanları sil:

   **`facts_gold` (7 hybrid alan — hepsi `"|"`li string prefilled):**
   - `kitchen_type`: `"amerikan_acik | kapali_ayri | yari_acik"` → tek değer bırak (örn. `"amerikan_acik"`) veya `null`
   - 6 bool alan (`has_balcony`, `has_elevator`, `has_parking`, `has_aircon`, `near_metro`, `near_metrobus`): `"true | false"` → tek değer bırak. **Quote'lar opsiyonel** — hem `"true"` hem `true` çalışır (normalize_gold_value otomatik bool'a çeviriyor). Bilinmiyorsa `null`.

   **`visual_gold` (12 alan prefilled):**
   - Multi-select (manzara, mutfak_ozellikleri, banyo_ozellikleri, salon_ozellikleri, site_imkanlari, depolama_gomme): tüm enum array içinde. **Fotoğrafta görmediklerini sil**, kalan değerler doğru. Hiçbiri görünmüyorsa `[]`.
   - Single-select (balkon_tipi, teras_tipi, mutfak_tipi, banyo_dus, zemin_tipi, pencere_tipi): `"enum1 | enum2 | enum3"` string. **Tek değer bırak** ('|' karakterlerini de sil). Uymuyorsa `null`.

### 2b. Visual gold doldurma rehberi

Her listing'in `data/images/<listing_id>/` klasöründeki fotoğraflara bak. 12 alan:

| Alan | Tip | Değerler |
|---|---|---|
| balkon_tipi | single | cam_balkon \| acik_balkon \| fransiz_balkon \| cikma_balkon \| yok |
| teras_tipi | single | cati_terasi \| normal_teras \| bahce_cikisli \| yok |
| manzara | multi | deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari |
| mutfak_tipi | single | amerikan_acik \| kapali_ayri \| yari_acik |
| mutfak_ozellikleri | multi | ada_tezgah, bar_tezgahi, ankastre, mutfakta_pencere |
| banyo_dus | single | dusakabin \| kuvet \| jakuzi \| sade_dus \| kuvet_ve_dusakabin |
| banyo_ozellikleri | multi | banyoda_pencere, cift_lavabo, hilton_tipi_ayri, ebeveyn_banyosu, duvardan_asma_klozet |
| zemin_tipi | single | parke \| laminat \| seramik \| granit \| mermer \| hali \| karma |
| pencere_tipi | single | standart \| boy_pencere \| panoramik \| cumba \| giyotin |
| salon_ozellikleri | multi | somine, nis, acik_plan_genis, ayri_yemek_alani |
| site_imkanlari | multi | havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani |
| depolama_gomme | multi | gomme_dolap_yatak, vestiyer_giris, gomme_kitaplik |

**Multi-select** alanlar JSON array, **single** alanlar string. Fotoğrafta görünmeyen = `null`.

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
