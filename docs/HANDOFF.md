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
python3 -m pytest -q   # 112 passed beklenir (2026-06-01)

# .env kontrolü — Read tool'una GÜVENME, gerçek key doluluğu için:
python3 -c "from dotenv import dotenv_values; v=dotenv_values('.env'); [print(f'{k}: {\"FILLED\" if val else \"EMPTY\"}') for k,val in v.items() if 'API_KEY' in k]"
# Beklenen: DEEPSEEK_API_KEY=FILLED, MOONSHOT_API_KEY=FILLED, GEMINI_API_KEY=FILLED
# OPENROUTER_API_KEY EMPTY OK (GLM-4.6 skip ediliyor)

# Aktif canonical dataset sağlık kontrolü
python3 -c "import json; rows=[json.loads(l) for l in open('data/processed/dataset.jsonl', encoding='utf-8')]; print(f'total={len(rows)}, slug={len(rows[0][\"filter_values\"])}, avg_images={sum(len(r[\"all_image_paths\"]) for r in rows)/len(rows):.2f}')"
# Beklenen: total=303, slug=150, avg_images=15.81
```

**Test kırmızıysa, key EMPTY ise veya dataset eksikse: durup kullanıcıya sor.**

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
M0 (scrape + clean)   ✅ DONE — 305 raw → 303 aktif ilan, 150 canonical slug
M1 (slot shootout)    ✅ DONE — text=deepseek_v4_pro (A/B revize), vision=kimi_k2_6
M2.0 (schema refactor)✅ DONE — schema sadeleştirme: kitchen_type çıktı, visual 12→7, imkanlar rename
M2 (manuel kontrol)   🔄 AKTİF — yeni 10 ilanlık JSONL çıktıları kullanıcı göz kontrolünde
M1.5 (vision shoot)   ✅ DONE — Kimi multi-image winner; description shootout açık
M3 (labeling)         ✅ DONE — full 303 satır labeled.jsonl + clean_json.json hazır
M4 (indexing+retr.)   ✅ BASELINE DONE — 303 Chroma row + checkpoint'li hafif eval
Canonical filters     ✅ scrape + download + cleaner + full labeling DONE; 112 passed
M5 (RAG+UI)           ✅ BASELINE DONE — browser ile canlı Streamlit smoke
M6                    ▶️ SIRADAKİ — RTX 5070 Ti üzerinde ölçüm kapılı BGE-M3 LoRA
M8                    ⏳ PENDING        M7 (time series) ⛔ KAPSAM DIŞI (arkadaş yaptı)
```

> **Mimari nesil ayrımı (önemli):** Bu projede üç mimari nesil var — Gen1 CLIP (terk), Gen2 el-yapımı JSON şema (tarihsel), Gen3 canonical Emlakjet registry (aktif). M4 ve M5 artık Gen3 baseline üzerinde tamamlandı. Detay: `docs/MIMARI_EVRIMI.md`.

**Acil sıradaki iş:** MacBook Air'de ağır süreç başlatma. RTX 5070 Ti makinede `docs/superpowers/plans/2026-06-01-bge-m3-lora-rtx.md` planını uygula; ardından `docs/SUNUM_AKISI.md` sayılarını final sonuçlarla güncelle.

> **MacBook Air sınırı:** Yerel PyTorch ortamı CPU-only (`mps_available=False`). Full `303` sorgu eval ve LoRA eğitimi burada koşulmaz. Küçük doğrulama gerekiyorsa eval limitleri ve checkpoint kullanılır.

### M6 MacBook hazırlığı (DONE — 2026-06-01)

- `finetune/text_embed/prepare_pairs.py`: `seed=42`, listing-ID split, split-local deterministik farklı-ID negatif.
- Üretilen pair çıktısı: `242` train, `61` validation, listing-ID overlap `0`.
- `finetune/text_embed/evaluate_dense.py`: dense-only R@1/R@5/R@10/MRR; metadata hard filter ve reranker yok.
- `finetune/text_embed/train_bge_m3_lora.py`: adapter-only PEFT, varsayılan `r=8`, `alpha=16`, `dropout=0.05`, `epochs=2`; explicit target doğrulama ve CUDA fail-loud kapısı.
- MacBook doğrulaması: `cuda_available=False`, `mps_available=False`; tam paket `112 passed`, skip yok. Gerçek eğitim ve full dense eval çalıştırılmadı.

RTX makinede sırayla:

```bash
python -m pytest -q
python -m finetune.text_embed.prepare_pairs
python -m finetune.text_embed.evaluate_dense \
  --output finetune/text_embed/results/baseline_dense.json
python -m finetune.text_embed.train_bge_m3_lora --inspect-target-modules
python -m finetune.text_embed.train_bge_m3_lora \
  --target-modules query key value
python -m finetune.text_embed.evaluate_dense \
  --adapter-path finetune/text_embed/checkpoints/bge_m3_lora \
  --output finetune/text_embed/results/lora_dense.json
python -m evaluation.run_retrieval_eval \
  --known-limit 4 \
  --benchmark-limit 3 \
  --checkpoint-every 1
```

> `query key value` komutu inspection çıktısından sonra çalıştırılır. BGE-M3 modül adları farklıysa CLI sessizce devam etmez; eksik target için açık hata verir.

---

## 1.5 M4/M5 baseline sonucu (DONE — 2026-06-01)

**Gerçek index:**
```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m indexing.build_chroma \
  --input data/processed/labeled.jsonl \
  --batch-size 8
```

- Chroma collection count: `303`.
- Metadata skaler-only; multi-enum exact flag'leri doğrulandı.
- İki ilandaki `"Açık Balkon, Açık Teras"` parser birleşimi mekanik olarak düzeltildi; invalid birleşik Chroma flag sayısı `0`.
- Batch `64` MacBook Air CPU-only ortamında bellek baskısı oluşturdu; varsayılan `8` yapıldı.

**Checkpoint'li gold-free eval:**
```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.run_retrieval_eval \
  --known-limit 4 \
  --benchmark-limit 3 \
  --checkpoint-every 1
```

| Metrik | Hafif örneklem sonucu |
|---|---:|
| Synthetic known-item sorgu | 4 |
| R@1 / R@5 / R@10 / MRR | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| Canlı benchmark sorgu | 3 |
| Query coverage | 1/3 = 0.3333 |
| Filter satisfaction | 8/8 = 1.0000 |

Bu sayı **full relevance gold değildir**. MacBook Air'i yormamak için bounded proxy örneklemidir. İlk ve üçüncü canlı benchmark sorgularının tüm filtrelerini karşılayan ilan aktif batch'te yoktur; bu nedenle coverage düşüktür. `"denize yakın"` canonical semantiği `near_sea` olarak düzeltildi (`has_sea_view` değil).

**Canlı Streamlit QA:**
```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m streamlit run ui/app.py
```

Browser smoke sorgusu: `"geniş salonlu denize yakın 2+1 asansörlü ev"`. Sonuç: `8` kart, canonical detaylar, `2+1` / `Asansör` / `Denize Yakın` match çipleri ve yalnız retrieved ilanlara dayalı RAG cevabı.

**Sıradaki:** RTX LoRA planı: `docs/superpowers/plans/2026-06-01-bge-m3-lora-rtx.md`.

---

## 2. TARİHSEL — Gen 2 manuel gold workflow → taşındı

Eski (Gen 2) manuel gold doldurma workflow'u (listing/visual gold) artık çalıştırılmıyor ve **`docs/MIMARI_EVRIMI.md` → "Tarihsel ek"** bölümüne taşındı. Canonical (Gen 3) akışta yeni gold template rebuild yapılmaz; değerlendirme 10-ilan göz kontrolü + M4 R@10 retrieval eval ile yapılır.

> İsteğe bağlı güçlendirme: **query gold** (`expected_listing_ids`) doldurulursa gerçek relevance R@10 ayrıca ölçülebilir. Kritik teslim yolu gold-free proxy + filter satisfaction ile tamamlandı. Komut: `python3 -m evaluation.gold_helper --query "..."`.

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

> **GÜNCEL (2026-05-31):** Aktif input `data/processed/dataset.jsonl`: 303 ilan, 150-slug `filter_values`. `--phase text|vision|combined` mevcut. DeepSeek yalnız başlık+açıklamadan null alanlara açık kanıtla true/false yazar. Kimi yalnız görsellerden kalan null boolean alanlara açık kanıtla true yazar; görünmeyen özellik false olmaz.

10 ilanlık ölçüm çıktıları: `data/processed/labeled_benchmark_deepseek_text_10.jsonl` ve `data/processed/labeled_benchmark_kimi_vision_10.jsonl`. Kullanıcı göz kontrolü sonrası **kalan: full 303-ilan build** → `data/processed/labeled.jsonl`.

CLI (gerçek argümanlar için `python3 -m labeling.run_labeling --help`):

```
labeling/run_labeling.py
  --input data/processed/dataset.jsonl
  --output data/processed/labeled_text.jsonl
  --phase text
  --batch-size 3              # DeepSeek ölçümünde seçilen worker sayısı
  --resume                    # idempotent, ID bazlı
  --max-cost-usd <cap>        # cost cap, aşılınca dur

VISION_MAX_IMAGE_EDGE=512 python3 -m labeling.run_labeling \
  --input data/processed/labeled_text.jsonl \
  --output data/processed/labeled.jsonl \
  --phase vision \
  --batch-size 5 \
  --vision-chunk-size 0 \
  --max-cost-usd <cap>
```

Kimi operasyon notu: default `512px`, `chunk=0`, `5 worker`. Tekil bir ilan art arda timeout verirse yalnız o ilanı `--resume --vision-chunk-size 5 --batch-size 1` fallback ile tamamla. Timeout transient retry kapsamındadır; paralel batch diğer başarılı row'ları kaydetmeye devam eder ve başarısız ID'leri sonunda açıkça raporlar.

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

**Tarihsel pre-flight test (kodda mevcut):** Gold'daki ilk 16 listing'i etiketle, `facts_gold` + `visual_gold` ile karşılaştır, accuracy raporla. Eski kapı:
- **`facts_all` (tam birleştirilmiş pipeline, 21 alan): ≥ 0.75** — eski compatibility gate buydu.
- visual (Jaccard ortalaması): ≥ 0.70
- `facts_llm` (TEXT_FACT_FIELDS, 6 hybrid/desc alan): tarihsel compatibility raporunda **sadece teşhis, kapı DEĞİL** idi (bkz. `tests/test_run_labeling.py::test_passes_thresholds_uses_facts_all_not_facts_llm`).

> **Canonical modalite ayrımı (2026-05-31):** scraper `İlan Özellikleri` bullet'larını deterministik fact olarak önceden doldurur. DeepSeek yalnız başlık+açıklamayı okur ve kalan null alanlara açık kanıtla `true`/`false` yazar. Kimi yalnız görselleri okur ve kalan null boolean alanlara açık kanıtla positive-only `true` ekler.

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
10. **`data/processed/` smoke kalıntıları:** `labeled_preflight*.jsonl` + boş `*_stdout/stderr.log` dosyaları M3 pre-flight denemelerinden kaldı; gerçek `labeled.jsonl` artık hazır. Temizlik kullanıcı onayıyla yapılır (silme = geri alınamaz, bkz. §7).

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
