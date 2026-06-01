# M4 + M5 — Canonical (Gen3) Tasarımı

> **Durum notu (2026-06-01):** Bu sözleşmenin Gen3 baseline uygulaması tamamlandı. Güncel ölçüm ve RTX LoRA devri için `docs/HANDOFF.md §1.5` ve `docs/superpowers/plans/2026-06-01-bge-m3-lora-rtx.md` oku.

> Tarih: 2026-05-31. Bağlam: `docs/MIMARI_EVRIMI.md` (Gen1 CLIP → Gen2 el-yapımı şema → **Gen3 canonical registry**). Bu spec M4 (indexing+retrieval) ve M5 (RAG+UI) modüllerini Gen3 sözleşmesine hizalar.

## Problem / mevcut durum

İnceleme (git + kod) "M4/M5 = eski mimari iskelet" etiketinin yanlış olduğunu gösterdi:
- **M4 zaten Gen3.** `composer.py` (`filter_values` 150 slug → Chroma skaler metadata + multi_enum `__option` flag), `retriever.py` (slot extract → `slots_to_where` → BGE-M3 → CrossEncoder, sonuç `filters` anahtarı), `build_slot_prompt` (registry'nin boş şemasını prompt'a basar) — hepsi canonical. Eksik: gerçek index build + ölçülü eval.
- **M5 gerçekten Gen2.** `chat/rag_response.py` ve `ui/app.py` registry'den önce (`a3b6804`) yazıldı; hâlâ retriever'ın eski `facts` sözleşmesini okuyor (retriever Gen3'te `filters` döndürüyor) → RAG yapısal fact görmüyor, kartlar boş. `CARD_FACTS` içinde `has_parking` var (Gen3'te yok; `has_open_parking`/`has_closed_parking`). Enum değerler slug (`kombi_dogalgaz`) — kart için Türkçe label gerek.

## Kararlar (kullanıcı onaylı)

1. **M4 eval = ölçülü R@10** (R@5/R@10/MRR). 30 gold sorgunun `expected_listing_ids`'i doldurulacak (manuel, kullanıcı). Hedef R@10 ≥ 0.60.
2. **Match reason = LLM inline metin + deterministik çip.** RAG cevabı serbest gerekçe yazar; ayrıca her kartta sorgunun karşılanan hard-filter'ları kod ile çip olarak gösterilir (Rule 5: kod cevaplıyor).
3. **Gen2 ölü kod:** dokunma (Rule 3). `flatten_slots` (retriever import ediyor ama çağırmıyor), `visual_qualities` (composer okumuyor), gen2 top-level fact'ler veride kalır — sadece belgelenir. Şema/veri temizliği ayrı iş.

## Bağımlılık / blok durumu (2026-05-31)

- `data/processed/labeled.jsonl` **YOK** (arka plan labeling job henüz üretmedi). → **C1, C2 ve R@10 ölçümü bloklu.** Kod + testler bloklu değil.
- `enriched_doc` doğrulaması: text-only benchmark'ta `Gorsel ozellikler:` boş çıkıyor. **C1'de kontrol:** vision geçişi sonrası `enriched_doc` regen ediliyor mu? Bozuksa yüksek sesle bildir (M3'ün işi).

## Durum (2026-05-31)

- **C4, C5, C6 DONE** (TDD, 84 passed; öncesi 70). M5 Gen3 göçü tamam.
- **C3 = GOLD-FREE eval** (kullanıcı kararı, 2026-05-31): manuel gold yok; iki otomatik metrik — (1) sentetik known-item R@K (her ilanın fact'lerinden sorgu üret → kendisi top-K'da mı), (2) filter constraint satisfaction@K (BENCHMARK_QUERIES top-K hard-filter karşılama oranı). Detay: `docs/HANDOFF.md §1.5`.
- **C1, C2 DEVİR** sıradaki agent'a — `labeled.jsonl`'a bloklu. Bkz. `docs/HANDOFF.md §1.5`.

## Komponentler

### C1 — M4 gerçek index build  *(BLOKLU: labeled.jsonl)*
`python -m indexing.build_chroma --input data/processed/labeled.jsonl`. Kod hazır (resume var). Doğrula: collection count == 303, metadata skaler-only (None düşmüş, list `|`, multi_enum flag bool).

### C2 — M4 retriever gerçek-Chroma smoke  *(BLOKLU: index)*
`slots_to_where` çıktısı gerçek ChromaDB grameriyle çalışıyor mu (FakeCollection değil). 5-8 temsili sorgu uçtan uca, top-K göz sağlaması.

### C3 — M4 gold-free eval  *(YENİ KOD, sıradaki agent, TDD)*
`evaluation/run_retrieval_eval.py`: (1) sentetik known-item R@1/R@5/R@10+MRR (ilan fact'lerinden sorgu üret → kendisi top-K'da mı), (2) filter satisfaction@K (BENCHMARK_QUERIES top-K kısıt karşılama). Saf metrik fonksiyonları + retriever DI. Çıktı `evaluation/results/retrieval_eval.{json,md}`.

### C4 — M5 göç: facts→filters  *(şimdi)*
- `rag_response._build_user_prompt`: `result["filters"]` oku (facts değil).
- `app.py`: `result["filters"]` oku; `CARD_FACTS`'teki `has_parking` → canonical; enum slug→Türkçe label.

### C5 — M5 deterministik match çipleri  *(şimdi)*
Retriever sonuç-dict'ine `matched_filters` ekle: sorgunun hard-filter'larından bu ilanın metadata'sında karşılananlar. Tek yerde (retriever, slots+metadata orada), test edilir. UI çip render eder.

### C6 — M5 label helper  *(şimdi)*
`schema/emlakjet_filters.py` veya util: `label_for(slug, value)` ters-arama (`heating_type="kombi_dogalgaz"`→"Kombi Doğalgaz", `has_elevator=True`→"Asansör", `district="Kadikoy"`→"Kadikoy"). Registry `labels`+`values` zaten var.

## Test (Rule 9/12)
Mevcut 70 yeşil korunur. Eklenenler: R@k/MRR metrik testi (alakalı id top-K'da sayılır, business intent), `matched_filters` testi (ilan filtreyi karşılamayı bırakınca çip kaybolur), rag_response `filters` okuma testi, `label_for` testi. Hiçbiri skip değil.

## Kapsam dışı
M3 labeling (job koşuyor), M6 NN, M8 rapor, Gen2 şema/veri temizliği, enriched_doc regen (M3 işi — sadece doğrula).

## Başarı kriteri
- Tüm yeni testler + mevcut 70 yeşil. (Yapıldı: 84 passed.)
- M5: gerçek `filters` ile dolu kart + Türkçe label + match çipi (benchmark verisiyle birim-doğrulandı; canlı demo build sonrası).
- labeled.jsonl hazır olunca: build → 303 collection → smoke + C3 gold-free metrikler (known-item R@K + filter satisfaction) + canlı demo göz kontrolü.
