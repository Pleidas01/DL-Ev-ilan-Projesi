# BGE-M3 LoRA RTX Experiment Plan

> Bu plan RTX 5070 Ti makinede uygulanır. MacBook Air üzerinde eğitim veya full eval koşma.

**Goal:** Hazır BGE-M3 baseline'ını bozmadan, küçük bir LoRA adapter eğit ve held-out retrieval metriğiyle dürüstçe karşılaştır.

**Scope:** Yalnız text embedding LoRA. ResNet/ViT, Qwen-VL ve demo refactor kapsam dışı.

## Başarı Kapısı

- Split listing ID bazlı ve deterministik olmalı (`seed=42`).
- Aynı ilandan türeyen sorgu train ve validation'a dağılmamalı.
- LoRA etkisi önce **dense-only** held-out retrieval ile ölçülmeli: metadata hard filter ve reranker kapalı. Aksi halde filtreler baseline'ı tavana taşıyıp adapter farkını gizler.
- İkinci ölçüm end-to-end non-regression smoke olmalı: mevcut filtre + reranker yolu bozulmamalı.
- Adapter yalnız held-out MRR artar ve R@10 gerilemezse demo adayıdır. İyileşme yoksa baseline korunur.

## Task 1: RTX Pre-flight

- [ ] Repo ve `.env` dosyasını RTX makineye taşı.
- [ ] CUDA kontrolü:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

- [ ] Beklenti: `cuda_available=True`, cihaz RTX 5070 Ti.
- [ ] `python -m pytest -q` koş; beklenen `104 passed`, skip yok.
- [ ] `data/processed/labeled.jsonl` için `303` unique ID ve boş olmayan `enriched_doc` doğrula.

## Task 2: Pair Dataset

**Create:**
- `finetune/text_embed/prepare_pairs.py`
- `tests/test_prepare_pairs.py`

- [ ] Önce RED test: deterministic ID split, leakage yok, her positive çift source ID taşıyor, negatif source ID'den farklı.
- [ ] Pozitif çift: kontrollü sentetik Türkçe sorgu + kaynak ilanın `enriched_doc`.
- [ ] Hard negative: baseline dense adaylarından aynı ID olmayan, benzer ama yanlış ilanlar.
- [ ] Çıktı:

```text
finetune/text_embed/data/train.jsonl
finetune/text_embed/data/validation.jsonl
finetune/text_embed/data/split_manifest.json
```

## Task 3: Dense-only Baseline

**Create:**
- `finetune/text_embed/evaluate_dense.py`
- `tests/test_evaluate_dense.py`

- [ ] Önce RED test: R@1/R@5/R@10/MRR hesapları ve source ID eşleşmesi.
- [ ] Validation sorgularını hard filter ve reranker olmadan baseline `BAAI/bge-m3` ile ölç.
- [ ] Sonucu yaz:

```text
finetune/text_embed/results/baseline_dense.json
```

## Task 4: Minimal LoRA Train

**Create:**
- `finetune/text_embed/train_bge_m3_lora.py`

- [ ] BGE-M3 attention modül adlarını `named_modules()` ile yazdır; `query`, `key`, `value` target'larını dosyadan doğrulamadan hard-code etme.
- [ ] PEFT LoRA adapter-only eğitim kullan.
- [ ] Başlangıç ayarı: `r=8`, `lora_alpha=16`, `dropout=0.05`, `epochs=2`, küçük batch + gradient accumulation, CUDA mixed precision.
- [ ] OOM olursa yalnız batch'i küçült; modeli veya değerlendirme sözleşmesini sessizce değiştirme.
- [ ] Adapter'ı ayrı kaydet:

```text
finetune/text_embed/checkpoints/bge_m3_lora/
```

## Task 5: Held-out Karşılaştırma

- [ ] Aynı validation split'i adapter ile dense-only ölç:

```text
finetune/text_embed/results/lora_dense.json
```

- [ ] Baseline vs adapter tablo üret:

```text
finetune/text_embed/results/comparison.md
```

- [ ] End-to-end küçük smoke koş: mevcut `evaluation.run_retrieval_eval` limitleri ve checkpoint kullan.
- [ ] Karar yaz: `ACCEPT` yalnız held-out MRR artar ve R@10 gerilemezse; aksi halde `REJECT_KEEP_BASELINE`.

## Task 6: Sunum Kaydı

- [ ] `docs/SUNUM_AKISI.md` içindeki LoRA slaydını gerçek pre/post sayılarla doldur.
- [ ] Negatif sonuç çıkarsa gizleme: “adapter fayda sağlamadı, baseline korundu” yaz.

