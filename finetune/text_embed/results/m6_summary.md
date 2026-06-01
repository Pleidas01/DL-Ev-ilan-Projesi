# M6 — BGE-M3 LoRA fine-tune sonucu (2026-06-01)

Kendi eğitilen NN: BGE-M3 multilingual embedding üzerine LoRA adapter (domain adaptation,
Türkçe gayrimenkul). Eğitim RTX 5070 Ti (CUDA) üzerinde.

## Veri
- Korpus: `data/processed_big/labeled_big.jsonl` = **716 ilan** (gece büyütme: 303 → 716).
- Pair'ler (`prepare_pairs`, seed=42, listing-ID split): **train=573, validation=143, listing overlap=0**.

## Eğitim ayarı
- Adapter-only PEFT LoRA, target = `attention.self.{query,key,value}` (24 katman).
- r=8, alpha=16, dropout=0.05, epochs=2, batch=2, grad-accum=4, lr=2e-5, max_len=512.
- Adapter: `finetune/text_embed/checkpoints/bge_m3_lora/adapter_model.safetensors` (4.7MB).

## Held-out dense retrieval (143 sorgu, 716 korpus, dense-only)

| Metrik | Baseline (BGE-M3) | + LoRA | Δ |
|---|---|---|---|
| R@1  | 0.720 | 0.965 | +0.245 |
| R@5  | 0.958 | 1.000 | +0.042 |
| R@10 | 0.993 | 1.000 | +0.007 |
| MRR  | 0.832 | 0.981 | +0.149 |

Sonuç dosyaları: `results/baseline_dense.json`, `results/lora_dense.json`.

## Yöntem geçerliliği / dürüst caveat'ler
- Sorgular sentetiktir (ilan alanlarından üretilmiş: Mahalle+İlçe+oda+fiyat+özellik). Metrik,
  templated-sorgu ↔ ilan embedding hizalamasını ölçer; gerçek-kullanıcı relevansı (insan-gold) değil.
- train/val listing-ID disjoint (overlap=0) → val'deki kazanım ezberleme değil, genelleme.
- Dense-only: metadata hard-filter ve reranker yok; saf embedding katkısı izole edilmiştir.
- R@5/R@10 baseline'da zaten ~tavan; LoRA asıl R@1 (+24.5p) ve MRR (+14.9p) iyileştirdi.

## Yeniden üretim
```
python -m finetune.text_embed.prepare_pairs --input data/processed_big/labeled_big.jsonl
python -m finetune.text_embed.evaluate_dense --listings data/processed_big/labeled_big.jsonl \
  --output finetune/text_embed/results/baseline_dense.json
python -m finetune.text_embed.train_bge_m3_lora --target-modules query key value
python -m finetune.text_embed.evaluate_dense --listings data/processed_big/labeled_big.jsonl \
  --adapter-path finetune/text_embed/checkpoints/bge_m3_lora \
  --output finetune/text_embed/results/lora_dense.json
```
