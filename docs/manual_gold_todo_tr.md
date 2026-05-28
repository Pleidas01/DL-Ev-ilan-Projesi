# Milestone 2 — Manual Gold Set (GÜNCEL DEĞİL — yönlendirme)

> **Uyarı:** Bu dosya ESKİ gold şemasını (`text_gold` / `image_gold` / `kitchen_type` vb.) anlatıyordu ve artık geçerli DEĞİL. Şema 2026-05 refactor'unda `facts_gold` (21 alan) + `visual_gold` (6 alan) olarak değişti.

Manuel gold doldurma için güncel ve tek doğru kaynak:

- **Adım adım workflow + alan tabloları:** [`docs/HANDOFF.md`](HANDOFF.md) §2 (Manuel gold doldurma)
- **Şema tanımları (single source of truth):** [`docs/PROJECT.md`](PROJECT.md) §7a (`facts_gold`) ve §7b (`visual_gold`)
- **Hedef dosyalar:** `labeling/gold_listings_manual_todo.jsonl` (listing gold), `evaluation/gold_queries_manual_todo.jsonl` (query gold — M4'e ertelendi)
- **Yardımcılar:** `python -m labeling.gold_helper --listing <ID>`, `python -m evaluation.gold_helper --query "..."`

Doluluk durumu ve "done" eşikleri için `docs/STATUS.md` (M2 bölümü).
