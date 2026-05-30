# Milestone 1 Shootout Report (final 4-model)

Dataset: `data/processed/dataset.jsonl` (1430 rows). Benchmark: 30 Turkish slot-extraction queries.

## Final scores

| Model | Status | quality | json | slot | 100K proj. |
|-------|--------|---------|------|------|------------|
| **`kimi_k2_6`** | **ok** | **0.8967** | 1.0000 | **0.8278** | $146.50 |
| `gemma_4_local` | ok | 0.8833 | 1.0000 | 0.8056 | $0.00 |
| `deepseek_v4_flash` | ok | 0.8733 | 1.0000 | 0.7889 | $15.40 |
| `gemini_3_5_flash` | error (429 quota) | — | — | — | $285.00 |

Kimi + Gemma rows: original run (2026-05-24). DeepSeek: `llm/shootout_rows_v2.json` v2 run. Gemini: v2 run hit per-minute then **daily free-tier cap (20 req/model)**; partial 13/30 not merged (not comparable).

Artifacts: `llm/shootout_rows.json`, `llm/shootout_rows_v2.json`, `llm/selected.json`.

## `text_model` winner: `kimi_k2_6`

Among **completed** benchmarks, Kimi has the highest `quality_score` (0.40×json + 0.60×slot):

1. **Kimi 0.8967** — best slot accuracy (0.8278); JSON perfect.
2. **Gemma 0.8833** — $0 at 100K but −1.3 pp quality vs Kimi.
3. **DeepSeek 0.8733** — cheapest API ($15.40/100K) but lowest slot score of the three OK models.

Gemini excluded from ranking until a full 30-query run succeeds (enable billing or wait for daily quota reset).

`vision_model`: `kimi_k2_6` (only vision-capable model with full ok benchmark among feasible rows under $500/100K).

## v2 run notes

- DeepSeek + Gemini shootout (~3 min API); Gemini failed on free-tier RPM then RPD limits.
- Session API spend ≪ $2 cap.
