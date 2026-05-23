# Milestone 1 Shootout Report

## Current Run

Bu koşuda yalnızca iki aday gerçek skor üretti:

- `kimi_k2_6`: `quality_score=0.8267`, `json_score=0.9667`, `slot_score=0.7333`, projected `100K=$146.50`
- `gemma_4_local`: `quality_score=0.8600`, `json_score=1.0000`, `slot_score=0.7667`, projected `100K=$0.00`

Diğer adaylar erişim eksikliği nedeniyle yarışmadı:

- `deepseek_v4_flash`: `missing_env:DEEPSEEK_API_KEY`
- `gemini_3_5_flash`: `missing_env:GEMINI_API_KEY`
- `glm_4_6`: `missing_env:OPENROUTER_API_KEY`

## Supervisor Checkpoint

Gemma 4 local'ın önde çıkması beklenmedik sayılır, ama kök neden kalite farkının tek başına çok güçlü olması değil:

1. Shootout bu koşuda **sadece text slot extraction** ölçtü.
2. Image labeling gold set henüz manuel doldurulmadığı için VLM kalitesi ölçülmedi.
3. DeepSeek, Gemini ve GLM erişim eksikliği yüzünden gerçek karşılaştırmaya girmedi.
4. Gemma local maliyetinin sıfır olması production feasibility açısından güçlü, fakat runtime ve görsel doğruluk henüz kanıtlanmadı.

Bu yüzden `llm/selected.json` şu an “provisional” kabul edilmeli:

- Text ön aday: `gemma_4_local`
- Vision ön aday: `gemma_4_local`
- Karşılaştırmalı API adayı: `kimi_k2_6`

## Decision

API maliyeti şimdiden yaklaşık `$0.50` harcandığı için aynı benchmark tekrar tekrar çalıştırılmayacak. Bir sonraki karar noktası Milestone 2 manuel gold setleri doldurulduktan sonra:

- 30 listing image+text gold seti ile Gemma 4 vs Kimi K2.6 spot-check yapılacak.
- Eğer Gemma image alanlarında zayıf kalırsa Kimi vision labeler olarak seçilecek.
- Eğer Gemma hem text hem image gold setinde yeterliyse labeling maliyetini sıfıra yakın tutmak için Gemma kullanılacak.

Milestone 3 labeling, bu manuel gold setler doldurulmadan başlatılmayacak.
