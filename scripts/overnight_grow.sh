#!/usr/bin/env bash
# Gece büyütme zinciri: scrape -> images -> clean -> yeni-filtre -> label (cost cap) -> merge.
# Çalışan 303'e DOKUNMAZ: tüm yeni çıktı data/processed_big/ altında; index'e elleme yok.
# Her adım dayanıklı: bir adım patlarsa loglar ve devam eder ("takıldığı yere kadar").
cd "C:/Users/mertc/Desktop/DL-Ev-ilan-Projesi" || exit 1
PY=.venv/Scripts/python.exe
BIG=data/processed_big
mkdir -p "$BIG"
LOG="$BIG/overnight.log"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== OVERNIGHT GROW START (hedef +500 yeni, ~800 toplam) ==="

log "[1/6] SCRAPE (--limit 500, resume from existing)"
"$PY" -m scraper.playwright_scraper --limit 500 >>"$LOG" 2>&1 || log "scrape exited non-zero ($?) -- devam"

log "[2/6] IMAGES (download new)"
"$PY" -m scraper.image_downloader >>"$LOG" 2>&1 || log "images exited non-zero ($?) -- devam"

log "[3/6] CLEAN -> $BIG/dataset.jsonl (combined raw)"
"$PY" -m scraper.cleaner --out "$BIG" >>"$LOG" 2>&1 || log "clean exited non-zero ($?) -- devam"

log "[4/6] DATASET_NEW (only listings not already labeled)"
"$PY" scripts/_dataset_diff.py >>"$LOG" 2>&1 || log "diff exited non-zero ($?) -- devam"

log "[5/6] LABEL combined (DeepSeek+Kimi, cost cap \$10, 512px, resume)"
VISION_MAX_IMAGE_EDGE=512 "$PY" -m labeling.run_labeling \
  --input "$BIG/dataset_new.jsonl" \
  --output "$BIG/labeled_new.jsonl" \
  --phase combined \
  --resume \
  --allow-full-batch \
  --max-cost-usd 10 \
  --batch-size 5 \
  --vision-chunk-size 0 >>"$LOG" 2>&1 || log "label exited non-zero ($?) -- devam"

log "[6/6] MERGE -> $BIG/labeled_big.jsonl (303 korunur)"
"$PY" scripts/_merge_labeled.py >>"$LOG" 2>&1 || log "merge exited non-zero ($?) -- devam"

log "=== OVERNIGHT GROW DONE ==="
log "Sonuç dosyaları: $BIG/labeled_big.jsonl (birleşik), $BIG/labeled_new.jsonl (yeni)"
