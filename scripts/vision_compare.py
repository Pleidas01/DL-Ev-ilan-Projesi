"""qwen3-vl (yerel, ücretsiz) vs Kimi (bulut, ücretli) vision-etiketleme karşılaştırması.
Aynı ilanlar + aynı canonical vision prompt + aynı (kısıtlı) fotolar; süre/maliyet/alan-uyumu.
Kullanım: PYTHONPATH=. OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct VISION_MAX_IMAGE_EDGE=512 python scripts/vision_compare.py
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

from llm.clients import candidate_by_id, complete_vision_json, estimate_cost_usd
from llm.shootout_vision import VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, parse_vision_json
from schema.emlakjet_filters import specs_for_source

N_LISTINGS = 5
N_PHOTOS = 8
VISION_FIELDS = [s.slug for s in specs_for_source("image_vlm")]

# Aynı ilanlar için görsel yolları (dataset.jsonl) — fotosu bol ilk N ilan.
rows = [json.loads(l) for l in open("data/processed/dataset.jsonl", encoding="utf-8") if l.strip()]
picked = [r for r in rows if len(r.get("all_image_paths", [])) >= 4][:N_LISTINGS]

MODELS = ["qwen3_vl_local", "kimi_k2_6"]
results = {m: {"times": [], "cost": 0.0, "outputs": {}} for m in MODELS}

for r in picked:
    lid = r["id"]
    imgs = [p for p in r["all_image_paths"][:N_PHOTOS] if Path(p).is_file()]
    if not imgs:
        continue
    print(f"\n=== ilan {lid} ({len(imgs)} foto) ===", flush=True)
    for m in MODELS:
        cand = candidate_by_id(m)
        t0 = time.time()
        try:
            raw = complete_vision_json(cand, VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, imgs)
            dt = time.time() - t0
            parsed = parse_vision_json(raw).get("filters", {}) or {}
        except Exception as e:
            dt = time.time() - t0
            parsed = {}
            print(f"  {m}: HATA {type(e).__name__}: {str(e)[:80]} ({dt:.1f}s)", flush=True)
            results[m]["times"].append(dt)
            results[m]["outputs"][lid] = {}
            continue
        # kaba maliyet: yerel=0; kimi ~5000 token/foto input + ~300 output
        cost = 0.0 if cand.local else estimate_cost_usd(cand, 5000 * len(imgs), 300)
        results[m]["cost"] += cost
        results[m]["times"].append(dt)
        results[m]["outputs"][lid] = parsed
        true_fields = [k for k in VISION_FIELDS if parsed.get(k) is True]
        print(f"  {m}: {dt:.1f}s, ${cost:.4f}, true alanlar={true_fields}", flush=True)

# alan-uyumu: qwen3 vs kimi (Kimi referans), her ilan her vision alanı için
agree = total = 0
for r in picked:
    lid = r["id"]
    q = results["qwen3_vl_local"]["outputs"].get(lid, {})
    k = results["kimi_k2_6"]["outputs"].get(lid, {})
    for f in VISION_FIELDS:
        total += 1
        if (q.get(f) is True) == (k.get(f) is True):
            agree += 1

def avg(xs): return sum(xs) / len(xs) if xs else 0.0
print("\n" + "=" * 60, flush=True)
print("KARŞILAŞTIRMA ÖZETİ", flush=True)
print(f"{'model':18} {'ort_süre':>10} {'top_maliyet':>12}", flush=True)
for m in MODELS:
    print(f"{m:18} {avg(results[m]['times']):>9.1f}s {results[m]['cost']:>11.4f}$", flush=True)
print(f"\nqwen3 vs Kimi alan-uyumu (true/değil): {agree}/{total} = {agree/total:.2%}" if total else "uyum: n/a", flush=True)
