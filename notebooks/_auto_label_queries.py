"""
Manual query set — weak-supervision auto-labeler.
Her query için BM25 top-K + content-word overlap filter → 3-5 relevant_id.
Ödev aşaması için hızlı ground-truth; sonra elle revize edilebilir.
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from rank_bm25 import BM25Okapi

DATA = Path("data/processed/dataset.jsonl")
OUT = Path("data/eval/manual_queries.jsonl")

# NLP-odaklı queries — filtrede OLMAYAN, semantik sıfat / görsel / konum nüansı
QUERIES = [
    ("deniz manzaralı ferah daire",          ["deniz", "manzara"]),
    ("aydınlık geniş salon",                 ["aydinlik", "ferah", "genis"]),
    ("modern beyaz mutfak",                  ["mutfak", "beyaz", "modern"]),
    ("büyük balkonlu güneş alan",            ["balkon", "gunes", "teras"]),
    ("bahçeli müstakil ev",                  ["bahce", "mustakil", "villa"]),
    ("sakin sessiz aile mahallesi",          ["aile", "sakin", "sessiz"]),
    ("metroya yürüme mesafesi merkezi",      ["metro", "merkezi", "yurume"]),
    ("yeni yapı sıfır lüks",                 ["sifir", "yeni", "luks"]),
    ("öğrenci için ekonomik eşyalı",         ["ogrenci", "esyali", "ekonomik", "bekar"]),
    ("yeşil alanlı site içinde",             ["yesil", "park", "site"]),
]

STOPWORDS = {
    "ve", "icin", "ile", "bir", "bu", "o", "da", "de", "mi",
    "kiralik", "daire", "ev", "kiralık",
}


def tok(s: str) -> list[str]:
    s = s.lower()
    tr = str.maketrans("çğıöşü", "cgiosu")
    s = s.translate(tr)
    return re.findall(r"[a-z0-9+]+", s)


def content_tokens(q: str) -> list[str]:
    return [t for t in tok(q) if t not in STOPWORDS and len(t) > 2]


def main() -> None:
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    print(f"[Veri] {len(recs)} ilan.")

    corpus = [tok(r.get("text", "") + " " + r.get("district", "")) for r in recs]
    bm25 = BM25Okapi(corpus)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    labeled = []

    for q, must_tokens in QUERIES:
        q_tok = tok(q)
        scores = bm25.get_scores(q_tok)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:20]

        # Filter: en az bir must_token text'te geçmeli
        picks = []
        for i in top:
            txt = corpus[i]
            if any(mt in txt for mt in must_tokens):
                picks.append((i, scores[i]))
            if len(picks) >= 5:
                break

        # Yetersizse (hiç eşleşme yoksa) top-3 BM25 skoru ile doldur
        if len(picks) < 3:
            for i in top[:5]:
                if i not in [p[0] for p in picks]:
                    picks.append((i, scores[i]))
                if len(picks) >= 3:
                    break

        relevant_ids = [recs[i]["id"] for i, _ in picks[:5]]
        labeled.append({
            "query": q,
            "relevant_ids": relevant_ids,
            "method": "auto-bm25+filter",
            "must_tokens": must_tokens,
            "notes": f"auto top-{len(relevant_ids)} BM25 + content-word filter",
        })

        dbg = [f"{recs[i]['id']}({scores[i]:.1f})" for i, _ in picks[:5]]
        print(f"  {q:<38} → {len(relevant_ids)} pick: {', '.join(dbg)}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[Çıktı] {OUT} yazıldı ({len(labeled)} query).")


if __name__ == "__main__":
    main()
