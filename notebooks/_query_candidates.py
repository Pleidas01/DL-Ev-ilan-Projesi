"""
Manual query set için aday üretici.
BM25 skoruyla her query için top-8 aday çıkarır.
"""

import json
import re
import sys
from pathlib import Path

# UTF-8 console
sys.stdout.reconfigure(encoding="utf-8")

from rank_bm25 import BM25Okapi

DATA = Path("data/processed/dataset.jsonl")
OUT_QUERIES = Path("data/eval/manual_queries.jsonl")

QUERIES = [
    # A. Lokasyon+oda
    "kadıköy 2+1 daire",
    "beşiktaş merkezi 1+1 kiralık",
    "şişli metroya yakın 3+1",
    "üsküdar eşyalı daire",
    "başakşehir bahçeşehir ferah 3+1",
    # B. Semantik sıfat
    "deniz manzaralı daire",
    "ferah aydınlık geniş salon",
    "yeni binada modern daire",
    "aileye uygun sakin mahalle",
    "ulaşımı kolay merkezi konum",
    # C. Fonksiyonel
    "öğrenciye uygun eşyalı stüdyo",
    "bekarlar için 1+1",
    "aile bütçesine uygun kiralık",
    # D. Görsel
    "modern beyaz mutfak",
    "büyük balkonlu daire",
    "bahçeli müstakil ev",
    "açık mutfak geniş salon",
    # E. Bina özelliği
    "asansörlü güvenlikli site",
    "otoparklı bina",
    "güvenlikli rezidans lüks",
    "doğalgaz kombili daire",
    # F. Kombinasyon
    "deniz manzaralı 2+1 kadıköy",
    "yeni yapı 1+1 merkezi",
    "bahçeli villa geniş aile",
    "eşyalı kiralık asansörlü",
]


def tok(s: str) -> list[str]:
    s = s.lower()
    # TR karakter normalize (BM25 için yardımcı)
    tr = str.maketrans("çğıöşü", "cgiosu")
    s = s.translate(tr)
    return re.findall(r"[a-z0-9+]+", s)


def load_records() -> list[dict]:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def short(rec: dict, n: int = 80) -> str:
    t = rec.get("text", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def main() -> None:
    recs = load_records()
    print(f"[Veri] {len(recs)} ilan yüklendi.\n")

    # BM25 corpus: text + district beraber (district match için bonus)
    corpus_tokens = [
        tok((r.get("text", "") + " " + r.get("district", "")))
        for r in recs
    ]
    bm25 = BM25Okapi(corpus_tokens)

    for qi, q in enumerate(QUERIES, 1):
        q_tok = tok(q)
        scores = bm25.get_scores(q_tok)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:8]

        print(f"=== Q{qi:02d}: \"{q}\" ===")
        for idx, i in enumerate(top, 1):
            r = recs[i]
            dist = r.get("district", "").replace("Istanbul - ", "").split(" - ")
            dist_short = " / ".join(dist[:2]) if dist else ""
            print(f"  [{idx}] id={r['id']:<10} {dist_short[:30]:<30} {r.get('price',''):<15} | {short(r, 90)}")
        print()


if __name__ == "__main__":
    main()
