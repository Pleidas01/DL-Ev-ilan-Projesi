"""Gece büyütme: mevcut 303 labeled.jsonl + yeni labeled_new.jsonl ->
labeled_big.jsonl (id'ye göre benzersiz, mevcut 303 öncelikli). Çalışan 303
dosyasına dokunmaz; sadece birleşik yeni dosya üretir."""
import json
from pathlib import Path

EXISTING = Path("data/processed/labeled.jsonl")
NEW = Path("data/processed_big/labeled_new.jsonl")
OUT = Path("data/processed_big/labeled_big.jsonl")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    merged: dict[str, dict] = {}
    for r in _rows(EXISTING):          # mevcut 303 önce (öncelikli)
        merged[str(r["id"])] = r
    added = 0
    for r in _rows(NEW):               # yeni ilanlar, var olanı ezmez
        if str(r.get("id")) not in merged:
            merged[str(r["id"])] = r
            added += 1
    with OUT.open("w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merge: existing={len(_rows(EXISTING))} new_added={added} total={len(merged)} -> {OUT}")


if __name__ == "__main__":
    main()
