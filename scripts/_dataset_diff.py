"""Gece büyütme: data/processed_big/dataset.jsonl içinden, mevcut 303 etiketli
ilanda OLMAYAN (yeni) satırları dataset_new.jsonl'ye yaz. Çalışan 303 yeniden
etiketlenmez (para israfı yok)."""
import json
from pathlib import Path

EXISTING_LABELED = Path("data/processed/labeled.jsonl")
BIG_DATASET = Path("data/processed_big/dataset.jsonl")
OUT = Path("data/processed_big/dataset_new.jsonl")


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(l)["id"]) for l in path.open(encoding="utf-8") if l.strip()}


def main() -> None:
    done = _ids(EXISTING_LABELED)
    rows = [json.loads(l) for l in BIG_DATASET.open(encoding="utf-8") if l.strip()]
    new = [r for r in rows if str(r.get("id")) not in done]
    with OUT.open("w", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dataset_diff: total={len(rows)} already_labeled={len(done)} new={len(new)} -> {OUT}")


if __name__ == "__main__":
    main()
