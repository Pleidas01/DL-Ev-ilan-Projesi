"""
EmlakDataset — mCLIP Fine-Tuning PyTorch Dataset
=================================================
data/processed/dataset.jsonl dosyasını okur.
Her item: görsel tensor + tokenize edilmiş Türkçe metin.

Kullanım:
    from model.dataset import EmlakDataset, get_splits
    full = EmlakDataset("data/processed/dataset.jsonl")
    train_ds, val_ds, test_ds = get_splits(full)
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer
from open_clip import create_model_and_transforms

# ── Sabitler ────────────────────────────────────────────────────────────────────
MCLIP_MODEL   = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
TOKENIZER_ID  = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
IMAGE_SIZE    = 224
MAX_TEXT_LEN  = 77   # CLIP tokenizer sınırı


class EmlakDataset(Dataset):
    """
    Her örnek:
        pixel_values : FloatTensor [3, 224, 224]  — ViT-B/32 için normalize edilmiş
        input_ids    : LongTensor  [max_len]       — XLM-R token id'leri
        attention_mask: LongTensor [max_len]
        meta         : dict — {id, title, price, district, image_path, url}
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer_id: str = TOKENIZER_ID,
        max_text_len: int = MAX_TEXT_LEN,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ) -> None:
        self.max_text_len = max_text_len
        self.augment = augment

        self.records = self._load_jsonl(Path(jsonl_path))

        # Tokenizer (XLM-RoBERTa)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # CLIP görsel ön-işleme (train veya eval transform)
        _, _, preprocess = create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        self.preprocess = preprocess

    # ── Yardımcı ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            # Bozuk görselde siyah kare döndür
            return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

    # ── Dataset API ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Görsel
        img = self._load_image(rec["image_path"])
        pixel_values = self.preprocess(img)  # FloatTensor [3, H, W]

        # Metin
        encoding = self.tokenizer(
            rec["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        return {
            "pixel_values":  pixel_values,
            "input_ids":     encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "meta": {
                "id":         rec.get("id", ""),
                "title":      rec.get("title", ""),
                "price":      rec.get("price", ""),
                "district":   rec.get("district", ""),
                "image_path": rec.get("image_path", ""),
                "url":        rec.get("url", ""),
            },
        }


# ── Split yardımcısı ────────────────────────────────────────────────────────────

def get_splits(
    dataset: EmlakDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Dataset'i train/val/test olarak böler.
    Returns: (train_subset, val_subset, test_subset)
    """
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ── Hızlı test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    jsonl = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset.jsonl"
    print(f"Dataset yükleniyor: {jsonl}")
    ds = EmlakDataset(jsonl)
    print(f"Toplam örnek: {len(ds)}")

    item = ds[0]
    print(f"pixel_values : {item['pixel_values'].shape}  dtype={item['pixel_values'].dtype}")
    print(f"input_ids    : {item['input_ids'].shape}")
    print(f"attention_mask: {item['attention_mask'].shape}")
    print(f"meta         : {item['meta']}")

    train, val, test = get_splits(ds)
    print(f"\nSplit → train={len(train)}  val={len(val)}  test={len(test)}")
