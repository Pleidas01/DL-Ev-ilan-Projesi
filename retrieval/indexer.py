"""
ChromaDB Indexer
================
dataset.jsonl'deki ilanları mCLIP ile encode edip ChromaDB'ye yazar.
Fine-tuned checkpoint varsa kullanır; yoksa base model ile çalışır.

Kullanım:
    # Base model ile (fine-tuning öncesi test):
    python retrieval/indexer.py

    # Fine-tuned model ile:
    python retrieval/indexer.py --checkpoint model/checkpoints/best.pt

    # Yeniden index (var olan koleksiyonu sil):
    python retrieval/indexer.py --reset
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import chromadb
import multilingual_clip.pt_multilingual_clip as pt_mclip
import transformers
from open_clip import create_model_and_transforms
from peft import PeftModel


# ── Sabitler ────────────────────────────────────────────────────────────────────
MCLIP_MODEL    = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
COLLECTION_NAME = "emlak_listings"
CHROMA_DIR     = "data/chroma"
BATCH_SIZE     = 16


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(checkpoint_path: str | None, device: torch.device):
    """Text encoder + Image encoder yükler."""
    print(f"[Model] {MCLIP_MODEL} yükleniyor...")

    text_model = pt_mclip.MultilingualCLIP.from_pretrained(MCLIP_MODEL)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(MCLIP_MODEL)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[Checkpoint] {checkpoint_path} yükleniyor...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # LoRA ağırlıkları varsa PeftModel olarak yükle
        try:
            text_model = PeftModel.from_pretrained(text_model, checkpoint_path)
        except Exception:
            text_model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    else:
        print("[Model] Checkpoint bulunamadı, base model kullanılıyor.")

    # XLM-R MPS'te bazı op'ları desteklemiyor — text encoder CPU'da çalışır
    text_device = torch.device("cpu")
    text_model = text_model.to(text_device).eval()

    image_model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    image_model = image_model.to(device).eval()

    return text_model, image_model, tokenizer, preprocess, text_device


def encode_batch_images(image_model, images: list[Image.Image], preprocess, device) -> torch.Tensor:
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = image_model.encode_image(tensors)
    return F.normalize(feats.float(), dim=-1).cpu()


def encode_batch_texts(text_model, tokenizer, texts: list[str], device) -> torch.Tensor:
    # mCLIP forward(txt, tokenizer) — raw metin + tokenizer alır
    with torch.no_grad():
        feats = text_model(texts, tokenizer)
    return F.normalize(feats.float(), dim=-1).cpu()


def load_records(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_index(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"[Device] {device}")

    records = load_records(args.data)
    print(f"[Veri] {len(records)} ilan bulundu.")

    text_model, image_model, tokenizer, preprocess, text_device = load_models(args.checkpoint, device)

    # ChromaDB bağlantısı
    client = chromadb.PersistentClient(path=args.chroma_dir)

    if args.reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print(f"[ChromaDB] '{COLLECTION_NAME}' koleksiyonu silindi.")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    new_records  = [r for r in records if r["id"] not in existing_ids]

    if not new_records:
        print("[Index] Tüm ilanlar zaten indexlenmiş.")
        return

    print(f"[Index] {len(new_records)} ilan encode ediliyor...")

    for i in tqdm(range(0, len(new_records), BATCH_SIZE), desc="Indexleniyor"):
        batch = new_records[i : i + BATCH_SIZE]

        # Görsel yükle
        images = []
        for rec in batch:
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), 0)
            images.append(img)

        texts = [rec["text"] for rec in batch]

        img_embs  = encode_batch_images(image_model, images, preprocess, device)
        txt_embs  = encode_batch_texts(text_model, tokenizer, texts, text_device)

        # Görsel + metin embedding'lerini ortala (late fusion)
        combined  = F.normalize((img_embs + txt_embs) / 2, dim=-1)

        collection.upsert(
            ids=[rec["id"] for rec in batch],
            embeddings=combined.tolist(),
            metadatas=[{
                "id":         rec["id"],
                "title":      rec["title"],
                "price":      rec["price"],
                "district":   rec["district"],
                "image_path": rec["image_path"],
                "url":        rec["url"],
                "text":       rec["text"][:200],
            } for rec in batch],
        )

    total = collection.count()
    print(f"\n[Index] Tamamlandı. Toplam koleksiyon boyutu: {total}")
    print(f"[ChromaDB] Kaydedildi: {args.chroma_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mCLIP ChromaDB Indexer")
    p.add_argument("--data",       default="data/processed/dataset.jsonl")
    p.add_argument("--chroma_dir", default=CHROMA_DIR)
    p.add_argument("--checkpoint", default=None, help="Fine-tuned model checkpoint")
    p.add_argument("--reset",      action="store_true", help="Koleksiyonu sıfırla")
    return p.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
