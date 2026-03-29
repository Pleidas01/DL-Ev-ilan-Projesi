"""
mCLIP LoRA Fine-Tuning — InfoNCE Contrastive Loss
==================================================
M-CLIP/XLM-Roberta-Large-Vit-B-32 modelinin Türkçe gayrimenkul
verisine göre fine-tuning'ini yapar. Text encoder'a LoRA uygulanır,
image encoder dondurulur (frozen).

Kullanım:
  # Mac smoke-test (CPU/MPS):
  python model/fine_tune.py --epochs 1 --batch_size 2 --limit 10

  # Masaüstü tam training (CUDA):
  python model/fine_tune.py --epochs 10 --batch_size 32 --fp16

  # Checkpoint'ten devam:
  python model/fine_tune.py --resume model/checkpoints/epoch_3.pt
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import multilingual_clip.pt_multilingual_clip as pt_mclip
import transformers
from open_clip import create_model_and_transforms
from peft import LoraConfig, TaskType, get_peft_model


from model.dataset import EmlakDataset, get_splits

# ── Sabitler ────────────────────────────────────────────────────────────────────
MCLIP_MODEL  = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
EMBED_DIM    = 512   # ViT-B/32 çıkış boyutu
TEMPERATURE  = 0.07  # InfoNCE sıcaklık parametresi

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
# XLM-R içindeki dikkat katmanları
LORA_TARGETS = ["query", "key", "value"]


# ── Device seçimi ───────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[Device] {dev}")
    return dev


# ── InfoNCE Loss ────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Simetrik InfoNCE (CLIP orijinal losu).
    image-text ve text-image yönlerini birlikte optimize eder.
    """
    def __init__(self, temperature: float = TEMPERATURE) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds  = F.normalize(text_embeds,  dim=-1)

        logits = (image_embeds @ text_embeds.T) / self.temperature.exp().clamp(0.01, 100)
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits,   labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


# ── Model kurulumu ──────────────────────────────────────────────────────────────

def build_models(device: torch.device):
    """
    Returns:
        text_model  — XLM-R (LoRA uygulanmış, eğitilecek)
        image_model — ViT-B/32 (frozen)
        tokenizer   — XLM-R tokenizer
    """
    # Text encoder (mCLIP)
    text_model = pt_mclip.MultilingualCLIP.from_pretrained(MCLIP_MODEL)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(MCLIP_MODEL)

    # LoRA yapılandırması (XLM-R'ın linear katmanlarına)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    text_model = get_peft_model(text_model, lora_cfg)
    text_model.print_trainable_parameters()
    text_model = text_model.to(device)

    # Image encoder (frozen)
    image_model, _, _ = create_model_and_transforms("ViT-B-32", pretrained="openai")
    image_model = image_model.to(device)
    for p in image_model.parameters():
        p.requires_grad_(False)
    image_model.eval()

    return text_model, image_model, tokenizer


# ── Encode yardımcıları ─────────────────────────────────────────────────────────

def encode_images(image_model, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = image_model.encode_image(pixel_values)
    return feats.float()


def encode_texts(text_model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    mCLIP'in transformer + mean pooling + LinearTransformation adımlarını
    tokenize edilmiş girdilerle manuel olarak çalıştırır.
    (LoRA gradient akışı için from_pretrained forward yerine elle çağrıyoruz.)
    """
    out = text_model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    embs = out.last_hidden_state
    # Mean pooling (padding maskesi ağırlıklı)
    mask = attention_mask.unsqueeze(-1).float()
    embs = (embs * mask).sum(1) / mask.sum(1)
    return text_model.LinearTransformation(embs)


# ── Eğitim döngüsü ─────────────────────────────────────────────────────────────

def train_epoch(
    text_model,
    image_model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: InfoNCELoss,
    device: torch.device,
    scaler=None,
) -> float:
    text_model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        pixel_values  = batch["pixel_values"].to(device)
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                img_emb  = encode_images(image_model, pixel_values)
                txt_emb  = encode_texts(text_model, input_ids, attention_mask)
                loss     = criterion(img_emb, txt_emb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            img_emb = encode_images(image_model, pixel_values)
            txt_emb = encode_texts(text_model, input_ids, attention_mask)
            loss    = criterion(img_emb, txt_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(
    text_model,
    image_model,
    loader: DataLoader,
    criterion: InfoNCELoss,
    device: torch.device,
) -> float:
    text_model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        img_emb = encode_images(image_model, pixel_values)
        txt_emb = encode_texts(text_model, input_ids, attention_mask)
        loss    = criterion(img_emb, txt_emb)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ── Checkpoint yardımcıları ─────────────────────────────────────────────────────

def save_checkpoint(text_model, optimizer, epoch: int, loss: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"epoch_{epoch:02d}.pt"
    torch.save({
        "epoch":      epoch,
        "val_loss":   loss,
        "model_state": text_model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(path: Path, text_model, optimizer) -> int:
    ckpt  = torch.load(path, map_location="cpu")
    text_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[Checkpoint] Epoch {ckpt['epoch']} yüklendi, val_loss={ckpt['val_loss']:.4f}")
    return ckpt["epoch"]


# ── Ana fonksiyon ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = get_device()

    # Veri
    print(f"\n[Veri] {args.data} yükleniyor...")
    full_ds = EmlakDataset(args.data)

    if args.limit and args.limit < len(full_ds):
        import random; random.seed(42)
        idxs = random.sample(range(len(full_ds)), args.limit)
        full_ds = Subset(full_ds, idxs)

    train_ds, val_ds, _ = get_splits(full_ds)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    print("\n[Model] Yükleniyor...")
    text_model, image_model, _ = build_models(device)

    criterion = InfoNCELoss().to(device)
    optimizer = torch.optim.AdamW(
        list(text_model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), text_model, optimizer)

    out_dir    = Path(args.out)
    log_path   = out_dir / "training_log.csv"
    best_loss  = float("inf")
    best_path  = None

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "elapsed_s"])

    print(f"\n[Training] {args.epochs} epoch  batch={args.batch_size}  fp16={args.fp16}")
    print("=" * 60)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(text_model, image_model, train_loader, optimizer, criterion, device, scaler)
        val_loss   = eval_epoch(text_model, image_model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{elapsed:.1f}"])

        ckpt_path = save_checkpoint(text_model, optimizer, epoch, val_loss, out_dir)

        if val_loss < best_loss:
            best_loss = val_loss
            best_path = out_dir / "best.pt"
            torch.save(torch.load(ckpt_path), best_path)
            print(f"  ✓ Best model kaydedildi: {best_path}")

    print("\n" + "=" * 60)
    print(f"Eğitim tamamlandı. En iyi val_loss={best_loss:.4f}  →  {best_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mCLIP LoRA Fine-Tuning")
    p.add_argument("--data",       default="data/processed/dataset.jsonl")
    p.add_argument("--out",        default="model/checkpoints")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--workers",    type=int,   default=0,
                   help="DataLoader num_workers (Mac'te 0 önerilen)")
    p.add_argument("--fp16",       action="store_true",
                   help="Mixed precision (sadece CUDA)")
    p.add_argument("--limit",      type=int,   default=None,
                   help="Smoke-test için kullanılacak max örnek sayısı")
    p.add_argument("--resume",     default=None,
                   help="Devam edilecek checkpoint dosyası")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
