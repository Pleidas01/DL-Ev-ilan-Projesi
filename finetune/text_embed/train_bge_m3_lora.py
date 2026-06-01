from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_TRAIN = Path("finetune/text_embed/data/train.jsonl")
DEFAULT_OUTPUT = Path("finetune/text_embed/checkpoints/bge_m3_lora")


def require_cuda(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA training; move this command to the RTX machine")


def validate_target_modules(model: Any, target_modules: list[str]) -> dict[str, list[str]]:
    module_names = [name for name, _module in model.named_modules()]
    matches = {
        target: [name for name in module_names if name == target or name.endswith(f".{target}")]
        for target in target_modules
    }
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise ValueError(f"LoRA target modules not found: {', '.join(missing)}")
    return matches


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_pool(torch_module: Any, output: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size()).float()
    return (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _encode(torch_module: Any, model: Any, tokenizer: Any, texts: list[str], max_length: int) -> Any:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to("cuda")
    embeddings = _mean_pool(torch_module, model(**batch), batch["attention_mask"])
    return torch_module.nn.functional.normalize(embeddings, p=2, dim=1)


def inspect_target_modules(model_name: str) -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    for name, _module in model.named_modules():
        if name.rsplit(".", 1)[-1] in {"query", "key", "value"}:
            print(name)


def train(args: argparse.Namespace) -> None:
    import torch

    require_cuda(torch)

    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    base_model = AutoModel.from_pretrained(args.model)
    validate_target_modules(base_model, args.target_modules)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        ),
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = _load_jsonl(args.train)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()
    try:
        for _epoch in range(args.epochs):
            for step, batch in enumerate(loader, start=1):
                with torch.amp.autocast("cuda"):
                    query = _encode(torch, model, tokenizer, batch["query"], args.max_length)
                    positive = _encode(torch, model, tokenizer, batch["positive"], args.max_length)
                    negative = _encode(torch, model, tokenizer, batch["negative"], args.max_length)
                    loss = torch.nn.functional.triplet_margin_loss(query, positive, negative)
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("CUDA OOM: rerun explicitly with a smaller --batch-size") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an adapter-only BGE-M3 LoRA model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inspect-target-modules", action="store_true")
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.inspect_target_modules:
        inspect_target_modules(args.model)
        return
    if not args.target_modules:
        parser.error("--target-modules is required after inspection")
    train(args)


if __name__ == "__main__":
    main()
