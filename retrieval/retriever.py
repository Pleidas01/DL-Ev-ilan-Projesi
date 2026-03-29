"""
MMR Retriever
=============
ChromaDB üzerinden metin veya görsel sorgusuyla ilan arar.
Maximal Marginal Relevance (MMR) ile çeşitlilik sağlar.

Kullanım:
    from retrieval.retriever import Retriever
    r = Retriever()
    results = r.query_text("deniz manzaralı 2+1 daire", k=5)
    results = r.query_image("foto.jpg", k=5)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import chromadb
import multilingual_clip.pt_multilingual_clip as pt_mclip
import transformers
from open_clip import create_model_and_transforms


MCLIP_MODEL     = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
COLLECTION_NAME = "emlak_listings"
CHROMA_DIR      = "data/chroma"
DEFAULT_K       = 10
MMR_LAMBDA      = 0.5   # relevance vs diversity tradeoff (0=çeşitlik, 1=alaka)
FETCH_MULT      = 4     # MMR için k * FETCH_MULT aday getirilir


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Retriever:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        checkpoint: str | None = None,
    ) -> None:
        self.device = get_device()

        # Modeller
        self.text_model = pt_mclip.MultilingualCLIP.from_pretrained(MCLIP_MODEL)
        self.tokenizer  = transformers.AutoTokenizer.from_pretrained(MCLIP_MODEL)

        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu")
            self.text_model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)

        # XLM-R MPS'te bazı op'ları desteklemiyor — text encoder CPU'da
        self.text_device = torch.device("cpu")
        self.text_model = self.text_model.to(self.text_device).eval()

        self.image_model, _, self.preprocess = create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.image_model = self.image_model.to(self.device).eval()

        # ChromaDB
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(COLLECTION_NAME)

    # ── Encode yardımcıları ──────────────────────────────────────────────────

    def _encode_text(self, query: str) -> list[float]:
        # mCLIP forward(txt, tokenizer) — raw metin + tokenizer alır
        with torch.no_grad():
            emb = self.text_model([query], self.tokenizer)
        return F.normalize(emb.float(), dim=-1).squeeze(0).cpu().tolist()

    def _encode_image(self, image: Image.Image | str | Path) -> list[float]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.image_model.encode_image(tensor)
        return F.normalize(emb.float(), dim=-1).squeeze(0).cpu().tolist()

    def _fuse(self, text_emb: list[float], image_emb: list[float], alpha: float = 0.5) -> list[float]:
        """Late fusion: alpha * text + (1-alpha) * image, normalize."""
        t = torch.tensor(text_emb)
        i = torch.tensor(image_emb)
        fused = alpha * t + (1 - alpha) * i
        return F.normalize(fused, dim=-1).tolist()

    # ── MMR ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _mmr(
        query_emb: list[float],
        candidates: list[dict],
        k: int,
        lam: float,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance seçimi.
        candidates: [{"embedding": [...], "metadata": {...}, "distance": float}, ...]
        """
        if len(candidates) <= k:
            return candidates

        import numpy as np
        q    = torch.tensor(query_emb, dtype=torch.float32)
        embs = torch.tensor(np.array([c["embedding"] for c in candidates]), dtype=torch.float32)

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected) < k and remaining:
            best_idx, best_score = None, -float("inf")
            for idx in remaining:
                relevance  = float(q @ embs[idx])
                if selected:
                    sim_to_sel = float((embs[selected] @ embs[idx]).max())
                else:
                    sim_to_sel = 0.0
                score = lam * relevance - (1 - lam) * sim_to_sel
                if score > best_score:
                    best_score, best_idx = score, idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    # ── Public API ───────────────────────────────────────────────────────────

    def query_text(
        self,
        text: str,
        k: int = DEFAULT_K,
        lam: float = MMR_LAMBDA,
    ) -> list[dict]:
        """Metin sorgusuyla ilan arar."""
        query_emb = self._encode_text(text)
        return self._search(query_emb, k, lam)

    def query_image(
        self,
        image: Image.Image | str | Path,
        k: int = DEFAULT_K,
        lam: float = MMR_LAMBDA,
    ) -> list[dict]:
        """Görsel sorgusuyla ilan arar."""
        query_emb = self._encode_image(image)
        return self._search(query_emb, k, lam)

    def query_multimodal(
        self,
        text: str,
        image: Image.Image | str | Path,
        k: int = DEFAULT_K,
        lam: float = MMR_LAMBDA,
        alpha: float = 0.5,
    ) -> list[dict]:
        """Metin + görsel birlikte sorgu."""
        text_emb  = self._encode_text(text)
        image_emb = self._encode_image(image)
        query_emb = self._fuse(text_emb, image_emb, alpha)
        return self._search(query_emb, k, lam)

    def _search(self, query_emb: list[float], k: int, lam: float) -> list[dict]:
        n_fetch = min(k * FETCH_MULT, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=max(n_fetch, k),
            include=["metadatas", "distances", "embeddings"],
        )

        candidates = []
        for meta, dist, emb in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["embeddings"][0],
        ):
            candidates.append({
                "metadata":  meta,
                "score":     1 - dist,   # cosine similarity
                "embedding": emb,
            })

        return self._mmr(query_emb, candidates, k, lam)


# ── Hızlı CLI testi ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "denize yakın daire"
    print(f"Sorgu: '{query}'")
    r = Retriever()
    results = r.query_text(query, k=5)

    print(f"\nTop-{len(results)} sonuçlar:")
    for i, res in enumerate(results, 1):
        m = res["metadata"]
        print(f"  {i}. [{m['id']}] {m['title'][:50]}  |  {m['price']}  |  score={res['score']:.3f}")
