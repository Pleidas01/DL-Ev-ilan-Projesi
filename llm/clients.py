from __future__ import annotations

import base64
from io import BytesIO
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Modality = Literal["text", "vision"]
# Düşük çözünürlük = küçük payload + az image token → 20+ fotolu ilanlarda
# Moonshot tek-çağrı timeout'unu azaltır. Env ile tune edilebilir (1024/1280/1536).
VISION_MAX_IMAGE_EDGE = int(os.getenv("VISION_MAX_IMAGE_EDGE", "1024"))
VISION_JPEG_QUALITY = int(os.getenv("VISION_JPEG_QUALITY", "85"))


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    display_name: str
    provider: str
    model_name: str
    modalities: tuple[Modality, ...]
    input_usd_per_million: float
    output_usd_per_million: float
    env_keys: tuple[str, ...] = ()
    base_url: str | None = None
    local: bool = False

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.modalities


CANDIDATES: tuple[ModelCandidate, ...] = (
    ModelCandidate(
        id="deepseek_v4_flash",
        display_name="DeepSeek V4 Flash",
        provider="deepseek",
        model_name="deepseek-v4-flash",
        modalities=("text",),
        input_usd_per_million=0.14,
        output_usd_per_million=0.28,
        env_keys=("DEEPSEEK_API_KEY",),
        base_url="https://api.deepseek.com",
    ),
    ModelCandidate(
        id="deepseek_v4_pro",
        display_name="DeepSeek V4 Pro",
        provider="deepseek",
        model_name="deepseek-v4-pro",
        modalities=("text",),
        input_usd_per_million=0.435,  # %75 indirimli cache-miss (normal $1.74); cache-hit $0.003625
        output_usd_per_million=0.87,  # %75 indirimli (normal $3.48)
        env_keys=("DEEPSEEK_API_KEY",),
        base_url="https://api.deepseek.com",
    ),
    ModelCandidate(
        id="kimi_k2_6",
        display_name="Kimi K2.6",
        provider="moonshot",
        model_name="kimi-k2.6",
        modalities=("text", "vision"),
        input_usd_per_million=0.95,
        output_usd_per_million=4.00,
        env_keys=("MOONSHOT_API_KEY",),
        base_url="https://api.moonshot.ai/v1",
    ),
    ModelCandidate(
        id="gemini_3_5_flash",
        display_name="Gemini 3.5 Flash",
        provider="gemini",
        model_name="gemini-3.5-flash",
        modalities=("text", "vision"),
        input_usd_per_million=1.50,
        output_usd_per_million=9.00,
        env_keys=("GEMINI_API_KEY",),
    ),
    ModelCandidate(
        id="glm_4_6",
        display_name="GLM-4.6",
        provider="openrouter",
        model_name="z-ai/glm-4.6",
        modalities=("text",),
        input_usd_per_million=0.30,
        output_usd_per_million=0.50,
        env_keys=("OPENROUTER_API_KEY",),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    ),
    ModelCandidate(
        id="gemma_4_local",
        display_name="Gemma 4 local",
        provider="ollama",
        model_name=os.getenv("OLLAMA_MODEL", "gemma4"),
        modalities=("text", "vision"),
        input_usd_per_million=0.0,
        output_usd_per_million=0.0,
        env_keys=("OLLAMA_HOST",),
        local=True,
    ),
    ModelCandidate(
        id="qwen3_vl_local",
        display_name="Qwen3-VL 8B local",
        provider="ollama",
        model_name=os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b"),
        modalities=("text", "vision"),
        input_usd_per_million=0.0,
        output_usd_per_million=0.0,
        env_keys=("OLLAMA_HOST",),
        local=True,
    ),
)


def candidate_by_id(model_id: str) -> ModelCandidate:
    for candidate in CANDIDATES:
        if candidate.id == model_id:
            return candidate
    raise KeyError(f"Unknown model candidate: {model_id}")


def estimate_cost_usd(candidate: ModelCandidate, input_tokens: int, output_tokens: int) -> float:
    if candidate.local:
        return 0.0
    return (
        input_tokens / 1_000_000 * candidate.input_usd_per_million
        + output_tokens / 1_000_000 * candidate.output_usd_per_million
    )


def _openai_chat_temperature(candidate: ModelCandidate) -> float:
    """Moonshot Kimi K2.x rejects temperature=0; API allows only 1."""
    if candidate.provider == "moonshot":
        return 1.0
    return 0.0


def missing_environment(candidate: ModelCandidate) -> list[str]:
    missing = []
    for key in candidate.env_keys:
        if key == "OLLAMA_HOST":
            continue
        if not os.getenv(key):
            missing.append(key)
    return missing


def complete_json(candidate: ModelCandidate, system_prompt: str, user_prompt: str) -> str:
    """Call a candidate with a JSON-only text prompt.

    The method is intentionally thin: provider routing belongs here, while
    benchmark scoring stays deterministic in `llm.shootout`.
    """
    if candidate.provider in {"deepseek", "moonshot", "openrouter"}:
        from openai import OpenAI

        api_key = os.environ[candidate.env_keys[0]]
        client = OpenAI(api_key=api_key, base_url=candidate.base_url, timeout=120)
        response = client.chat.completions.create(
            model=candidate.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=_openai_chat_temperature(candidate),
        )
        return response.choices[0].message.content or "{}"

    if candidate.provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=candidate.model_name,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        return response.text or "{}"

    if candidate.provider == "ollama":
        from ollama import Client

        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        response = client.chat(
            model=candidate.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0},
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content") or "{}"
        message = getattr(response, "message", None)
        return (getattr(message, "content", None) if message is not None else None) or "{}"

    raise ValueError(f"Unsupported provider: {candidate.provider}")


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((VISION_MAX_IMAGE_EDGE, VISION_MAX_IMAGE_EDGE))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        pass
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def complete_vision_json(
    candidate: ModelCandidate,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str],
) -> str:
    """Call a vision-capable candidate with one or more local images, JSON output.

    Tüm fotoğraflar tek istekte gider; model tek bütünleşik JSON döndürür
    (per-image aggregate gerekmez). Tek elemanlı liste de çalışır.
    """
    if not candidate.supports_vision:
        raise ValueError(f"Candidate {candidate.id} does not support vision")
    if not image_paths:
        raise ValueError("complete_vision_json requires at least one image path")

    if candidate.provider in {"moonshot", "openrouter"}:
        from openai import OpenAI

        api_key = os.environ[candidate.env_keys[0]]
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})
        # timeout: multi-image tek çağrı daha büyük → sonsuz beklememek için 120s.
        client = OpenAI(api_key=api_key, base_url=candidate.base_url, timeout=120)
        response = client.chat.completions.create(
            model=candidate.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=_openai_chat_temperature(candidate),
        )
        return response.choices[0].message.content or "{}"

    if candidate.provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        contents: list[Any] = []
        for image_path in image_paths:
            image_bytes = Path(image_path).read_bytes()
            mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(f"{system_prompt}\n\n{user_prompt}")
        response = client.models.generate_content(
            model=candidate.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        return response.text or "{}"

    if candidate.provider == "ollama":
        from ollama import Client

        # Ollama "images" SAF base64 listesi bekler (data URL prefix'i DEĞİL) — aksi
        # halde "illegal base64 data at input byte 4" hatası verir.
        encoded = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in image_paths]
        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        response = client.chat(
            model=candidate.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": encoded},
            ],
            format="json",
            options={"temperature": 0},
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content") or "{}"
        message = getattr(response, "message", None)
        return (getattr(message, "content", None) if message is not None else None) or "{}"

    raise ValueError(f"Unsupported vision provider: {candidate.provider}")
