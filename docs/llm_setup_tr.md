# LLM Shootout Kurulum Rehberi

Bu adımda amaç GPT-5.5 veya Claude Opus kullanmadan, plandaki feasible adayları gerçek sorgularla karşılaştırmak:

- DeepSeek V4 Flash
- Kimi K2.6
- Gemini 3.5 Flash
- GLM-4.6
- Gemma 4 local

## 1. Ortam dosyasını hazırla

`.env.example` dosyasını referans alıp proje kökünde `.env` oluştur. `.env` git'e eklenmez.

```bash
cp .env.example .env
```

Sonra erişebildiğin servislerin key'lerini doldur:

```dotenv
DEEPSEEK_API_KEY=...
MOONSHOT_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4
```

## 2. En kolay API yolu

İlk kez kullanıyorsan pratik sıra:

1. Gemini API key al ve `GEMINI_API_KEY` doldur.
2. OpenRouter API key al ve `OPENROUTER_API_KEY` doldur; GLM-4.6 bu yoldan çağrılır.
3. DeepSeek key al ve `DEEPSEEK_API_KEY` doldur.
4. Kimi/Moonshot erişimin varsa `MOONSHOT_API_KEY` doldur.

Tüm key'ler hazır değilse harness ilgili modeli `missing_env:*` olarak işaretler; çalışan modellerin skorunu yine kaydeder. Ancak final winner kararı için en az bir text ve bir vision adayı çalışmalı.

## 3. Local Gemma / Ollama yolu

API kullanmak istemiyorsan Ollama kurup local Gemma modelini çalıştırabilirsin.

1. Ollama'yı kur (macOS: `brew install ollama` veya ollama.com).
2. Yeni terminal aç.
3. Kurulu modeli kontrol et:

```bash
ollama list
```

4. Planlanan Gemma model tag'i neyse onu indir:

```bash
ollama pull <gemma-model-tag>
```

5. `.env` içinde `OLLAMA_MODEL=<gemma-model-tag>` yap.

## 4. Shootout'u çalıştır

Bağımlılıkları kur:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Shootout:

```bash
python3 -m llm.shootout --rows-out llm/shootout_rows.json --out llm/selected.json
```

Başarılı koşuda iki dosya oluşur:

- `llm/shootout_rows.json`: tüm adayların skor ve maliyet satırları
- `llm/selected.json`: seçilen text ve vision modelleri

> **Not (2026-05-30):** Bu rehber M1 slot shootout kurulumu içindir ve **tamamlandı**. Güncel seçim `llm/selected.json`'da: `text_model=deepseek_v4_pro` (gold A/B revize), `vision_model=kimi_k2_6`. M3 labeling zaten başlatılabilir durumda (iskele hazır). Aday listesine sonradan `deepseek_v4_pro` ve `qwen3_vl_local` (yerel vision) eklendi.
