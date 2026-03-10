# mCLIP Türkçe Gayrimenkul Multimodal Arama Sistemi

**END439E — Neural Network Mod.in Ind.Sys.**  
İTÜ İşletme Fakültesi | Danışman: Prof. Dr. Muammer Altan Çakır  
**Takım:** Damla Hatice Selçuk, Elif Kamalak, Mertcan El

---

## Proje Özeti

Kullanıcıların metin (`"ferah salon"`, `"deniz manzaralı 3+1"`) veya fotoğraf kullanarak Türkçe gayrimenkul ilanlarında anlamsal arama yapabildiği multimodal (çok-kipli) bir arama sistemi.

**Model:** `M-CLIP/XLM-Roberta-Large-Vit-B-32` + LoRA fine-tuning  
**Retrieval:** ChromaDB + MMR (λ=0.5)  
**UI:** Streamlit

---

## Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Playwright tarayıcısını kur
playwright install chromium

# 3. (İlk kez) playwright-stealth
pip install playwright-stealth
```

---

## Kullanım

### Adım 1 — Veri Toplama
```bash
# Emlakjet'ten 2000 ilan topla
python scraper/playwright_scraper.py --site emlakjet --limit 2000 --out data/raw

# Her iki siteden toplam 5000
python scraper/playwright_scraper.py --site all --limit 5000 --out data/raw

# Debug modu (görsel tarayıcı)
python scraper/playwright_scraper.py --site emlakjet --limit 20 --headed
```

### Adım 2 — Veriyi Temizle
```bash
python scraper/cleaner.py --raw data/raw --out data/processed
```

### Adım 3 — Fine-Tuning (Colab / GPU)
```bash
python model/fine_tune.py --epochs 10 --batch_size 32 --out model/checkpoints
```

### Adım 4 — Indexleme
```bash
python retrieval/indexer.py --data data/processed --model model/checkpoints/best
```

### Adım 5 — Arayüzü Başlat
```bash
streamlit run ui/app.py
```

### Evaluation
```bash
python evaluation/metrics.py
python evaluation/fusion_sweep.py  # alpha sweep
```

---

## Proje Yapısı

```
├── scraper/
│   ├── playwright_scraper.py   # Playwright stealth scraper
│   └── cleaner.py              # Görsel & metin temizleme
├── model/
│   ├── dataset.py              # PyTorch Dataset
│   └── fine_tune.py            # LoRA fine-tuning
├── retrieval/
│   ├── indexer.py              # ChromaDB indexer
│   └── retriever.py            # MMR retrieval
├── ui/
│   └── app.py                  # Streamlit UI
├── evaluation/
│   ├── metrics.py              # Recall@K, MRR
│   └── fusion_sweep.py         # α sweep
├── chatbot/                    # [Opsiyonel] RAG chatbot
│   └── rag_bot.py
├── data/
│   ├── raw/                    # Ham scraping verisi
│   └── processed/              # Temizlenmiş dataset
└── requirements.txt
```

---

## Değerlendirme Metrikleri

| Metrik | Açıklama |
|---|---|
| Recall@K | K=1,5,10 — doğru ilan ilk K sonuçta var mı? |
| MRR | İlk doğru sonucun ortalama sıralaması |
| α sweep | Late fusion ağırlık optimizasyonu |
