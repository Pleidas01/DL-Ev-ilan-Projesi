# Sunum İçeriği (~20-30 dk, HTML deck için)

> Hoca AI kullandığımızı biliyor → mesaj: **AI'yı anlayarak ve efektif kullandık.** Her adımda
> "neyi NASIL yaptık + NEDEN öyle + ÖLÇTÜĞÜMÜZ sayı." Veri toplama ve performans metrikleri
> en az eğittiğimiz NN kadar değerli. Çoğunu Mertcan anlatır (akıcı tek-anlatıcı tonu).
> İşaretler: [SLAYT]=ekranda ne, [ANLAT]=konuşma metni.

---

## 0. Açılış (1 dk)
[SLAYT] Başlık + takım (Damla, Elif, Mertcan). Tek cümle: "Türkçe serbest-metin emlak arama; image-text problemini text-text'e çevirdik, uçtan uca kendi sistemimizi kurduk ve kendi embedding NN'imizi eğitip ölçtük."
[ANLAT] İki şey göstereceğiz: (1) baştan sona çalışan, ölçülmüş bir sistem, (2) kendi eğittiğimiz + ölçtüğümüz NN. AI'yı bol kullandık ama her kararı biz verdik, her parçayı biz tasarladık.

## 1. Problem (2 dk)
[SLAYT] Örnek sorgu: *"Kadıköy'de 30 bin altı, geniş salonlu, asansörlü, denize yakın 2+1 öğrenciye uygun daire."*
[ANLAT] Sorguda iki katman var: (a) Emlakjet panelinde karşılığı olan **net istekler → hard filter** (ilçe, oda, max fiyat, asansör, denize yakınlık); (b) registry dışı **serbest niteleme → embedding/anlamsal sıralama** ("geniş salonlu", "öğrenciye uygun"). Çıktı: ilgili ilanlar + **neden eşleştiği**. Bu ikiliyi baştan ayırdık ve ayrı ayrı ölçtük.

## 2. İlk başarısızlık → PIVOT (2-3 dk)  *(yöntem anlayışımızın kanıtı)*
[SLAYT] Üç nesil: Gen1 CLIP (terk) → Gen2 el-yapımı şema (terk) → Gen3 canonical registry (aktif).
[ANLAT] İlk yaklaşım **M-CLIP fine-tuning** (HW6) idi → **R@10 = 0.037**, çöktü. *Neden* çöktüğünü çözmek önemliydi:
- **Çoklu-foto problemi**: 1 ilan = 5-15 foto. CLIP tek görsel embedding'e bunu nasıl sığdırır?
- **Modality gap**: Türkçe metin ↔ ev fotoğrafı semantik mesafesi büyük; contrastive uzayda hizalama zayıf.
**Pivot içgörüsü**: VLM/LLM ile fotoğrafları yapılandırılmış JSON özelliklere çevir → **image-text retrieval'ı text-text retrieval'a indirge.** Böylece olgun, güçlü çok-dilli metin embedding'leri kullanırız. Projenin kalbi: başarısızlığı anlayıp mimariyi değiştirmek.

## 3. ⭐ Veri toplama (4-5 dk)  *(en az NN kadar değerli — detaylı)*
[SLAYT] Boru hattı: `Playwright scraper → görsel indirici → cleaner → temiz dataset`.
[ANLAT — scraper] `scraper/playwright_scraper.py`, **Playwright + Chromium** (headless). Emlakjet'in bot koruması var → **stealth**: `navigator.webdriver` siliniyor, gerçekçi user-agent rotasyonu, tr-TR locale/timezone, ek HTTP header'lar; **Cloudflare challenge** sayfası tespit edilip bekleniyor. Sayfa yapısından çıkarım: "İlan Bilgileri" tablosu, "İlan Özellikleri"nin **3 sekmesi** (İç/Dış/Konum — lazy render, tıklanıp toplanıyor), "Açıklaması" + "Daha Fazla Gör" expand. **Capture-all**: tüm satırlar kayıpsız alınır (whitelist yok → balkon gibi alanlar sessizce düşmesin). **Resume + dedup**: mevcut ilan ID'leri `seen_ids`'e yüklenir, tekrar çekilmez.
[ANLAT — indirici] `image_downloader.py` asenkron (aiohttp), retry + üstel backoff, idempotent (indirilmişi atla). Önemli detay: bir ilana ait görsel, dosya adında **kendi ID'sini** taşımalı — Emlakjet'in "benzer ilanlar" thumbnail'leri farklı ID içeriyor, onları eliyoruz.
[ANLAT — cleaner] `cleaner.py`: görsel doğrulama (min boyut/format), **duplicate eleme** (URL hash + görsel MD5), Türkçe→ASCII normalize (lower'dan ÖNCE, "İ".lower() bug'ı), fiyat/int parse, ısıtma enum normalize, konum ayrıştırma (şehir/ilçe/mahalle).
[SLAYT — sayılar] 305 ilk → **gece +500 scrape** → 805 ham → cleaner 743 geçerli → ~13.4k foto, **0 indirme hatası**. Final etiketli: **716 ilan**. (Gece scrape'inde satılık karıştı → demo kiralık-scope 356; bölüm 9'da anlatılacak ders.)

## 4. ⭐ Etiketleme = AI'yı anlayarak kullanma (5-6 dk)  *(detaylı)*
[SLAYT] Kanıt zinciri: `scraper fact (deterministik) → DeepSeek (yalnız metin, null-only) → Kimi (yalnız görsel, positive-only)`.
[ANLAT — neden böyle] **Modaliteleri ayırdık**: scraper "İlan Bilgileri/Özellikleri"ni deterministik yazar; **DeepSeek yalnız başlık+açıklama** okuyup kalan null alanları açık metin kanıtıyla doldurur; **Kimi yalnız fotoğrafları** okuyup kalan null boolean'lara yalnız açık görsel kanıt varsa `true` ekler. **Monotonik dolum**: önceki kaynak ezilmez. Böylece her değerin kaynağı bellidir, karışım yok.
[SLAYT — API mühendisliği] `llm/clients.py` — tek `ModelCandidate` soyutlaması (provider, model, fiyat/M token, env key, base_url).
[ANLAT — API bağlamaları] 
- **Tek SDK, çok provider**: DeepSeek, Moonshot(Kimi), OpenRouter hepsi **OpenAI-uyumlu** → aynı `OpenAI` SDK, sadece `base_url`+key değişiyor. Gemini `google-genai`, yerel Gemma/Qwen `ollama`.
- **Yapılandırılmış çıktı**: `response_format={"type":"json_object"}` → model her zaman geçerli JSON döner (parse garantisi).
- **Öğrendiğimiz quirk'ler**: Moonshot `temperature=0`'ı reddediyor → Kimi'ye 1.0, diğerlerine 0 veriyoruz. Vision'da Ollama **saf base64** ister (data-URL prefix'i değil) yoksa "illegal base64" hatası.
- **Görsel ön-işleme**: PIL ile EXIF-düzelt → **512px'e küçült** (payload+token azalır, timeout düşer) → JPEG q85 → base64 data-URL. **Multi-image tek çağrı**: tüm fotolar tek istekte, model tek bütünleşik JSON (per-image aggregate gerekmez → ~3x ucuz).
- **Maliyet + dayanıklılık**: token bazlı maliyet tahmini + **cost-cap** (aşınca dur); transient timeout retry; paralel batch'te tek ilan hata verirse diğerleri kaydedilir, başarısız ID'ler raporlanır, `--resume` yalnız eksikleri işler.
[SLAYT — model seçimi METRİKLE] Tahmin değil, ölçüp seçtik (shootout):
| Karar | Nasıl ölçtük | Sonuç |
|---|---|---|
| Text modeli | gold A/B (16 ilan + 30 sorgu) | **DeepSeek** slot 0.815 > Kimi 0.792, **~3.5x ucuz, ~5x hızlı** |
| Vision modeli | 10 ilan gold | **Kimi multi-image**; ortak ilanlarda 0.931 ≥ per-image 0.917, **~3x ucuz** |
| Yerel alternatif | Gemma4/Qwen3-VL denendi | Gemma zayıf (0.135); Qwen 16GB'ı darladı → future work |
[ANLAT] Yani "en pahalıyı seçtik" değil; **kalite/maliyet ölçüp** text'i ucuz+hızlı DeepSeek'e, vision'ı en doğru Kimi'ye verdik.

## 5. Indexing & sorgu akışı (3-4 dk)  *(detaylı)*
[SLAYT] `enriched_doc → BGE-M3 embedding → ChromaDB`. Sorgu: `slot extraction → hard filter → dense → rerank → çip → RAG`.
[ANLAT — index] Her ilan için **enriched_doc** (embedding metni) + **canonical metadata** (skalar slug'lar + multi-enum için seçenek başına exact-flag, örn. `balcony_type__acik_balkon`). `null` metadata yazılmaz → explicit filtreyle eşleşmez. **BGE-M3** ile 1024-dim vektör, ChromaDB persistent collection (**HNSW** yaklaşık en-yakın-komşu, cosine).
[ANLAT — sorgu] (1) **DeepSeek slot extraction**: serbest Türkçe → canonical JSON (registry şeması + enum değer ipuçları few-shot ile). (2) **Hard filter**: metadata eşleme; ilçe Türkçe→ASCII fold ("Kadıköy"→"Kadikoy"), kiralık-scope, yaş aralığı listesi. (3) **BGE-M3 dense** retrieval (cosine + HNSW). (4) **Cross-encoder rerank** (bge-reranker-v2-m3). (5) **Match çipleri** — deterministik (kod cevaplıyor, LLM'e bırakılmaz; çip asla karşılanmamış filtreyi iddia etmez). (6) **RAG**: yalnız retrieved ilanlardan kısa cevap (grounded).
[SLAYT — bi vs cross] Bi-encoder (BGE-M3): sorgu/ilan AYRI kodlanır → ilan vektörleri önceden hesaplı, **hızlı** → aday daralt. Cross-encoder (reranker): (sorgu,ilan) BİRLİKTE → **doğru ama yavaş** → sadece top-N. Klasik iki-aşamalı retrieval.

## 6. Kendi eğittiğimiz NN: BGE-M3 + LoRA (4-5 dk)  *(sade ama savunulabilir; detay çalışma kağıdında)*
[SLAYT] Neden + ne + sonuç.
[ANLAT — neden] BGE-M3 genel amaçlı çok-dilli bir embedding modeli; biz onu **Türkçe emlak alanına uyarladık** (domain adaptation) — kendi NN'imiz.
[ANLAT — LoRA sezgisi] Koca modeli baştan eğitmek yerine: **base ağırlıkları DONDUR, yanına küçük bir "adapter" ekle ve sadece onu eğit.** Adapter = düşük-rank iki küçük matris (B·A). Sadece **~%0.2 parametre** eğitilir (4.7MB adapter). **Backprop**: kayıp gradyanı tüm ağdan geçer ama **yalnız adapter'ın ağırlıkları güncellenir** (donuk base sabit kalır). Avantaj: hızlı (dakikalar), küçük veride overfit riski düşük, base model "unutmaz".
[ANLAT — nasıl öğretiyoruz: triplet loss] Eğitim verisi **üçlüler**: (sorgu, doğru ilan = pozitif, farklı ilan = negatif). Hedef: sorguyu kendi ilanına **yaklaştır**, farklı ilandan bir **margin** kadar **uzaklaştır**. Formül: `L = max(0, mesafe(sorgu,pozitif) − mesafe(sorgu,negatif) + margin)`.
[SLAYT — sonuç] Held-out (143 sorgu, train/val listing **sızıntısı=0**), dense-only:
| Metrik | Baseline | + LoRA | Δ |
|---|---:|---:|---:|
| R@1 | 0.720 | **0.965** | **+0.245** |
| MRR | 0.832 | **0.981** | **+0.149** |
| R@5 / R@10 | 0.958 / 0.993 | 1.000 / 1.000 | +0.04 / +0.01 |
[ANLAT] *Neden iyileşti*: adapter Türkçe emlak sorgu↔ilan hizalamasını öğrendi → benzer ilanları ayırt etme bariz arttı (R@1 +24.5 puan). **Gerçekten eğittik, ölçtük, sızıntısız.**

## 7. ⭐ Performans metrikleri (3 dk)  *(öne çıkan — konsolide)*
[SLAYT] Tek tabloda her şey:
| Aşama | Metrik | Sonuç |
|---|---|---|
| Veri | toplanan / temiz / etiketli | 805 / 743 / **716** ilan, ~13.4k foto |
| Slot extraction (text) | gold A/B kalite | DeepSeek 0.815 (Kimi 0.792) |
| Vision etiketleme | gold accuracy | Kimi multi 0.931 (ortak ilan) |
| Retrieval (NN) | held-out R@1 / MRR | **0.72→0.97 / 0.83→0.98** (LoRA) |
| Yazılım kalitesi | test paketi | **118 passed**, skip yok |
| Maliyet | toplam API | ~**$5-6** ( $100 bütçenin çok altında) |
| Hız | ilan başı | ~4s scrape, ~$0.005 etiketleme |
[ANLAT — feasibility] İlan başı sabit maliyet + paralelleştirme + iki-aşamalı retrieval (HNSW) → **yüz binlerce ilana lineer ölçeklenir.** Hocanın "scale eder mi" sorusunun cevabı bu sayılarda.

## 8. UI & canlı demo (2-3 dk)
[SLAYT] Streamlit; çalışan sorgular.
[ANLAT] `ui/app.py` — chat kutusu + ilan kartları (kapak foto + başlık + fiyat + **match çipleri** + canonical detaylar) + 3 demo butonu. Canlı göster:
- `Üsküdar'da 100 m² üstü doğalgazlı 2+1` → 5 sonuç, çip "Kombi Doğalgaz".
- `balkonlu eşyalı 1+1 daire` → 5 sonuç (skor 0.85-0.97).
- `geniş salonlu denize yakın 2+1 asansörlü` → çip "2+1 / Asansör / Denize Yakın".
Her kartta **neden uygun** çipleri (deterministik), RAG cevabı yalnız retrieved ilanlardan (halüsinasyon yok).

## 9. Zorluklar & optimizasyonlar (3 dk)  *(emek + mühendislik)*
[SLAYT] Zorluk → çözüm:
- CLIP çöküşü (R@10=0.037) → text-text pivot.
- **Türkçe karakter bug'ı**: slot "Kadıköy", index "Kadikoy" → hard filter sessizce 0. **Çözüm**: query-zamanı deterministik ASCII-fold (deterministik dönüşüm koda, modele değil).
- **Veri kalitesi**: gece scrape'i `--tip kiralik`'siz koşunca satılık karıştı (716'nın 360'ı). **Çözüm + ders**: retriever'a default `trade_type=kiralik` scope; scrape filtresi unutulmamalı.
- **Slot enum uyuşmazlığı**: LLM "natural_gas"/"0-20" üretip filtre sessizce düşüyordu. **Çözüm**: prompt'a geçerli enum değerleri + "aralığı listeye genişlet".
- **VLM timeout**: Kimi 27 ilanda timeout → dayanıklı resume + cost-cap + foto-parçalı fallback.
[SLAYT] Optimizasyonlar: 512px görsel, multi-image tek çağrı (3x ucuz), cost-cap'li paralel batch, hard-filter→vektör→rerank, HNSW, **CUDA** (Blackwell cu128), LoRA (full fine-tune yerine %0.2 param).
[ANLAT] Her bug'ı **dosyadan doğrulayıp test yazarak** çözdük; "çalışıyor sandık" değil, kanıtladık.

## 10. Dürüst sınırlılıklar & future work (2 dk)  *(rigor = güven)*
[SLAYT] Liste.
[ANLAT] 716 demo ölçeği, üretim değil. Eval **sentetik sorgularla** (ilan alanlarından) — gerçek-kullanıcı relevansını insan-gold ile ölçmedik. LoRA'da eğitim mean-pool, eval CLS-pool (küçük tutarsızlık; ama baseline+LoRA aynı eval kullandığı için **karşılaştırma adil**). Future work: gerçek relevance gold; yerel VLM (qwen3) ile maliyetsiz ölçek; daha büyük dataset; image classifier (ResNet/ViT) ile VLM karşılaştırması.
[SLAYT — kapanış] CLIP'i anlayıp bıraktık → text-text pivot → uçtan uca ölçülmüş sistem → kendi LoRA NN'imiz R@1 0.72→0.97. AI'yı çok kullandık ama **anlayarak ve ölçerek**.

---

## Notlar
- **Sunan**: çoğunlukla Mertcan (akıcı anlatım). Damla/Elif: demo + seçili bölümler (esnek).
- **Çalışma kağıdı (sonra)**: bölüm 6 (LoRA/triplet/backprop) için derinleştirilmiş Q&A — birkaç saatte anlaşılır seviye.
- **Olası hoca soruları**: backprop LoRA'da nasıl (donuk base, sadece adapter güncellenir); neden triplet/cosine; overfit (%0.2 param, sızıntısız split); pooling (mean vs CLS, karşılaştırma adil); scale eder mi (metrikler bölüm 7).
