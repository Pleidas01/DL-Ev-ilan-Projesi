# MİMARİ EVRİMİ — Üç Nesil, Hangi Fikri Neden Terk Ettik

> Bu doküman projenin **karar tarihçesini** anlatır: hangi mimariyi denedik, neden bıraktık, şu an neden bu yoldayız. Sunumda "projeyi anladığımızı" göstermek için kullanılır. Güncel teknik durum için `STATUS.md`, sıradaki iş için `HANDOFF.md`, mimari referans için `PROJECT.md`.

Proje üç farklı mimari nesilden geçti. Her geçiş bir öncekinin somut bir sınırını çözmek için yapıldı.

---

## Gen 1 — CLIP (görsel-metin ortak uzay) — **TERK EDİLDİ**

**Ne denedik:** HW6'da M-CLIP (`M-CLIP/XLM-Roberta-Large-Vit-B-32`) fine-tuning. Fikir: ilan fotoğrafı ile Türkçe metni aynı embedding uzayına koyup doğrudan görsel↔metin arama (image-text retrieval) yapmak. Kod: `END439E_HW6.ipynb`, `archive/`.

**Neden terk ettik (iki somut sorun):**
1. **Çoklu-foto problemi:** Bir ilanın 5-15 fotoğrafı var (salon, mutfak, balkon, cephe...). CLIP tek bir ilanı tek bir görsel embedding'e nasıl indirgesin? Hangi foto ilanı "temsil ediyor"?
2. **Modality gap:** Türkçe ilan metni ile ev fotoğrafı arasındaki semantik mesafe büyük; fine-tune sonrası bile **R@10 = 0.037** ile pratikte kullanılamaz çıktı.

**Dönüm noktası:** Ders asistanının önerisi — *"LLM/VLM ile fotoğrafları JSON özelliklere sınıflandır, görsel-metin aramayı metin-metin aramaya çevir."* Bu öneri Gen 2'yi başlattı.

---

## Gen 2 — El-yapımı JSON şema (LLM/VLM etiketleme) — **TARİHSEL**

**Ne denedik:** TA önerisini uyguladık. Her ilanı LLM/VLM ile yapısal JSON'a çevirip **metin-metin** aramaya geçtik. Şema **elle tasarlandı**:
- `facts_gold` (21 alan): oda sayısı, ısıtma, balkon, asansör, otopark, metro yakınlığı vb.
- `visual_gold` (6-7 alan): `balkon_ozellikleri`, `manzara`, `mutfak_tipi`, `banyo_ozellikleri`, `imkanlar`.

Retrieval mekanizması bugünküyle aynı paradigmadaydı: BGE-M3 embedding → ChromaDB metadata filtre → CrossEncoder reranker.

**Neden terk ettik (üç somut sorun):**
1. **Modalite karışıklığı:** Bazı alanlar (örn. `imkanlar`) hem metinden hem fotoğraftan karışık dolduruluyordu. Aynı alanda iki farklı kanıt kaynağı çelişince hangisinin doğru olduğu belirsizdi; hata analizi bu karışımı net gösterdi.
2. **Şema, sitenin filtre paneliyle hizalı değildi:** El-yapımı 21+7 alan, Emlakjet'in kullanıcıya sunduğu gerçek filtrelerle birebir örtüşmüyordu. Sonuç: kullanıcının sitede gördüğü bir filtre (örn. "Görüntülü Gezilebilir") bizim hard-filter setimizde yoktu — yani "hard filter" ile "kullanıcının beklediği filtre" ayrışıyordu.
3. **Gold maliyeti:** Her şema değişikliğinde elle doldurulan gold template'i (30 listing × ~19 değerlendirme) yeniden üretmek gerekiyordu; bu, iterasyonu pahalı ve yavaş yapıyordu.

---

## Gen 3 — Canonical Emlakjet registry — **AKTİF**

**Ne yapıyoruz:** Gen 2'nin **metin-metin paradigmasını koruyoruz** (aynı BGE-M3 → Chroma → reranker hattı), ama şemayı **Emlakjet'in kendi filtre panelinin birebir kopyası** ile değiştirdik.

**Üç temel değişiklik:**
1. **Canonical registry (`schema/emlakjet_filters.py`):** Emlakjet kiralık konut sol-filtre paneli = **150 slug** (26 structured + 124 checkbox). Tek merkezi tanım. Sonuç: hard-filter seti = kullanıcının sitede gördüğü filtre seti. Hizalama problemi (Gen 2 #2) çözüldü.
2. **Katı modalite ayrımı:** Her alan tek bir kanıt kaynağına bağlı:
   - **scraper** → `İlan Bilgileri` + seçili `İlan Özellikleri` (deterministik)
   - **DeepSeek (text-only)** → yalnız başlık + açıklama; kalan null alanları açık metin kanıtıyla `true`/`false`
   - **Kimi (image-only)** → yalnız fotoğraflar; kalan null boolean alanları yalnız açık görsel kanıtla `true` (positive-only)
   - Modalite karışıklığı (Gen 2 #1) çözüldü.
3. **Monotonik kaynak izleme:** `filter_values` (değer) + `filter_sources` (kaynak). Sonraki aşama önceki kaynağın değerini **ezemez**; her dolu alan nereden geldiğini taşır. Açıklanabilirlik ve hata ayıklama kolaylaştı.

**Neden bu mimari iyi (hocaya iki argüman):**
- **Feasibility / scale:** Etiketleme bir defa yapılıp veriye yazılıyor; sorgu anında yalnız metin-metin retrieval çalışıyor. Yüzbinlerce ilana ölçeklenir (CLIP'in çoklu-foto + modality gap problemleri yok).
- **Açıklanabilirlik:** Her sonuç için "neden eşleşti" deterministik gösterilebilir — sorgunun hangi canonical filtresini ilanın hangi `filter_value`'su karşıladı (match reason).

---

## Ne değişti — özet

| Boyut | Gen 1 (CLIP) | Gen 2 (el-yapımı) | Gen 3 (canonical) |
|---|---|---|---|
| Arama paradigması | görsel↔metin | metin↔metin | metin↔metin |
| Şema | yok (embedding) | el-yapımı 21+7 alan | Emlakjet paneli = 150 slug |
| Modalite | tek görsel embedding | karışık (text+image aynı alanda) | katı ayrım (scraper/text/image) |
| Kaynak izleme | yok | yok | `filter_sources`, monotonik |
| Sonuç | R@10=0.037, terk | iterasyon pahalı, terk | aktif |

---

## Tarihsel ek — Gen 2 manuel gold doldurma workflow'u

> Bu bölüm Gen 2 dönemine aittir ve **artık çalıştırılmaz** (canonical akışta yeni gold template rebuild yapılmıyor; değerlendirme 10-ilan göz kontrolü + R@10 retrieval eval ile yapılıyor). HANDOFF.md'den buraya tarihsel kayıt olarak taşındı.

**Workflow:** Kullanıcı fotoğraflara (`data/images/<ID>/`) bakar, her ilan için özellikleri seçer, supervisor'a `ID + özellik listesi` verir; supervisor JSON'a çevirip `labeling/gold_listings_manual_todo.jsonl`'a yazar.

### Tek listing inceleme

```bash
python3 -m labeling.gold_helper --listing <ID>
```

Çıktı: 15 structured facts (otomatik dolu) + suggested hybrid + suggested visual + manuel TODO listesi.

**Doldurma kuralı (gold = ground truth):**
- Gördüğün / emin olduğun → işaretle
- Negatif kanıtla **emin** "yok" → `false` (bool) veya `[]` (multi-select)
- Olabilir ama kanıt yok / görünmüyor → `null` (benchmark'ta skip edilir)

**`facts_gold` user-fill (6 bool):** `has_balcony`, `has_elevator`, `has_parking`, `has_aircon`, `near_metro`, `near_metrobus`. (`kitchen_type` kaldırıldı; structured 15 alan otomatik dolu.)

### Visual gold doldurma rehberi

Her listing'in `data/images/<listing_id>/` klasöründeki fotoğraflara bak. **5 compatibility visual alan** (single-select: mutfak_tipi; diğerleri multi-select):

| Alan | Tip | Değerler |
|---|---|---|
| balkon_ozellikleri | multi | cam_balkon, acik_balkon, fransiz_balkon, cikma_balkon, teras |
| manzara | multi | deniz, bogaz, orman_yesil, park, sehir_panorama, dag, ic_avlu, komsu_duvari |
| mutfak_tipi | single | amerikan_acik \| kapali_ayri |
| banyo_ozellikleri | multi | dusakabin, kuvet, jakuzi, banyoda_pencere, birden_fazla_banyo |
| imkanlar | multi | havuz, yesil_alan_peyzaj, guvenlik_kabini, kapali_otopark, acik_otopark, cocuk_parki, spor_alani |

**Multi-select** alanlar JSON array, **single** alanlar string. `imkanlar` site dışı binalarda da geçerli (güvenlik/otopark site olmadan da olabilir).

### Query için doğru ilan ID'leri bulma

```bash
python3 -m evaluation.gold_helper --query "Kadıköy'de 45 bin TL altı eşyalı 2+1"
```

Çıktı: top-10 candidate ID + title + price + URL. Kullanıcı en alakalı 1-3 ID'yi seçer, `evaluation/gold_queries_manual_todo.jsonl`'da `expected_listing_ids` doldurur.

> **Not (Gen 3):** Bu son adım (query gold doldurma) hâlâ geçerli — M4 retrieval'ın R@10 ölçümü için 30 sorgunun `expected_listing_ids`'i bu komutla doldurulacak. Yukarıdaki listing/visual gold doldurma ise Gen 2'ye özgü ve artık kullanılmıyor.

**Tahmini süre (Gen 2):** Listing başına ~6-8 dakika × 30 = ~3-4 saat. Query'ler için ~30 dk.
