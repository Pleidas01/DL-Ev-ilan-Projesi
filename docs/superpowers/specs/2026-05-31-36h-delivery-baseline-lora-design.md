# 36 Saatlik Teslim: Baseline Demo + Olcum Kapili LoRA

> Tarih: 2026-05-31. Bu belge `2026-05-31-m4-m5-canonical-design.md` uzerine teslim onceligi ekler.

## Amaç

Once calisan ve olculen Gen3 Streamlit demosunu teslim edilebilir hale getir. Ardindan BGE-M3 LoRA deneyini ayri bir asama olarak kos. LoRA yalniz held-out retrieval metriklerini iyilestirirse aktif modele alinir; aksi halde baseline demo korunur ve deney sonucu raporlanir.

## Sabit Kararlar

- Final veri `305` ham ilandan cleaner sonrasi kalan `303` aktif ilandir.
- DeepSeek sorgu zamaninda iki rolde korunur: canonical slot extraction ve kisa RAG cevabi.
- Kural tabanli parser bu teslimin kritik yoluna girmez.
- Hazir `BAAI/bge-m3` baseline index once kurulur.
- Gold doldurulmayacak. Retrieval raporu gold-free otomatik metriklerle uretilir.
- LoRA demo icin zorunlu degildir. Iyilestirme gostermeyen adapter production yoluna baglanmaz.

## Baseline Veri Akisi

```text
Turkce sorgu
  -> DeepSeek canonical hard_filters JSON
  -> Chroma metadata hard filter
  -> BGE-M3 dense embedding aday arama
  -> bge-reranker-v2-m3 yeniden siralama
  -> deterministik match chip'leri
  -> DeepSeek kisa RAG cevabi
  -> Streamlit ilan kartlari
```

Kimi sorgu zamaninda cagrilmaz. Fotograf analizi offline tamamlanmis `filter_values`, `filter_sources` ve `enriched_doc` alanlarinda saklanir.

## Gold-Free Degerlendirme

1. **Synthetic known-item R@1/R@5/R@10 + MRR:** Her ilanin canonical degerlerinden deterministik Turkce sorgu uretilir. Kaynak ilan top-K sonucunda aranir. Bu gercek relevance gold degil, kontrollu bir retrieval proxy metriktir.
2. **Filter satisfaction@K:** `llm.shootout.BENCHMARK_QUERIES[*].expected` ground truth kabul edilir. Donen ilanlar beklenen hard filter degerlerini gercekten karsiliyor mu kodla sayilir.
3. **Query coverage:** En az bir ilan donduren benchmark sorgularinin orani ayrica raporlanir. Bos sonuc sahte `%100 satisfaction` uretmez.
4. **Canli smoke:** 5-8 temsili sorgu gercek Chroma ve gercek slot extraction ile gozle kontrol edilir.

## LoRA Deneyi

- Baseline demo ve baseline rapor tamamlanmadan LoRA baslatilmaz.
- Listing ID bazli train/test ayrimi yapilir. Ayni ilandan tureyen sorgu iki tarafa dagitilmaz.
- Pozitif cift: sentetik sorgu + kaynak ilan `enriched_doc`.
- Zor negatif: benzer konum/oda/fiyat tasiyan fakat kaynak ilan olmayan ilanlar.
- Tam modeli bastan egitmek yerine PEFT LoRA adapter egitilir.
- Baseline ve adapter ayni held-out sentetik sorgularla karsilastirilir.
- Adapter yalniz R@K/MRR iyilesirse aktif embedder adayi olur.

## Teslim Sirasi

1. Final `labeled.jsonl` butunlugunu dosyadan dogrula.
2. Baseline Chroma index kur ve metadata kontratini kontrol et.
3. Gold-free eval harness'ini TDD ile ekle.
4. Gercek smoke, otomatik rapor ve Streamlit QA kos.
5. Baseline sayilarini sunum tablosuna yaz.
6. Zaman kalirsa LoRA deneyini ayri planla uygula.

## Basari Kriteri

- Final veri `303` benzersiz ilan ve bos olmayan `enriched_doc` tasir.
- Chroma collection count `303` olur.
- Full pytest suite skip olmadan yesildir.
- `evaluation/results/retrieval_eval.json` ve `.md` uretilir.
- Streamlit kartlari filtreleri ve deterministik match chip'lerini gosterir.
- LoRA sonucu baseline'dan ayri ve durust bir deney olarak raporlanir.
