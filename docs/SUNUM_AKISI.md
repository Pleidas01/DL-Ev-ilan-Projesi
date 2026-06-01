# Sunum Akışı

## 1. Problem

Türkçe emlak sorgusunu yalnız metin benzerliğiyle aramak yetersizdi. Kullanıcı hem yapılandırılmış filtre hem de serbest metin beklentisi taşıyor.

## 2. Mimari Evrim

| Nesil | Yaklaşım | Karar |
|---|---|---|
| Gen1 | CLIP tabanlı görsel-metin retrieval | Modality gap nedeniyle terk |
| Gen2 | El yapımı sınırlı JSON şema | Genişletmesi pahalı olduğu için terk |
| Gen3 | Emlakjet canonical registry + hibrit retrieval | Aktif mimari |

## 3. Veri

- `305` ham ilan, cleaner sonrası `303` aktif ilan.
- Her ilanda `150` canonical slot.
- Scraper mevcut fact'leri deterministik dolduruyor.
- DeepSeek yalnız açıklama ve başlıktan kalan null alanları tamamlıyor.
- Kimi görsellerden positive-only özellik ekliyor; sorgu zamanında çağrılmıyor.

## 4. Canlı Akış

```text
Türkçe sorgu
  -> DeepSeek slot extraction
  -> Chroma metadata hard filter
  -> BGE-M3 dense retrieval
  -> bge-reranker-v2-m3
  -> match chip'leri
  -> DeepSeek kısa RAG cevabı
  -> Streamlit kartları
```

## 5. Baseline Ölçümü

Bu ölçüm manuel relevance gold değildir; deadline nedeniyle kontrollü gold-free proxy kullanıldı.

| Metrik | Sonuç | Açıklama |
|---|---:|---|
| Chroma row | 303 | Final aktif ilanların tamamı |
| Synthetic known-item sorgu | 4 | MacBook Air'de bounded örneklem |
| R@1 / R@5 / R@10 / MRR | 1.0000 / 1.0000 / 1.0000 / 1.0000 | Kontrollü retrieval proxy |
| Canlı benchmark sorgu | 3 | DeepSeek slot extraction dahil |
| Query coverage | 1/3 = 0.3333 | İki sorguya dataset içinde uygun ilan yok |
| Filter satisfaction | 8/8 = 1.0000 | Dönen sonuçların hard filter doğruluğu |
| Test paketi | 104 passed | Skip yok |

## 6. Demo

Sorgu: `geniş salonlu denize yakın 2+1 asansörlü ev`

- `8` kart render edildi.
- Kartlarda `2+1`, `Asansör`, `Denize Yakın` eşleşme çipleri görüldü.
- RAG cevabı yalnız retrieved ilanlara dayandı.

## 7. Kendi Eğittiğimiz NN

BGE-M3 üzerine küçük LoRA adapter deneyi RTX 5070 Ti makinede koşulacak.

| Ölçüm | Baseline | LoRA |
|---|---:|---:|
| Dense-only held-out R@10 | TBD | TBD |
| Dense-only held-out MRR | TBD | TBD |
| Karar | baseline | ACCEPT veya REJECT_KEEP_BASELINE |

Adapter yalnız held-out MRR artar ve R@10 gerilemezse demo adayı olacak.

## 8. Dürüst Sınırlılıklar

- `303` ilan demo ölçeğidir; üretim ölçeği değildir.
- Manuel relevance gold doldurulmadı; proxy metriği gerçek kullanıcı relevansı gibi sunmuyoruz.
- MacBook Air ortamı CPU-only olduğu için full eval ve LoRA eğitimi RTX makineye taşındı.

