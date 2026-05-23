# Milestone 2 Manual Gold Set TODO

Bu milestone otomatik doldurulmayacak. Buradaki amaç 30 ilanın image+text etiketlerini ve 30 arama sorgusunun beklenen ilan ID'lerini Mertcan'ın elle belirlemesi.

## 1. Listing Labeling Gold Set

Dosya:

- `labeling/gold_listings_manual_todo.jsonl`

Her satırda `listing_id`, `title`, boş `text_gold` ve boş `image_gold` var.

Her ilan için `data/processed/dataset.jsonl` içindeki `text`, `title`, `district`, `attributes` alanlarını oku; ayrıca `all_image_paths` altındaki fotoğraflara bak. Sonra şu alanları manuel doldur:

```json
{
  "text_gold": {
    "elevator": true,
    "in_gated_complex": null,
    "security": null,
    "furnished_text": false,
    "heating": "bilinmiyor",
    "parking": "bilinmiyor",
    "near_transit": true,
    "sea_view_mentioned": false,
    "school_nearby": null,
    "pet_friendly": null
  },
  "image_gold": {
    "room_types_seen": ["salon", "mutfak"],
    "natural_light": "medium",
    "spaciousness": "ferah",
    "modernity": "sade",
    "furnished_visual": true,
    "balcony_visual": false,
    "view": "yok",
    "salon_size": "genis",
    "kitchen_type": "standart",
    "floor_material": "parke",
    "overall_condition": "temiz",
    "notes_tr": "Kısa manuel gözlem."
  }
}
```

`null` kullanımı serbest: ilanda/fotoğrafta anlaşılmıyorsa tahmin etme.

## 2. Retrieval Gold Queries

Dosya:

- `evaluation/gold_queries_manual_todo.jsonl`

Her satırdaki `query` için `expected_listing_ids` alanına 1-3 doğru ilan ID'si yaz. Bu liste retrieval evaluation'da R@5, R@10 ve MRR için kullanılacak.

Örnek:

```json
{"query":"Kadıköy'de 45 bin TL altı eşyalı 2+1","expected_listing_ids":["19166530"],"todo":"done"}
```

## 3. Done Kontrolü

Milestone 3'e geçmeden önce:

- `labeling/gold_listings_manual_todo.jsonl` içinde 30 satırın tamamında `text_gold` ve `image_gold` boş olmamalı.
- `evaluation/gold_queries_manual_todo.jsonl` içinde 30 satırın tamamında `expected_listing_ids` boş olmamalı.
- Emin olmadığın alanlarda `null` kullan; rastgele doğruymuş gibi doldurma.

Bu dosyalar dolmadan 1500 ilanlık labeling batch başlatılmayacak.
