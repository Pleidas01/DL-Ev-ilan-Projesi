"""
Veri Temizleyici — Scraping Sonrası Kalite Filtresi
====================================================
Görevler:
  1. Bozuk / küçük görselleri filtrele
  2. Duplicate ilanları temizle (URL hash + görsel hash)
  3. Türkçe metin normalizasyonu
  4. Fine-tuning dataseti oluştur: image_path + text çiftleri
  5. İstatistik raporu yaz

Kullanım:
    python scraper/cleaner.py --raw data/raw --images data/images --out data/processed
"""

import json
import hashlib
import argparse
import unicodedata
import re
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# ─── Sabitler ─────────────────────────────────────────────────────────────────
MIN_IMAGE_W = 200   # piksel — minimum genişlik
MIN_IMAGE_H = 150   # piksel — minimum yükseklik
MAX_TEXT_LEN = 256  # karakter — fine-tuning text maksimum uzunluğu
MAX_DESCRIPTION_LEN = 4000  # ham açıklama — M3 LLM için tam metin

# Temizlenecek HTML kalıpları
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Fiyat normalizasyonu
PRICE_PATTERN = re.compile(r"[\d.,]+")


# ─── Metin Normalizasyonu ──────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Türkçe metin normalizasyonu."""
    if not text:
        return ""
    # HTML tag'lerini temizle (tam tag'ler + sonda kesik açılış tag'i)
    text = HTML_PATTERN.sub(" ", text)
    text = re.sub(r'<[^<]*$', '', text)
    # Unicode normalize (NFC — Türkçe karakterler için)
    text = unicodedata.normalize("NFC", text)
    # Fazla boşlukları temizle
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def normalize_price(price_str: str) -> str:
    """Fiyat metnini standart forma getir."""
    if not price_str:
        return ""
    # Sadece rakamları ve birimi koru
    nums = PRICE_PATTERN.findall(price_str.replace(".", "").replace(",", ""))
    amount = nums[0] if nums else ""

    # Para birimi tespiti
    if "TL" in price_str.upper() or "₺" in price_str:
        currency = "TL"
    elif "USD" in price_str.upper() or "$" in price_str:
        currency = "USD"
    elif "EUR" in price_str.upper() or "€" in price_str:
        currency = "EUR"
    else:
        currency = "TL"

    return f"{amount} {currency}".strip() if amount else price_str.strip()


def build_text_description(listing: dict) -> str:
    """
    Fine-tuning için kısa, bilgi yoğun bir metin açıklaması oluştur.
    Format: "<başlık>, <fiyat>, <konum>. <açıklama özeti>"
    """
    parts = []
    if listing.get("title"):
        parts.append(normalize_text(listing["title"]))
    if listing.get("price"):
        price = normalize_price(listing["price"])
        if price:
            parts.append(price)
    if listing.get("district"):
        parts.append(normalize_text(listing["district"]))

    header = ", ".join(parts)

    desc = normalize_text(listing.get("description", ""))
    if desc:
        # Açıklamayı kısalt — başlık + fiyat + konum sonrası kalan yer kadar
        remaining = MAX_TEXT_LEN - len(header) - 2
        if remaining > 30:
            desc = desc[:remaining].rsplit(" ", 1)[0] + "…"
        else:
            desc = ""

    full_text = f"{header}. {desc}".strip() if desc else header
    return full_text[:MAX_TEXT_LEN]


def _parse_tl(value: object) -> int | None:
    """Parse Turkish lira strings such as '60.000 TL' to integer kuruş-free TL."""
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value).replace(".", ""))
    return int(match.group(0)) if match else None


def _normalize_blob(value: object) -> str:
    # Türkçe karakter haritası lower()'dan ÖNCE uygulanır:
    # Python "İ".lower() → "i̇" (U+0069 + U+0307 combining), sonradan translate edemez.
    text = normalize_text(str(value or ""))
    table = str.maketrans({
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "Ç": "c", "Ğ": "g", "İ": "i", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u",
        "â": "a", "Â": "a",
    })
    return text.translate(table).lower()


def _split_location(location: object) -> tuple[str, str, str]:
    parts = [normalize_text(part) for part in str(location or "").split(" - ")]
    parts = [part for part in parts if part]
    city = parts[0] if len(parts) > 0 else ""
    district = parts[1] if len(parts) > 1 else ""
    neighborhood = parts[2] if len(parts) > 2 else ""
    return city, district, neighborhood


def _parse_yes_no(value: object) -> bool | None:
    norm = _normalize_blob(value)
    if norm in {"evet", "var", "yes", "true", "1"}:
        return True
    if norm in {"hayir", "hayir.", "yok", "no", "false", "0"}:
        return False
    return None


def _feature_blob(attrs: dict, raw: dict) -> str:
    features = attrs.get("propertyFeatures") or []
    if not isinstance(features, list):
        features = [features]
    return _normalize_blob(" ".join(str(item) for item in features) + " " + str(raw.get("title", "")))


def _normalize_heating(value: object) -> str | None:
    norm = _normalize_blob(value)
    if "yerden" in norm:
        return "yerden_isitma"
    if "kombi" in norm:
        return "kombi"
    if "dogalgaz" in norm or "dogal gaz" in norm:
        return "dogalgaz"
    if "merkezi" in norm:
        return "merkezi"
    if "klima" in norm:
        return "klima"
    return None


def _parse_furnished(value: object) -> bool | None:
    """Parse 'Eşya Durumu' attribute. Returns True/False/None."""
    norm = _normalize_blob(value)
    if not norm:
        return None
    if "esyasiz" in norm:
        return False
    if "esyali" in norm or "mobilyali" in norm:
        return True
    return None


def _detect_aircon(attrs: dict, feature_blob: str) -> bool | None:
    """has_aircon: property_features veya heating tipi 'klima' içeriyorsa True."""
    if "klima" in feature_blob:
        return True
    if "klima" in _normalize_blob(attrs.get("heating", "")):
        return True
    return None


def _infer_kitchen_type(blob: str) -> str | None:
    if "amerikan" in blob or "acik mutfak" in blob:
        return "amerikan_acik"
    if "yari acik" in blob:
        return "yari_acik"
    if "kapali mutfak" in blob or "ayri mutfak" in blob:
        return "kapali_ayri"
    return None


def _contains_any(blob: str, needles: tuple[str, ...]) -> bool | None:
    return True if any(needle in blob for needle in needles) else None


# ─── Görsel Doğrulama ─────────────────────────────────────────────────────────
def validate_image(image_path: str) -> tuple[bool, str | None]:
    """
    Görselin kullanılabilir olup olmadığını kontrol et.
    Returns: (is_valid, reason_if_invalid)
    """
    path = Path(image_path)
    if not path.exists():
        return False, "Dosya bulunamadı"
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_IMAGE_W or h < MIN_IMAGE_H:
                return False, f"Çok küçük ({w}x{h})"
            if img.mode not in {"RGB", "RGBA", "L"}:
                return False, f"Desteklenmeyen renk modu: {img.mode}"
        return True, None
    except UnidentifiedImageError:
        return False, "Tanımsız görsel formatı"
    except Exception as e:
        return False, str(e)


def clean_record(raw: dict, *, image_path: str, all_image_paths: list[str]) -> dict:
    """Ham listing kaydını M3-ready dataset satırına dönüştürür."""
    attrs = raw.get("attributes") or {}
    text = build_text_description(raw)
    city, district, neighborhood = _split_location(raw.get("district", ""))
    features = attrs.get("propertyFeatures") or []
    feature_blob = _feature_blob(attrs, raw)
    furnished_attr = _parse_furnished(attrs.get("furnishedStatus"))
    is_furnished_value = (
        furnished_attr
        if furnished_attr is not None
        else _contains_any(feature_blob, ("esyali", "mobilyali", "full esyali", "ful esyali"))
    )
    return {
        "id": raw.get("id", ""),
        "url": raw.get("url", ""),
        "image_path": image_path,
        "all_image_paths": all_image_paths,
        "text": text,
        "title": normalize_text(raw.get("title", "")),
        "price": normalize_price(raw.get("price", "")),
        "city": city,
        "district": district,
        "neighborhood": neighborhood,
        "price_tl": _parse_tl(raw.get("price", "")),
        "room_count": attrs.get("roomCount"),
        "gross_size_m2": _parse_int(attrs.get("grossSize")),
        "net_size_m2": _parse_int(attrs.get("netSize")),
        "building_age": attrs.get("buildingAge", "") or "",
        "floor": attrs.get("floor", "") or "",
        "total_floors": _parse_int(attrs.get("totalFloors")),
        "heating_type": _normalize_heating(attrs.get("heating")),
        "kitchen_type": _infer_kitchen_type(feature_blob),
        "has_balcony": _contains_any(feature_blob, ("balkon", "teras")),
        "has_elevator": _contains_any(feature_blob, ("asansor",)),
        "has_aircon": _detect_aircon(attrs, feature_blob),
        "is_furnished": is_furnished_value,
        "deposit_tl": _parse_tl(attrs.get("deposit")),
        "has_parking": _contains_any(feature_blob, ("otopark", "garaj", "park yeri")),
        "in_gated_complex": _parse_yes_no(attrs.get("inGatedComplex")),
        "near_metro": None,
        "near_metrobus": None,
        "title_deed_status": attrs.get("titleDeedStatus") or None,
        "description": normalize_text(raw.get("description", ""))[:MAX_DESCRIPTION_LEN],
        "heating": attrs.get("heating", "") or "",
        "property_features": features,
        "visual_qualities": {},
        "attributes": attrs,
        "scraped_at": raw.get("scraped_at", ""),
    }


def image_hash(image_path: str) -> str | None:
    """Görsel içeriğinin MD5 hash'ini hesapla (duplicate tespiti)."""
    try:
        with Image.open(image_path) as img:
            # Küçük thumbnail üzerinden hash al (hızlı)
            img_thumb = img.resize((32, 32)).convert("RGB")
            data = img_thumb.tobytes()
            return hashlib.md5(data).hexdigest()
    except Exception:
        return None


# ─── Ana Temizleme Fonksiyonu ──────────────────────────────────────────────────
def clean_dataset(raw_dir: Path, out_dir: Path, images_dir: Path | None = None) -> dict:
    """
    Ham JSONL verisini temizle, fine-tuning dataseti oluştur.
    Returns: İstatistik sözlüğü
    """
    raw_jsonl = raw_dir / "listings.jsonl"
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Ham veri bulunamadı: {raw_jsonl}")

    # İndirilen görsellerin kök dizini (varsayılan: raw_dir/../images)
    if images_dir is None:
        images_dir = raw_dir.parent / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "dataset.jsonl"
    report_path = out_dir / "cleaning_report.json"

    stats = {
        "total_raw": 0,
        "skipped_no_image": 0,
        "skipped_bad_image": 0,
        "skipped_duplicate_url": 0,
        "skipped_duplicate_image": 0,
        "skipped_no_text": 0,
        "saved": 0,
    }

    seen_urls: set[str] = set()
    seen_image_hashes: set[str] = set()

    print(f"\n[Temizleme] {raw_jsonl} okunuyor...")

    # Ham veri satırlarını say
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    stats["total_raw"] = len(lines)

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines, desc="Temizleniyor", unit="listing"):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── URL duplicate kontrolü ──────────────────────────────────
            url = raw.get("url", "")
            if url in seen_urls:
                stats["skipped_duplicate_url"] += 1
                continue
            seen_urls.add(url)

            # ── Görsel kontrolü ─────────────────────────────────────────
            # İndirilen görseller data/images/{listing_id}/ klasöründe
            lid = raw.get("id", "")
            listing_img_dir = images_dir / lid
            local_images = sorted(listing_img_dir.glob("*")) if listing_img_dir.exists() else []
            # Sadece bu ilana ait görseller: dosya adında kendi listing ID'si geçmeli.
            # Emlakjet sayfasındaki "benzer ilanlar" thumbnail'leri farklı ID içeriyor.
            local_images = [
                str(p) for p in local_images
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                and lid in p.stem
            ]

            if not local_images:
                stats["skipped_no_image"] += 1
                continue

            # İlk geçerli görseli seç
            primary_image = None
            for img_path in local_images:
                valid, reason = validate_image(img_path)
                if valid:
                    primary_image = img_path
                    break

            if not primary_image:
                stats["skipped_bad_image"] += 1
                continue

            # ── Görsel duplicate kontrolü ───────────────────────────────
            img_h = image_hash(primary_image)
            if img_h and img_h in seen_image_hashes:
                stats["skipped_duplicate_image"] += 1
                continue
            if img_h:
                seen_image_hashes.add(img_h)

            # ── Metin oluştur ───────────────────────────────────────────
            text = build_text_description(raw)
            if len(text) < 10:  # Çok kısa metinleri atla
                stats["skipped_no_text"] += 1
                continue

            # ── Temiz kaydı yaz ─────────────────────────────────────────
            record = clean_record(
                {**raw, "url": url},
                image_path=primary_image,
                all_image_paths=local_images,
            )
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["saved"] += 1

    # ─── Rapor ────────────────────────────────────────────────────────────────
    stats["retention_rate"] = round(stats["saved"] / max(stats["total_raw"], 1) * 100, 1)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Scraping Verisi Temizleyici")
    parser.add_argument("--raw",    default="data/raw",       help="Ham veri dizini")
    parser.add_argument("--images", default="data/images",    help="İndirilen görsel kök dizini")
    parser.add_argument("--out",    default="data/processed", help="Temizlenmiş veri dizini")
    args = parser.parse_args()

    stats = clean_dataset(Path(args.raw), Path(args.out), Path(args.images))

    print("\n" + "=" * 50)
    print("TEMİZLEME RAPORU")
    print("=" * 50)
    print(f"  Ham listing sayısı    : {stats['total_raw']}")
    print(f"  Görsel yok            : {stats['skipped_no_image']}")
    print(f"  Bozuk görsel          : {stats['skipped_bad_image']}")
    print(f"  Duplicate URL         : {stats['skipped_duplicate_url']}")
    print(f"  Duplicate görsel      : {stats['skipped_duplicate_image']}")
    print(f"  Metin yok/kısa        : {stats['skipped_no_text']}")
    print(f"  ─────────────────────")
    print(f"  Kaydedilen            : {stats['saved']}")
    print(f"  Tutma oranı           : %{stats['retention_rate']}")
    print(f"  Çıktı                 : data/processed/dataset.jsonl")
    print("=" * 50)
