"""
Emlakjet Image Downloader
=========================
listings.jsonl dosyasındaki her ilanın image_urls alanındaki
görselleri asenkron olarak indirir.

Dizin yapısı:
    data/images/
    └── {listing_id}/
        ├── 00_<filename>.jpg
        ├── 01_<filename>.jpg
        └── ...

Kullanım:
    python scraper/image_downloader.py
    python scraper/image_downloader.py --input data/raw/listings.jsonl --out data/images
    python scraper/image_downloader.py --concurrency 10 --retries 5
"""

import asyncio
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from tqdm.asyncio import tqdm

# ── Varsayılan ayarlar ───────────────────────────────────────────────────────
DEFAULT_INPUT  = "data/raw/listings.jsonl"
DEFAULT_OUT    = "data/images"
CONCURRENCY    = 8       # aynı anda max bağlantı
RETRIES        = 3       # başarısız istek için yeniden deneme sayısı
RETRY_DELAY    = 2.0     # saniye (üstel artışla: delay * 2^attempt)
TIMEOUT_SEC    = 30      # toplam istek zaman aşımı
CHUNK_SIZE     = 64_000  # bayt cinsinden okuma parçası

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.emlakjet.com/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def slugify_url(url: str) -> str:
    """URL'den güvenli bir dosya adı oluşturur."""
    path = urlparse(url).path          # /listing/123/ABCDEF123.jpg
    name = Path(path).name             # ABCDEF123.jpg
    # Saat zaman damgası ya da garip karakter yoksa olduğu gibi kullan
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "image"


async def download_one(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    sem: asyncio.Semaphore,
    retries: int = RETRIES,
) -> bool:
    """Tek bir görseli indirir; başarılı olursa True döner."""
    if dest.exists() and dest.stat().st_size > 0:
        return True  # zaten indirilmiş

    for attempt in range(retries + 1):
        try:
            async with sem:
                timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC)
                async with session.get(url, headers=HEADERS, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                        )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".tmp")
                    with tmp.open("wb") as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                    tmp.rename(dest)
                    return True

        except Exception as exc:
            if attempt < retries:
                delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                tqdm.write(f"  [HATA] {url}  →  {exc}")
                return False

    return False


async def download_listing(
    session: aiohttp.ClientSession,
    listing: dict,
    out_dir: Path,
    sem: asyncio.Semaphore,
    pbar: tqdm,
) -> dict:
    """
    Bir ilanın tüm görselleri indirir.
    Geri dönüş: {listing_id, ok, fail, skipped} istatistikleri
    """
    lid        = listing["id"]
    # Sadece bu ilana ait URL'ler: /listing/{lid}/ yolunu içerenleri al.
    # Geri kalanlar Emlakjet'in "benzer ilanlar" bölümünden geliyor.
    image_urls = [
        url for url in listing.get("image_urls", [])
        if f"/listing/{lid}/" in url
    ]
    listing_dir = out_dir / lid

    stats = {"id": lid, "ok": 0, "fail": 0, "skipped": 0}

    tasks = []
    for idx, url in enumerate(image_urls):
        filename = f"{idx:02d}_{slugify_url(url)}"
        dest = listing_dir / filename
        tasks.append(download_one(session, url, dest, sem))

    results = await asyncio.gather(*tasks)

    for success in results:
        if success:
            stats["ok"] += 1
        else:
            stats["fail"] += 1

    pbar.update(1)
    pbar.set_postfix({"son_ilan": lid, "ok": stats["ok"], "hata": stats["fail"]})
    return stats


async def run(input_path: Path, out_dir: Path, concurrency: int, retries: int) -> None:
    listings = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                listings.append(json.loads(line))

    if not listings:
        print("listings.jsonl dosyası boş veya okunamadı.")
        sys.exit(1)

    total_images = sum(len(l.get("image_urls", [])) for l in listings)
    print(f"Toplam ilan : {len(listings)}")
    print(f"Toplam görsel: {total_images}")
    print(f"Hedef klasör: {out_dir.resolve()}")
    print(f"Eşzamanlılık: {concurrency}  |  Yeniden deneme: {retries}")
    print()

    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(listings), desc="İlanlar", unit="ilan") as pbar:
            tasks = [
                download_listing(session, listing, out_dir, sem, pbar)
                for listing in listings
            ]
            all_stats = await asyncio.gather(*tasks)

    # ── Özet ────────────────────────────────────────────────────────────────
    total_ok   = sum(s["ok"]   for s in all_stats)
    total_fail = sum(s["fail"] for s in all_stats)

    print()
    print("=" * 50)
    print(f"Tamamlandı!")
    print(f"  Başarılı  : {total_ok}")
    print(f"  Başarısız : {total_fail}")
    print(f"  Toplam    : {total_ok + total_fail}")
    print("=" * 50)

    if total_fail > 0:
        print("\nBazı görseller indirilemedi. Yeniden çalıştırarak eksikler tamamlanabilir.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emlakjet ilan görsellerini asenkron indirir."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"listings.jsonl yolu (varsayılan: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Görsellerin kaydedileceği klasör (varsayılan: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
        help=f"Eşzamanlı bağlantı sayısı (varsayılan: {CONCURRENCY})"
    )
    parser.add_argument(
        "--retries", type=int, default=RETRIES,
        help=f"Başarısız istek için yeniden deneme sayısı (varsayılan: {RETRIES})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(
        input_path  = Path(args.input),
        out_dir     = Path(args.out),
        concurrency = args.concurrency,
        retries     = args.retries,
    ))
