"""
Emlakjet Scraper v4 — XHR Interception + DOM Fallback
======================================================
Strateji:
  1. Playwright ile liste sayfasını aç
  2. page.on('response') ile api.emlakjet.com veya search.emlakjet.com
     XHR request'lerini yakala → listing verisi JSON olarak geliyor
  3. Fallback: DOM'dan a[href^='/ilan/'] linklerini topla, detay sayfasında
     imaj.emlakjet.com görsellerini ve metni çek
  4. Sadece imaj.emlakjet.com/listing/ URL'lerini kaydet — reklam yok

URL-only: Mac'e görsel indirilmez, Colab'da training sırasında indirilir.

Kullanım:
    python scraper/playwright_scraper.py --limit 5000 --out data/raw
    python scraper/playwright_scraper.py --limit 50 --headed  # debug
"""

import asyncio
import json
import random
import re
import time
import argparse
from pathlib import Path

import aiofiles
from tqdm import tqdm

from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError, Response

try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
]

# Emlakjet'in kullandığı listing görsel CDN URL pattern'i
# → imaj.emlakjet.com/listing/{id}/HASH.jpg veya /resize/W/H/listing/...
IMG_PATTERN = re.compile(
    r'https://imaj\.emlakjet\.com/(?:resize/\d+/\d+/)?listing/\d+/[A-F0-9]+\.\w+',
    re.IGNORECASE
)

LIST_URLS = [
    "https://www.emlakjet.com/satilik-konut/",
    "https://www.emlakjet.com/kiralik-konut/",
    "https://www.emlakjet.com/satilik-konut/istanbul/",
    "https://www.emlakjet.com/satilik-konut/ankara/",
    "https://www.emlakjet.com/satilik-konut/izmir/",
    "https://www.emlakjet.com/satilik-konut/bursa/",
    "https://www.emlakjet.com/satilik-konut/antalya/",
    "https://www.emlakjet.com/satilik-konut/adana/",
    "https://www.emlakjet.com/satilik-konut/kocaeli/",
    "https://www.emlakjet.com/satilik-konut/mersin/",
    "https://www.emlakjet.com/kiralik-konut/istanbul/",
    "https://www.emlakjet.com/kiralik-konut/ankara/",
    "https://www.emlakjet.com/kiralik-konut/izmir/",
]

ID_RE = re.compile(r'-(\d{6,10})(?:/|$)')


def extract_images_from_html(html: str) -> list[str]:
    """HTML içindeki Emlakjet CDN görsel URL'lerini bul."""
    urls = list(dict.fromkeys(IMG_PATTERN.findall(html)))
    # Normalize: resize URL → original
    result = []
    for u in urls:
        u = re.sub(r'/resize/\d+/\d+/', '/', u)
        result.append(u)
    return list(dict.fromkeys(result))


def parse_json_listings(data: dict | list) -> list[dict]:
    """
    API JSON'undan listing objelerini recursive olarak topla.
    Emlakjet'in farklı endpoint formatlarını destekler.
    """
    results = []

    def walk(obj, depth=0):
        if depth > 6 or not obj:
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            # title + id → listing detect
            lid = obj.get('id') or obj.get('listingId') or obj.get('listing_id')
            title = obj.get('title') or obj.get('name')
            if lid and title:
                results.append(obj)
                return   # alt dallara inme
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)

    walk(data)
    return results


def build_listing_record(obj: dict, source_url: str = "") -> dict | None:
    lid = str(obj.get('id') or obj.get('listingId') or '').strip()
    title = str(obj.get('title') or obj.get('name') or '').strip()
    if not lid or not title:
        return None

    # Fiyat
    price_raw = obj.get('price') or obj.get('salePrice') or obj.get('rentPrice') or {}
    if isinstance(price_raw, dict):
        price = f"{price_raw.get('value', '')} {price_raw.get('currency', 'TL')}".strip()
    else:
        price = str(price_raw)

    # Konum
    loc = obj.get('location') or obj.get('address') or {}
    if isinstance(loc, dict):
        district = ' - '.join(filter(None, [
            loc.get('city') or loc.get('cityName') or '',
            loc.get('district') or loc.get('districtName') or '',
            loc.get('neighborhood') or loc.get('neighborhoodName') or '',
        ]))
    else:
        district = str(loc)

    description = str(obj.get('description') or obj.get('shortDescription') or '').strip()[:500]

    # URL
    slug = obj.get('slug') or obj.get('url') or obj.get('seoUrl') or ''
    if slug.startswith('http'):
        listing_url = slug
    elif slug:
        listing_url = f"https://www.emlakjet.com/ilan/{slug}"
    else:
        listing_url = source_url or f"https://www.emlakjet.com/ilan/{lid}"

    # Görseller — sadece Emlakjet CDN
    image_urls: list[str] = []
    for key in ['images', 'photos', 'gallery', 'coverImages', 'coverImage', 'thumbnailUrl']:
        val = obj.get(key)
        if not val:
            continue
        if isinstance(val, str) and IMG_PATTERN.match(val):
            u = re.sub(r'/resize/\d+/\d+/', '/', val)
            image_urls.append(u)
        elif isinstance(val, list):
            for img in val:
                url = ''
                if isinstance(img, str):
                    url = img
                elif isinstance(img, dict):
                    url = img.get('url') or img.get('src') or img.get('path') or ''
                if url and IMG_PATTERN.match(url):
                    url = re.sub(r'/resize/\d+/\d+/', '/', url)
                    image_urls.append(url)

    image_urls = list(dict.fromkeys(image_urls))

    attrs = {k: obj[k] for k in ['roomCount', 'grossSize', 'netSize', 'floor',
                                   'buildingAge', 'propertyType', 'tradeType']
             if k in obj}

    return {
        'id': lid,
        'url': listing_url,
        'title': title,
        'price': price,
        'description': description,
        'district': district,
        'image_urls': image_urls,
        'image_count': len(image_urls),
        'attributes': attrs,
        'scraped_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


class EmlakjetScraper:
    def __init__(self, out_dir: Path, limit: int, headless: bool = True):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / 'listings.jsonl'
        self.limit = limit
        self.headless = headless
        self.seen_ids: set[str] = set()
        out_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        if not self.jsonl_path.exists():
            return
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.seen_ids.add(str(json.loads(line).get('id', '')))
                except Exception:
                    pass
        if self.seen_ids:
            print(f'[Resume] {len(self.seen_ids)} listing mevcut.')

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-blink-features=AutomationControlled'],
            )
            ctx = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1440, 'height': 900},
                locale='tr-TR',
                timezone_id='Europe/Istanbul',
            )
            page = await ctx.new_page()
            if STEALTH_AVAILABLE:
                await stealth_async(page)

            saved = 0
            xhr_listings: list[dict] = []   # XHR'dan yakalanan listingler

            # ── XHR Response interceptor ──────────────────────────────────
            async def on_response(response: Response):
                url = response.url
                ct = response.headers.get('content-type', '')
                if 'json' not in ct:
                    return
                if not any(d in url for d in ['api.emlakjet.com', 'search.emlakjet.com',
                                               'emlakjet.com/api', 'emlakjet.com/_next/data']):
                    return
                try:
                    data = await response.json()
                    items = parse_json_listings(data)
                    for item in items:
                        rec = build_listing_record(item)
                        if rec and rec['id'] not in self.seen_ids:
                            xhr_listings.append(rec)
                except Exception:
                    pass

            page.on('response', lambda r: asyncio.ensure_future(on_response(r)))

            # ── Detay sayfası HTML scraper (fallback) ─────────────────────
            async def scrape_detail(detail_page, listing_url: str) -> dict | None:
                try:
                    await detail_page.goto(listing_url, wait_until='domcontentloaded', timeout=25000)
                    await detail_page.wait_for_timeout(random.randint(800, 1500))
                    html = await detail_page.content()
                    image_urls = extract_images_from_html(html)

                    title = ''
                    try:
                        title = await detail_page.locator('h1').first.inner_text(timeout=2000)
                        title = title.strip()
                    except Exception:
                        pass

                    price = ''
                    for sel in ["[class*='styles_price']", "[class*='price__']", "[class*='Price']"]:
                        try:
                            price = await detail_page.locator(sel).first.inner_text(timeout=1500)
                            price = price.strip()
                            if price:
                                break
                        except Exception:
                            pass

                    district = ''
                    for sel in ["[class*='styles_location']", "[class*='location__']", "[class*='address']"]:
                        try:
                            district = await detail_page.locator(sel).first.inner_text(timeout=1500)
                            district = district.strip()
                            if district:
                                break
                        except Exception:
                            pass

                    if not title or not image_urls:
                        return None

                    m = ID_RE.search(listing_url)
                    lid = m.group(1) if m else listing_url[-12:]

                    return {
                        'id': lid,
                        'url': listing_url,
                        'title': title,
                        'price': price,
                        'description': '',
                        'district': district,
                        'image_urls': image_urls[:10],
                        'image_count': len(image_urls),
                        'attributes': {},
                        'scraped_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }
                except Exception:
                    return None

            async with aiofiles.open(self.jsonl_path, 'a', encoding='utf-8') as out_f:

                async def save(rec: dict):
                    nonlocal saved
                    if rec['id'] in self.seen_ids:
                        return
                    if rec['image_count'] == 0:
                        return
                    await out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    await out_f.flush()
                    self.seen_ids.add(rec['id'])
                    saved += 1

                # Detay sayfası için ayrı bir page
                detail_page = await ctx.new_page()
                if STEALTH_AVAILABLE:
                    await stealth_async(detail_page)

                for list_url_base in LIST_URLS:
                    if saved >= self.limit:
                        break

                    print(f'\n→ {list_url_base}')

                    for page_num in range(1, 40):
                        if saved >= self.limit:
                            break

                        purl = list_url_base if page_num == 1 else f"{list_url_base}?sayfa={page_num}"
                        xhr_listings.clear()

                        try:
                            await page.goto(purl, wait_until='domcontentloaded', timeout=25000)
                            # XHR'ların gelmesi için bekle (networkidle yerine explicit wait)
                            await page.wait_for_timeout(random.randint(3000, 4500))
                        except PWTimeoutError:
                            print(f'  [Sayfa {page_num}] Timeout — sonraki URL')
                            break
                        except Exception as e:
                            print(f'  [Sayfa {page_num}] Hata: {e}')
                            break

                        # ── Yol A: XHR'dan yakalananlar ──────────────────
                        if xhr_listings:
                            page_saved = 0
                            for rec in xhr_listings:
                                await save(rec)
                                page_saved += 1
                                if saved >= self.limit:
                                    break
                            print(f'  [Sayfa {page_num}] XHR: {len(xhr_listings)} listing, {page_saved} kaydedildi | Toplam: {saved}')
                            if len(xhr_listings) < 5:
                                break  # Son sayfa
                            await asyncio.sleep(random.uniform(1.5, 2.5))
                            continue

                        # ── Yol B: DOM'dan linkler + detay sayfası ────────
                        hrefs = await page.evaluate("""
                            () => Array.from(document.querySelectorAll('a[href^="/ilan/"]'))
                                      .map(a => a.getAttribute('href'))
                                      .filter((h, i, arr) => arr.indexOf(h) === i)
                        """)

                        if not hrefs:
                            print(f'  [Sayfa {page_num}] Hiç link yok — sonraki URL')
                            break

                        print(f'  [Sayfa {page_num}] {len(hrefs)} link bulundu, detay sayfaları ziyaret ediliyor...')
                        page_saved = 0

                        for href in hrefs:
                            if saved >= self.limit:
                                break
                            full_url = 'https://www.emlakjet.com' + href
                            m = ID_RE.search(href)
                            if m and m.group(1) in self.seen_ids:
                                continue

                            rec = await scrape_detail(detail_page, full_url)
                            if rec:
                                await save(rec)
                                page_saved += 1

                            await asyncio.sleep(random.uniform(1.2, 2.0))

                        print(f'  [Sayfa {page_num}] DOM+Detay: {page_saved} kaydedildi | Toplam: {saved}')

                        if page_saved == 0 and page_num > 1:
                            break

                        await asyncio.sleep(random.uniform(1.5, 3.0))

            await detail_page.close()
            await page.close()
            await ctx.close()
            await browser.close()

        print(f"\n{'='*55}")
        print(f'TAMAMLANDI — {saved} listing')
        print(f'Çıktı: {self.jsonl_path}')
        print(f'Görseller: Sadece URL (Mac disk kullanılmadı)')
        print(f"{'='*55}")
        return saved


async def main(args):
    scraper = EmlakjetScraper(
        out_dir=Path(args.out),
        limit=args.limit,
        headless=not args.headed,
    )
    await scraper.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emlakjet Scraper v4')
    parser.add_argument('--limit',  type=int, default=5000)
    parser.add_argument('--out',    default='data/raw')
    parser.add_argument('--headed', action='store_true')
    args = parser.parse_args()
    asyncio.run(main(args))
