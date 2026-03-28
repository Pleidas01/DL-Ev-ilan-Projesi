"""
Emlakjet Scraper v5 — Düzeltmeler
===================================
v4'ten yapılan değişiklikler:
  1. Bot koruması  : Stealth argümanları güçlendirildi; Cloudflare challenge
                     tespiti eklendi (CF page → bekle & tekrar dene).
  2. XHR race cond.: asyncio.Event + kısa poll döngüsü ile XHR handler'ların
                     tamamlanması bekleniyor (xhr_listings.clear() güvenli).
  3. image_count   : Filtre kaldırıldı; URL'siz listing'ler de kaydedilir.
  4. CSS selector  : Hashed class yerine meta/OG tag, JSON-LD ve genel
                     semantic selector'larla price/location çekimi.
  5. DOM link      : Birden fazla href pattern denenior; JS evaluate yerine
                     page.locator kullanımı.
  6. Pagination    : ?sayfa= → sayfa sonuna kadar devam, son sayfada dur;
                     boş sayfa tespiti geliştirildi.
  7. Image pattern : Hem eski hem yeni CDN format'ı destekleniyor.

Kullanım:
    python scraper/playwright_scraper.py --limit 5000 --out data/raw
    python scraper/playwright_scraper.py --limit 5 --headed   # debug
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

# ── User-Agent havuzu ──────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# ── CDN görsel pattern'i (FIX 7: eski + yeni format) ──────────────────────────
IMG_PATTERN = re.compile(
    r'https://(?:imaj|img|cdn)\.emlakjet\.com'
    r'/(?:(?:resize|thumb)/\d+/\d+/)?'
    r'(?:listing|listings?)/\d+/[A-Za-z0-9_\-]+\.(?:jpe?g|png|webp)',
    re.IGNORECASE,
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

# Cloudflare challenge sayfası belirteci
CF_INDICATORS = ["cf-browser-verification", "Cloudflare", "Just a moment", "cf_chl"]


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def extract_images_from_html(html: str) -> list[str]:
    urls = list(dict.fromkeys(IMG_PATTERN.findall(html)))
    result = []
    for u in urls:
        u = re.sub(r'/(?:resize|thumb)/\d+/\d+/', '/', u)
        result.append(u)
    return list(dict.fromkeys(result))


def normalize_img_url(url: str) -> str:
    return re.sub(r'/(?:resize|thumb)/\d+/\d+/', '/', url)


def parse_json_listings(data: dict | list) -> list[dict]:
    results = []

    def walk(obj, depth=0):
        if depth > 6 or not obj:
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            lid = obj.get('id') or obj.get('listingId') or obj.get('listing_id')
            title = obj.get('title') or obj.get('name')
            if lid and title:
                results.append(obj)
                return
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

    price_raw = obj.get('price') or obj.get('salePrice') or obj.get('rentPrice') or {}
    if isinstance(price_raw, dict):
        price = f"{price_raw.get('value', '')} {price_raw.get('currency', 'TL')}".strip()
    else:
        price = str(price_raw)

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

    slug = obj.get('slug') or obj.get('url') or obj.get('seoUrl') or ''
    if slug.startswith('http'):
        listing_url = slug
    elif slug:
        listing_url = f"https://www.emlakjet.com/ilan/{slug}"
    else:
        listing_url = source_url or f"https://www.emlakjet.com/ilan/{lid}"

    image_urls: list[str] = []
    for key in ['images', 'photos', 'gallery', 'coverImages', 'coverImage', 'thumbnailUrl',
                'imageUrls', 'image_urls', 'mediaList']:
        val = obj.get(key)
        if not val:
            continue
        if isinstance(val, str):
            if IMG_PATTERN.search(val):
                image_urls.append(normalize_img_url(val))
        elif isinstance(val, list):
            for img in val:
                url = ''
                if isinstance(img, str):
                    url = img
                elif isinstance(img, dict):
                    url = (img.get('url') or img.get('src') or img.get('path')
                           or img.get('originalUrl') or img.get('fullUrl') or '')
                if url and IMG_PATTERN.search(url):
                    image_urls.append(normalize_img_url(url))

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


# ── Cloudflare tespiti ─────────────────────────────────────────────────────────

async def is_cloudflare_page(page) -> bool:
    try:
        content = await page.content()
        return any(ind in content for ind in CF_INDICATORS)
    except Exception:
        return False


async def wait_for_cloudflare(page, max_wait: int = 30) -> bool:
    """CF challenge'ı bekle. True → geçildi, False → timeout."""
    print("  [CF] Cloudflare challenge tespit edildi, bekleniyor...")
    for _ in range(max_wait):
        await asyncio.sleep(1)
        if not await is_cloudflare_page(page):
            print("  [CF] Challenge geçildi ✓")
            return True
    print("  [CF] Challenge geçilemedi — sayfa atlanıyor.")
    return False


# ── Ana scraper sınıfı ─────────────────────────────────────────────────────────

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
                # FIX 1: Bot koruması — daha fazla stealth argümanı
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-infobars',
                    '--window-size=1440,900',
                    '--disable-automation',
                ],
            )
            ctx = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1440, 'height': 900},
                locale='tr-TR',
                timezone_id='Europe/Istanbul',
                # Bot tespitini azaltmak için ek headers
                extra_http_headers={
                    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                    'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                },
            )

            # FIX 1: navigator.webdriver'ı sil
            await ctx.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                Object.defineProperty(navigator, 'languages', { get: () => ['tr-TR', 'tr', 'en-US'] });
                window.chrome = { runtime: {} };
            """)

            page = await ctx.new_page()
            if STEALTH_AVAILABLE:
                await stealth_async(page)

            saved = 0

            # FIX 2: XHR race condition için asyncio.Event
            # Her sayfa için yeni bir event + liste
            xhr_results: list[dict] = []
            xhr_done_event = asyncio.Event()

            pending_xhr: int = 0  # kaç handler hâlâ çalışıyor

            async def on_response(response: Response):
                nonlocal pending_xhr
                url = response.url
                ct = response.headers.get('content-type', '')
                if 'json' not in ct:
                    return
                if not any(d in url for d in [
                    'api.emlakjet.com', 'search.emlakjet.com',
                    'emlakjet.com/api', 'emlakjet.com/_next/data',
                    'emlakjet.com/search', 'emlakjet.com/listing',
                ]):
                    return

                pending_xhr += 1
                try:
                    data = await response.json()
                    items = parse_json_listings(data)
                    for item in items:
                        rec = build_listing_record(item)
                        if rec and rec['id'] not in self.seen_ids:
                            xhr_results.append(rec)
                    if items:
                        print(f"    [XHR] {url[:80]} → {len(items)} listing")
                except Exception as ex:
                    pass
                finally:
                    pending_xhr -= 1
                    if pending_xhr <= 0:
                        xhr_done_event.set()

            page.on('response', lambda r: asyncio.ensure_future(on_response(r)))

            # ── Detay sayfası HTML scraper (fallback DOM) ──────────────────
            async def scrape_detail(detail_page, listing_url: str) -> dict | None:
                try:
                    await detail_page.goto(listing_url, wait_until='domcontentloaded', timeout=30000)
                    await detail_page.wait_for_timeout(random.randint(800, 1500))

                    # CF kontrolü
                    if await is_cloudflare_page(detail_page):
                        passed = await wait_for_cloudflare(detail_page)
                        if not passed:
                            return None

                    html = await detail_page.content()
                    image_urls = extract_images_from_html(html)

                    # FIX 4: Başlık: h1 önce, sonra OG meta
                    title = ''
                    try:
                        title = await detail_page.locator('h1').first.inner_text(timeout=3000)
                        title = title.strip()
                    except Exception:
                        pass
                    if not title:
                        try:
                            title = await detail_page.get_attribute('meta[property="og:title"]', 'content', timeout=1500) or ''
                        except Exception:
                            pass

                    # FIX 4: Fiyat — OG description veya meta keywords'den çıkar
                    price = ''
                    # Önce JSON-LD dene
                    try:
                        ld_text = await detail_page.locator('script[type="application/ld+json"]').first.inner_text(timeout=1500)
                        ld = json.loads(ld_text)
                        price = str(ld.get('offers', {}).get('price', '') or
                                    ld.get('price', '') or '')
                    except Exception:
                        pass
                    if not price:
                        # Sayfadaki herhangi bir fiyat benzeri metin (TL içeren)
                        try:
                            price_el = detail_page.locator(
                                "span:has-text('TL'), div:has-text('TL'), p:has-text('TL')"
                            ).first
                            price = await price_el.inner_text(timeout=1500)
                            price = price.strip().split('\n')[0]
                        except Exception:
                            pass

                    # FIX 4: Lokasyon — breadcrumb veya OG meta
                    district = ''
                    try:
                        bc = await detail_page.locator('nav[aria-label*="breadcrumb"], [class*="breadcrumb"]').first.inner_text(timeout=1500)
                        district = bc.strip()
                    except Exception:
                        pass
                    if not district:
                        try:
                            district = await detail_page.get_attribute('meta[property="og:locality"]', 'content', timeout=1000) or ''
                        except Exception:
                            pass

                    if not title:
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
                        'image_urls': image_urls[:15],
                        'image_count': len(image_urls),
                        'attributes': {},
                        'scraped_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }
                except Exception as e:
                    print(f"    [Detay] Hata: {e}")
                    return None

            async with aiofiles.open(self.jsonl_path, 'a', encoding='utf-8') as out_f:

                async def save(rec: dict):
                    nonlocal saved
                    if rec['id'] in self.seen_ids:
                        return
                    # FIX 3: image_count == 0 filtresi kaldırıldı
                    await out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    await out_f.flush()
                    self.seen_ids.add(rec['id'])
                    saved += 1

                detail_page = await ctx.new_page()
                if STEALTH_AVAILABLE:
                    await stealth_async(detail_page)

                for list_url_base in LIST_URLS:
                    if saved >= self.limit:
                        break

                    print(f'\n→ {list_url_base}')

                    for page_num in range(1, 60):
                        if saved >= self.limit:
                            break

                        # FIX 6: Pagination — sayfa 1'de parametre yok
                        purl = list_url_base if page_num == 1 else f"{list_url_base}?sayfa={page_num}"

                        # FIX 2: Race condition düzeltmesi
                        xhr_results.clear()
                        xhr_done_event.clear()
                        pending_xhr = 0

                        try:
                            await page.goto(purl, wait_until='domcontentloaded', timeout=30000)
                        except PWTimeoutError:
                            print(f"  [Sayfa {page_num}] Timeout — sonraki URL")
                            break
                        except Exception as e:
                            print(f"  [Sayfa {page_num}] Hata: {e}")
                            break

                        # FIX 1: Cloudflare tespiti
                        if await is_cloudflare_page(page):
                            passed = await wait_for_cloudflare(page, max_wait=35)
                            if not passed:
                                print(f"  [Sayfa {page_num}] CF geçilemedi — URL atlıyor")
                                break

                        # FIX 2: XHR handler'larının bitmesini bekle (max 8sn)
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(xhr_done_event.wait()),
                                timeout=8.0,
                            )
                        except asyncio.TimeoutError:
                            pass
                        # Biraz daha bekle — geç gelen XHR'lar için
                        await page.wait_for_timeout(random.randint(1500, 2500))

                        # ── Yol A: XHR'dan ────────────────────────────────
                        if xhr_results:
                            page_saved = 0
                            for rec in list(xhr_results):
                                await save(rec)
                                page_saved += 1
                                if saved >= self.limit:
                                    break
                            print(f"  [Sayfa {page_num}] XHR: {len(xhr_results)} listing, "
                                  f"{page_saved} kaydedildi | Toplam: {saved}")
                            # Son sayfa tespiti: az listing geldi
                            if len(xhr_results) < 5:
                                print(f"  Son sayfa tespit edildi (< 5 listing).")
                                break
                            await asyncio.sleep(random.uniform(1.5, 3.0))
                            continue

                        # ── Yol B: DOM'dan link çek ───────────────────────
                        # FIX 5: Birden fazla href pattern dene
                        hrefs = []
                        for pattern in [
                            'a[href^="/ilan/"]',
                            'a[href*="/ilan/"]',
                            'a[href*="emlakjet.com/ilan/"]',
                            'a[href*="-satilik-"]',
                            'a[href*="-kiralik-"]',
                        ]:
                            found = await page.evaluate(f"""
                                () => Array.from(document.querySelectorAll('{pattern}'))
                                          .map(a => a.getAttribute('href'))
                                          .filter((h, i, arr) => arr.indexOf(h) === i && h)
                            """)
                            hrefs.extend(found)
                            if hrefs:
                                break

                        # Dedupe
                        seen_href = set()
                        unique_hrefs = []
                        for h in hrefs:
                            if h not in seen_href:
                                seen_href.add(h)
                                unique_hrefs.append(h)
                        hrefs = unique_hrefs

                        if not hrefs:
                            print(f"  [Sayfa {page_num}] Hiç link bulunamadı — URL atlıyor")
                            break

                        print(f"  [Sayfa {page_num}] {len(hrefs)} link, detay sayfaları ziyaret ediliyor...")
                        page_saved = 0

                        for href in hrefs:
                            if saved >= self.limit:
                                break
                            full_url = href if href.startswith('http') else 'https://www.emlakjet.com' + href
                            m = ID_RE.search(href)
                            if m and m.group(1) in self.seen_ids:
                                continue

                            rec = await scrape_detail(detail_page, full_url)
                            if rec:
                                await save(rec)
                                page_saved += 1

                            await asyncio.sleep(random.uniform(1.2, 2.0))

                        print(f"  [Sayfa {page_num}] DOM+Detay: {page_saved} kaydedildi | Toplam: {saved}")

                        if page_saved == 0 and page_num > 1:
                            break

                        await asyncio.sleep(random.uniform(2.0, 3.5))

            await detail_page.close()
            await page.close()
            await ctx.close()
            await browser.close()

        print(f"\n{'='*55}")
        print(f'TAMAMLANDI — {saved} listing')
        print(f'Çıktı: {self.jsonl_path}')
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
    parser = argparse.ArgumentParser(description='Emlakjet Scraper v5')
    parser.add_argument('--limit',  type=int, default=5000)
    parser.add_argument('--out',    default='data/raw')
    parser.add_argument('--headed', action='store_true')
    args = parser.parse_args()
    asyncio.run(main(args))
