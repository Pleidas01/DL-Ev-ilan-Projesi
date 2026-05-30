"""
Emlakjet Scraper v6 — Filtre Desteği
======================================
v5'ten yapılan değişiklikler:
  8. Filtreler: --tip, --sehir, --ilce, --mahalle, --oda, --ilan-yasi
     argümanları eklendi. URL'ler dinamik olarak oluşturulur.
     Oda sayısı ve ilan yaşı için kayıt düzeyinde post-filtre de uygulanır.

Kullanım:
    python scraper/playwright_scraper.py --limit 200 --out data/raw
    python scraper/playwright_scraper.py --limit 500 --tip satilik --sehir istanbul
    python scraper/playwright_scraper.py --limit 200 --tip kiralik --sehir ankara --ilce cankaya
    python scraper/playwright_scraper.py --limit 300 --oda 3+1 --ilan-yasi 30
    python scraper/playwright_scraper.py --limit 5 --headed  # debug
"""

import asyncio
import html as html_module
import json
import random
import re
import time
import argparse
from pathlib import Path

import aiofiles
from tqdm import tqdm

from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError, Response
from schema.emlakjet_filters import (
    extract_scraper_filter_facts,
    raw_attribute_key_for_info_label,
)

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
    r'(?:listing|listings?)/\d+/[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-+=]+)?',
    re.IGNORECASE,
)

# Filtre verilmediğinde varsayılan URL listesi
DEFAULT_LIST_URLS = [
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

# Oda sayısı → Emlakjet URL slug eşlemesi
ODA_SLUG_MAP = {
    "1+0": "1_0", "1+1": "1_1", "2+0": "2_0", "2+1": "2_1",
    "3+1": "3_1", "3+2": "3_2", "4+1": "4_1", "4+2": "4_2",
    "5+1": "5_1", "5+2": "5_2",
}


def build_list_urls(tip: str, sehir: str, ilce: str, mahalle: str,
                    oda: str, ilan_yasi: int) -> list[str]:
    """CLI argümanlarından Emlakjet arama URL'lerini oluşturur."""
    tips = ['satilik', 'kiralik'] if tip == 'hepsi' else [tip]
    urls = []
    for t in tips:
        path = f"https://www.emlakjet.com/{t}-konut/"
        if sehir:
            path += f"{sehir}/"
        if ilce:
            path += f"{ilce}/"
        if mahalle:
            path += f"{mahalle}/"

        params: list[str] = []
        if oda:
            slug = ODA_SLUG_MAP.get(oda, oda.replace('+', '_'))
            params.append(f"oda-sayisi[]={slug}")
        if ilan_yasi:
            params.append(f"ilanYasi={ilan_yasi}")
        if params:
            path += "?" + "&".join(params)
        urls.append(path)
    return urls


def record_matches_filters(rec: dict, oda: str, ilan_yasi: int) -> bool:
    """Scrape edilen kaydı filtre kriterlerine göre kontrol eder (post-filtre)."""
    if oda:
        room = str(rec.get('attributes', {}).get('roomCount', ''))
        oda_norm = oda.replace(' ', '').lower()
        room_norm = room.replace(' ', '').lower()
        if room_norm and oda_norm not in room_norm:
            return False

    if ilan_yasi:
        published = str(rec.get('attributes', {}).get('publishedAt', ''))
        if published:
            try:
                from datetime import datetime, timezone
                pub_dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                delta_days = (now - pub_dt).days
                if delta_days > ilan_yasi:
                    return False
            except Exception:
                pass
    return True

ID_RE = re.compile(r'-(\d{6,10})(?:/|$)')
MAX_DESCRIPTION_LEN = 4000

# İlan Bilgileri tablosu: <li><span>key</span><span>value</span></li>
INFO_ITEM_RE = re.compile(
    r'<li>\s*<span[^>]*>(?P<key>[^<]+)</span>\s*<span[^>]*>(?P<val>[^<]*)</span>\s*</li>',
    re.IGNORECASE,
)

def _info_label_key(label: str) -> str:
    """Türkçe etiketleri ASCII-benzeri anahtara çevir (Windows locale güvenli)."""
    text = label.strip()
    try:
        text = text.encode('latin1').decode('utf-8')
    except UnicodeError:
        pass
    text = text.lower()
    for src, dst in (
        ('ı', 'i'), ('ş', 's'), ('ğ', 'g'), ('ü', 'u'), ('ö', 'o'), ('ç', 'c'),
        ('İ', 'i'), ('I', 'i'),
    ):
        text = text.replace(src, dst)
    for src, dst in (
        ('ı', 'i'), ('ş', 's'), ('ğ', 'g'), ('ü', 'u'), ('ö', 'o'), ('ç', 'c'),
        ('İ', 'i'), ('I', 'i'),
    ):
        text = text.replace(src, dst)
    text = text.replace('\u0307', '')
    return text

# Cloudflare challenge sayfası belirteci
CF_INDICATORS = ["cf-browser-verification", "Cloudflare", "Just a moment", "cf_chl"]


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def extract_images_from_html(html: str, listing_id: str = '') -> list[str]:
    urls = list(dict.fromkeys(IMG_PATTERN.findall(html)))
    result = []
    for u in urls:
        u = re.sub(r'/(?:resize|thumb)/\d+/\d+/', '/', u)
        result.append(u)
    result = list(dict.fromkeys(result))
    if listing_id:
        needle = f'/listing/{listing_id}/'
        result = [u for u in result if needle in u]
    return result


def _html_section(html: str, start_marker: str, end_markers: list[str], max_len: int = 20000) -> str:
    start = html.find(start_marker)
    if start < 0:
        return ''
    chunk = html[start:]
    end = min(len(chunk), max_len)
    for marker in end_markers:
        pos = chunk.find(marker, len(start_marker))
        if pos > 0:
            end = min(end, pos)
    return chunk[:end]


def _strip_html_text(fragment: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', fragment, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'<.*$', '', text, flags=re.DOTALL)
    return html_module.unescape(re.sub(r'\s+', ' ', text).strip())


def parse_listing_info_table(html: str) -> dict[str, str]:
    """İlan Bilgileri (#ilan-hakkinda) key/value çiftlerini çıkarır."""
    section = _html_section(html, 'İlan Bilgileri', ['İlan Özellikleri', ' Açıklaması'])
    if not section:
        section = _html_section(html, 'id="ilan-hakkinda"', ['İlan Özellikleri', ' Açıklaması'])
    if not section and INFO_ITEM_RE.search(html):
        section = html
    attrs: dict[str, str] = {}
    for match in INFO_ITEM_RE.finditer(section):
        key = match.group('key').strip()
        val = match.group('val').strip()
        attr_key = raw_attribute_key_for_info_label(_info_label_key(key))
        if attr_key and val:
            attrs[attr_key] = val
    return attrs


def parse_description_from_dom_html(html: str) -> str:
    """'{Title} Açıklaması' h2 başlığının altındaki gerçek açıklama metni."""
    match = re.search(
        r'<h2[^>]*>[^<]*Açıklaması</h2>(.*?)(?:'
        r'id="konum-bilgisi"|'
        r'>İlan Özellikleri<|'
        r'>Fiyat Bilgisi<|'
        r'>Bölge Raporu<|'
        r'>Firma Künyesi<|'
        r'<strong[^>]*>\s*İlan Özellikleri|'
        r'<strong[^>]*>\s*Fiyat Bilgisi)',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ''
    return _strip_html_text(match.group(1))[:MAX_DESCRIPTION_LEN]


def parse_property_features(html: str) -> list[str]:
    """İlan Özellikleri sekmesindeki tüm madde işaretli özellikler."""
    section = _html_section(
        html,
        'İlan Özellikleri',
        ['Fiyat Bilgisi', 'Bölge Raporu', 'Firma Künyesi', 'Benzer İlanlar'],
        max_len=30000,
    )
    features: list[str] = []
    for match in re.finditer(r'<li>([^<]+)</li>', section):
        feat = html_module.unescape(match.group(1).strip())
        if feat and feat not in features:
            features.append(feat)
    return features


def is_template_description(text: str) -> bool:
    """JSON-LD şablon açıklaması (ajans + m² + fiyat özeti) mi?"""
    if not text:
        return True
    return bool(re.search(r'Emlakjet\s*-\s*#\d{6,10}\s*$', text.strip()))


def _description_from_sources(html: str, ld_product: dict) -> str:
    description = parse_description_from_dom_html(html)
    if not description:
        ld_desc = str(ld_product.get('description') or '').strip()
        if ld_desc and not is_template_description(ld_desc):
            description = ld_desc
    return description[:MAX_DESCRIPTION_LEN]


async def _resolve_description(detail_page, html: str, ld_product: dict) -> tuple[str, str]:
    """DOM açıklaması; boşsa bekle, genişlet, HTML'i yenile ve tekrar dene."""
    description = _description_from_sources(html, ld_product)
    if description:
        return description, html

    try:
        await detail_page.wait_for_selector(
            'h2:has-text("Açıklaması")', timeout=8000,
        )
        expand = detail_page.get_by_text('Daha Fazla Gör', exact=False).first
        if await expand.count() > 0:
            await expand.click(timeout=2500)
            await detail_page.wait_for_timeout(1000)
        html = await detail_page.content()
        description = _description_from_sources(html, ld_product)
    except Exception:
        pass

    return description, html


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
            # Gerçek ilan tespiti: price VEYA listing-spesifik alan içermeli
            # → şehir/kategori objelerini (sadece id+name) dışlar
            has_price = bool(
                obj.get('price') or obj.get('salePrice') or obj.get('rentPrice')
                or obj.get('priceText') or obj.get('formattedPrice')
            )
            has_listing_field = any(k in obj for k in [
                'roomCount', 'grossSize', 'netSize', 'buildingAge',
                'propertyType', 'tradeType', 'slug', 'photos', 'images',
                'coverImage', 'thumbnailUrl', 'listingId',
            ])
            if lid and title and (has_price or has_listing_field):
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

    description = str(obj.get('description') or obj.get('shortDescription') or '').strip()[:MAX_DESCRIPTION_LEN]

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

    filter_values, filter_sources = extract_scraper_filter_facts(attrs)
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
        'filter_values': filter_values,
        'filter_sources': filter_sources,
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
    def __init__(self, out_dir: Path, limit: int, headless: bool = True,
                 list_urls: list[str] | None = None,
                 filter_oda: str = '', filter_ilan_yasi: int = 0):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / 'listings.jsonl'
        self.limit = limit
        self.headless = headless
        self.list_urls = list_urls or DEFAULT_LIST_URLS
        self.filter_oda = filter_oda
        self.filter_ilan_yasi = filter_ilan_yasi
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
            # pending counter list olarak tutulur — async closure içinde
            # nonlocal int mutasyonu Python'da güvenilmez olabiliyor.
            xhr_results: list[dict] = []
            xhr_done_event = asyncio.Event()
            pending_xhr = [0]  # [0] = mutable counter

            async def on_response(response: Response):
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
                # Konum/kategori API'lerini atla — ilan verisi içermiyor
                if any(skip in url for skip in [
                    '/location/', '/city', '/district', '/neighborhood',
                    '/category', '/filter/', '/autocomplete',
                ]):
                    return

                pending_xhr[0] += 1
                try:
                    data = await response.json()
                    items = parse_json_listings(data)
                    for item in items:
                        rec = build_listing_record(item)
                        if rec and rec['id'] not in self.seen_ids:
                            xhr_results.append(rec)
                    if items:
                        print(f"    [XHR] {url[:80]} -> {len(items)} listing")
                except Exception:
                    pass
                finally:
                    pending_xhr[0] -= 1
                    if pending_xhr[0] <= 0:
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

                    try:
                        await detail_page.wait_for_selector('#ilan-hakkinda', timeout=12000)
                    except Exception:
                        pass
                    try:
                        await detail_page.wait_for_selector(
                            'h2:has-text("Açıklaması")', timeout=5000,
                        )
                    except Exception:
                        pass

                    html = await detail_page.content()

                    m = ID_RE.search(listing_url)
                    lid = m.group(1) if m else ''

                    image_urls = extract_images_from_html(html, lid)

                    # ── 1. window.dataLayer — en zengin kaynak ────────────
                    # {ilan_fiyat, city, town, neighborhood, district,
                    #  oda_sayisi, property_subcategory, property_status, ...}
                    dl: dict = {}
                    try:
                        dl = await detail_page.evaluate("""
                            () => {
                                const layers = window.dataLayer || [];
                                const obj = {};
                                for (const item of layers) {
                                    if (item && typeof item === 'object') {
                                        Object.assign(obj, item);
                                    }
                                }
                                return obj;
                            }
                        """) or {}
                    except Exception:
                        pass

                    # ── 2. JSON-LD Product schema ─────────────────────────
                    ld_product: dict = {}
                    try:
                        ld_scripts = await detail_page.evaluate("""
                            () => Array.from(
                                document.querySelectorAll('script[type="application/ld+json"]')
                            ).map(s => s.textContent)
                        """)
                        for ld_text in (ld_scripts or []):
                            try:
                                ld = json.loads(ld_text)
                                if isinstance(ld, dict) and ld.get('@type') == 'Product':
                                    ld_product = ld
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # ── Başlık ────────────────────────────────────────────
                    title = str(ld_product.get('name') or dl.get('item_name') or '').strip()
                    if not title:
                        try:
                            title = await detail_page.locator('h1').first.inner_text(timeout=3000)
                            title = title.strip()
                        except Exception:
                            pass
                    if not title:
                        try:
                            title = await detail_page.get_attribute(
                                'meta[property="og:title"]', 'content', timeout=1500) or ''
                        except Exception:
                            pass

                    # ── Fiyat ─────────────────────────────────────────────
                    price_raw = (dl.get('ilan_fiyat') or
                                 ld_product.get('offers', {}).get('price') or
                                 dl.get('price') or '')
                    if isinstance(price_raw, (int, float)) and price_raw:
                        price = f"{int(price_raw):,} TL".replace(',', '.')
                    elif price_raw:
                        price = f"{price_raw} TL" if not str(price_raw).endswith('TL') else str(price_raw)
                    else:
                        price = ''

                    # ── Konum ─────────────────────────────────────────────
                    city = str(dl.get('city') or '').replace('-', ' ').title()
                    town = str(dl.get('town') or '').replace('-', ' ').title()
                    nbhd = str(dl.get('neighborhood') or '').replace('-', ' ').title()
                    district = ' - '.join(filter(None, [city, town, nbhd]))

                    # ── DOM: İlan Bilgileri / Açıklama / Özellikler ─────────
                    dom_info = parse_listing_info_table(html)
                    dom_features = parse_property_features(html)
                    description, html = await _resolve_description(
                        detail_page, html, ld_product,
                    )

                    # ── Attributes ────────────────────────────────────────
                    attrs: dict = {}
                    if dl.get('oda_sayisi'):
                        attrs['roomCount'] = dl['oda_sayisi']
                    if dl.get('property_subcategory'):
                        attrs['propertyType'] = dl['property_subcategory']
                    if dl.get('property_status'):
                        attrs['tradeType'] = dl['property_status']
                    if dl.get('property_category'):
                        attrs['category'] = dl['property_category']
                    for dl_key, attr_key in [
                        ('gross_m2', 'grossSize'), ('net_m2', 'netSize'),
                        ('building_age', 'buildingAge'), ('floor', 'floor'),
                        ('ilan_yayinlanma_tarihi', 'publishedAt'),
                    ]:
                        if dl.get(dl_key):
                            attrs[attr_key] = dl[dl_key]
                    for key, val in dom_info.items():
                        attrs[key] = val
                    if dom_features:
                        attrs['propertyFeatures'] = dom_features
                    filter_values, filter_sources = extract_scraper_filter_facts(attrs, dom_features)

                    if not title:
                        return None

                    if not lid:
                        lid = str(dl.get('item_id') or listing_url[-12:])

                    final_images = list(dict.fromkeys(image_urls))

                    return {
                        'id': lid,
                        'url': listing_url,
                        'title': title,
                        'price': price,
                        'description': description,
                        'district': district,
                        'image_urls': final_images[:20],
                        'image_count': len(final_images),
                        'attributes': attrs,
                        'filter_values': filter_values,
                        'filter_sources': filter_sources,
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
                    if not record_matches_filters(rec, self.filter_oda, self.filter_ilan_yasi):
                        return
                    await out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    await out_f.flush()
                    self.seen_ids.add(rec['id'])
                    saved += 1

                detail_page = await ctx.new_page()
                if STEALTH_AVAILABLE:
                    await stealth_async(detail_page)

                for list_url_base in self.list_urls:
                    if saved >= self.limit:
                        break

                    print(f'\n-> {list_url_base}')

                    consecutive_empty = 0  # ardışık sıfır-kayıt sayfa sayacı

                    for page_num in range(1, 300):
                        if saved >= self.limit:
                            break

                        # FIX 6: Pagination — sayfa 1'de parametre yok
                        purl = list_url_base if page_num == 1 else f"{list_url_base}?sayfa={page_num}"

                        # FIX 2: Race condition düzeltmesi
                        xhr_results.clear()
                        xhr_done_event.clear()
                        pending_xhr[0] = 0

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
                            if page_saved == 0:
                                consecutive_empty += 1
                            else:
                                consecutive_empty = 0
                            # Gerçek son sayfa: XHR az döndü VE zaten hepsi görülmüş
                            if len(xhr_results) < 5 and consecutive_empty >= 2:
                                print(f"  Son sayfa tespit edildi.")
                                break
                            if consecutive_empty >= 5:
                                print(f"  5 ardışık boş sayfa — URL atlıyor.")
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
                            # Gerçek boş sayfa: site bu sayfayı gösteremiyor
                            consecutive_empty += 1
                            print(f"  [Sayfa {page_num}] Hiç link bulunamadı ({consecutive_empty}/3)")
                            if consecutive_empty >= 3:
                                print(f"  Son sayfa tespit edildi — URL atlıyor.")
                                break
                            await asyncio.sleep(random.uniform(2.0, 3.5))
                            continue

                        # Sayfadaki linklerden kaçı yeni (henüz scraplanmamış)?
                        new_hrefs = []
                        for h in hrefs:
                            m = ID_RE.search(h)
                            if not m or m.group(1) not in self.seen_ids:
                                new_hrefs.append(h)

                        if not new_hrefs:
                            # Tüm linkler zaten scraplanmış — sayfayı geç, sayacı artırma
                            print(f"  [Sayfa {page_num}] {len(hrefs)} link, tümü zaten indirilmiş — geçiliyor")
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                            continue

                        print(f"  [Sayfa {page_num}] {len(hrefs)} link ({len(new_hrefs)} yeni), detay sayfaları ziyaret ediliyor...")
                        page_saved = 0

                        for href in new_hrefs:
                            if saved >= self.limit:
                                break
                            full_url = href if href.startswith('http') else 'https://www.emlakjet.com' + href

                            rec = await scrape_detail(detail_page, full_url)
                            if rec:
                                await save(rec)
                                page_saved += 1

                            await asyncio.sleep(random.uniform(1.2, 2.0))

                        print(f"  [Sayfa {page_num}] DOM+Detay: {page_saved} kaydedildi | Toplam: {saved}")

                        if page_saved == 0:
                            consecutive_empty += 1
                        else:
                            consecutive_empty = 0

                        if consecutive_empty >= 3:
                            print(f"  3 ardışık gerçek boş sayfa — URL atlıyor.")
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
    # Filtre verilmişse dinamik URL oluştur, verilmemişse varsayılan listeyi kullan
    herhangi_filtre = any([args.tip != 'hepsi', args.sehir, args.ilce,
                           args.mahalle, args.oda, args.ilan_yasi])
    if herhangi_filtre:
        list_urls = build_list_urls(
            tip=args.tip,
            sehir=args.sehir,
            ilce=args.ilce,
            mahalle=args.mahalle,
            oda=args.oda,
            ilan_yasi=args.ilan_yasi,
        )
        print("\n[Filtreler]")
        print(f"  Tip      : {args.tip}")
        print(f"  Şehir    : {args.sehir or '—'}")
        print(f"  İlçe     : {args.ilce or '—'}")
        print(f"  Mahalle  : {args.mahalle or '—'}")
        print(f"  Oda      : {args.oda or '—'}")
        print(f"  İlan yaşı: {f'son {args.ilan_yasi} gün' if args.ilan_yasi else '—'}")
        print(f"  URL(ler) : {list_urls}")
    else:
        list_urls = None
        print("\n[Filtre yok — varsayılan URL listesi kullanılıyor]")

    scraper = EmlakjetScraper(
        out_dir=Path(args.out),
        limit=args.limit,
        headless=not args.headed,
        list_urls=list_urls,
        filter_oda=args.oda,
        filter_ilan_yasi=args.ilan_yasi,
    )
    await scraper.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Emlakjet Scraper v6',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Örnekler:
  python scraper/playwright_scraper.py --limit 200
  python scraper/playwright_scraper.py --limit 500 --tip satilik --sehir istanbul
  python scraper/playwright_scraper.py --limit 200 --tip kiralik --sehir ankara --ilce cankaya
  python scraper/playwright_scraper.py --limit 300 --oda 3+1
  python scraper/playwright_scraper.py --limit 300 --oda 2+1 --ilan-yasi 30 --sehir izmir
"""
    )
    parser.add_argument('--limit', type=int, default=5000,
                        help='Kaç ilan toplanacak (varsayılan: 5000)')
    parser.add_argument('--out', default='data/raw',
                        help='Çıktı klasörü (varsayılan: data/raw)')
    parser.add_argument('--headed', action='store_true',
                        help='Tarayıcıyı görünür aç (debug için)')
    parser.add_argument('--tip', default='hepsi',
                        choices=['satilik', 'kiralik', 'hepsi'],
                        help='İlan tipi (varsayılan: hepsi)')
    parser.add_argument('--sehir', default='',
                        help='Şehir slug (istanbul, ankara, izmir, bursa ...)')
    parser.add_argument('--ilce', default='',
                        help='İlçe slug (kadikoy, besiktas, cankaya ...)')
    parser.add_argument('--mahalle', default='',
                        help='Mahalle slug (moda, etiler ...)')
    parser.add_argument('--oda', default='',
                        help='Oda sayısı (1+1, 2+1, 3+1, 4+1 ...)')
    parser.add_argument('--ilan-yasi', type=int, default=0, dest='ilan_yasi',
                        help='Son kaç günde yayınlanan ilanlar (örn: 30)')
    args = parser.parse_args()
    asyncio.run(main(args))
