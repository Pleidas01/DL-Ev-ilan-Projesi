import re
from pathlib import Path

from scraper.playwright_scraper import build_listing_record


def test_build_listing_record_keeps_long_description_for_feature_extraction():
    description = "asansorlu dogalgazli site icinde " * 60

    record = build_listing_record(
        {
            "id": "listing-1",
            "title": "Test ilan",
            "description": description,
            "price": "25000 TL",
        }
    )

    assert record is not None
    assert len(record["description"]) > 500
    assert len(record["description"]) <= 2000


def test_scraper_source_does_not_keep_500_character_description_cap():
    source = Path("scraper/playwright_scraper.py").read_text(encoding="utf-8")

    assert not re.search(r"description\s*=.*\[:500\]", source)
