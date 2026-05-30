# Emlakjet Filter Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the Emlakjet rental filter panel as the canonical search schema, populate exact listing facts with the scraper first, then fill only remaining `null` values from DeepSeek text extraction and finally Kimi image extraction.

**Architecture:** Add one central filter registry. Every field has a stable slug, group, type, allowed values where applicable, and permitted evidence sources. The merge order is strict and monotonic: `scraper -> DeepSeek(title + description) -> Kimi(images)`. Later stages may fill `null`; they must never overwrite earlier evidence.

**Tech Stack:** Python, Playwright scraper, DeepSeek V4 Pro JSON extraction, Kimi K2.6 multi-image JSON extraction, ChromaDB metadata, pytest.

---

## Decisions

- Emlakjet's left-side rental filter panel is the schema source of truth: <https://www.emlakjet.com/kiralik-konut>.
- Values explicitly entered under `İlan Bilgileri` or selected under `İlan Özellikleri` are facts.
- Missing optional values are `null`, not `false`.
- Explicit negative structured values such as `Balkon Durumu = Yok` are `false`.
- Property-feature checkboxes are positive-only evidence: selected means `true`; omitted means `null`.
- DeepSeek reads only listing title and description. It fills only `null` fields with text-supported values.
- Kimi reads only images. It fills only still-`null` fields that can be visually supported.
- A user-requested filter is hard: only listings with matching proven values survive retrieval. `null` does not match.
- Keep the existing `imkanlar` concept only as a compatibility/output view if useful. Canonical search fields are typed fields from the registry, not an open-ended list.
- Remove `salon_ozellikleri`.

## Data Contract

Store typed values and provenance separately:

```json
{
  "filter_values": {
    "price_tl": 30000,
    "room_count": "2+1",
    "has_elevator": true,
    "has_aircon": true,
    "near_metro": null
  },
  "filter_sources": {
    "price_tl": "scraper_info",
    "has_elevator": "kimi_image",
    "has_aircon": "scraper_property_feature"
  }
}
```

Keep current top-level facts temporarily for compatibility. Build them from `filter_values`; do not maintain two independent implementations.

## Canonical Registry

Create `schema/emlakjet_filters.py`.

Registry groups:

- Structured: address, price, gross size, room count, building age, floor, total floors, heating type, bathroom count, building condition, occupancy, deed status, furnished status, balcony status, balcony type, gated complex, seller type, listing age.
- `İç Özellikler`: infrastructure, bathroom, decoration, kitchen.
- `Dış Özellikler`: building features, facade, social amenities.
- `Konum Özellikleri`: view, transportation.

Each registry entry must define:

```python
FilterSpec(
    slug="has_elevator",
    group="dis_ozellikler.bina",
    value_type="bool",
    labels=("Asansör",),
    sources=("property_feature", "description_llm", "image_vlm"),
)
```

For categorical values use normalized enums, for example:

```python
FilterSpec(
    slug="balcony_type",
    group="structured",
    value_type="multi_enum",
    labels=("Balkon Tipi",),
    values={
        "Açık Balkon": "acik_balkon",
        "Açık Teras": "acik_teras",
        "Fransız Balkon": "fransiz_balkon",
        "Kapalı Balkon": "kapali_balkon",
        "Kapalı Teras": "kapali_teras",
    },
    sources=("listing_info", "description_llm", "image_vlm"),
)
```

Copy all headings, subheadings, and options from the live rental filter panel into this registry. Do not invent fields outside that panel during this milestone.

> Checkpoint revision (2026-05-30): Task 1 registry was re-verified against the user's authoritative full list. Added category chain (`trade_type`, `property_category`, `property_type`), canonical multi-currency price fields (`price_amount`, `price_currency`), and the omitted `Saten Boya` checkbox. `price_tl` remains a temporary compatibility output outside the canonical registry.

## Task 1: Central Filter Registry

**Files:**
- Create: `schema/__init__.py`
- Create: `schema/emlakjet_filters.py`
- Create: `tests/test_emlakjet_filters.py`

- [x] Add tests that assert representative fields from every group exist, slugs are unique, labels are unique within their source, and `salon_ozellikleri` is absent.
- [x] Implement `FilterSpec`, registry constants, label normalization, and lookup helpers:

```python
def spec_for_info_label(label: str) -> FilterSpec | None: ...
def spec_for_property_feature(label: str) -> FilterSpec | None: ...
def empty_filter_values() -> dict[str, Any]: ...
```

- [x] Run: `.venv/bin/python3 -m pytest -q tests/test_emlakjet_filters.py`
- [x] Commit: `feat: add canonical emlakjet filter registry`

## Task 2: Scraper Captures Exact Facts

**Files:**
- Modify: `scraper/playwright_scraper.py`
- Modify: `scraper/cleaner.py`
- Modify: `tests/test_scraper_extraction.py`
- Modify: `tests/test_cleaner_preserves_fields.py`

- [x] Extend saved fixture tests with `Balkon Durumu`, `Balkon Tipi`, `Banyo Sayısı`, `Kullanım Durumu`, and selected `İlan Özellikleri`.
- [x] Replace the hand-maintained `INFO_FIELD_MAP` with registry-backed lookup.
- [x] Parse listing-info values into typed `filter_values`.
- [x] Convert every selected `propertyFeatures` item into positive boolean or enum evidence using the registry.
- [x] Add `filter_sources[field] = "scraper_info"` or `"scraper_property_feature"`.
- [x] Preserve raw `attributes` and `property_features` for auditability.
- [x] Verify omitted property features remain `null`; do not set them to `false`.
- [x] Run: `.venv/bin/python3 -m pytest -q tests/test_scraper_extraction.py tests/test_cleaner_preserves_fields.py`
- [x] Commit: `feat: populate canonical filters from scraper facts`

## Task 3: DeepSeek Null-Only Text Enrichment

> Task 2 checkpoint revision (2026-05-30): deterministic scraper extraction also fills canonical category chain and multi-currency listing price. Temporary top-level `price_tl` is populated only when canonical `price_currency` is `TL`.

**Files:**
- Modify: `labeling/run_labeling.py`
- Modify: `tests/test_run_labeling.py`

- [x] Add tests proving the prompt contains title, description, and only the current null-field schema.
- [x] Add tests proving DeepSeek cannot overwrite scraper values.
- [x] Build the prompt from title + description only. Do not send `attributes` or `property_features`.
- [x] Ask DeepSeek for only fields whose registry allows `description_llm` and whose value is currently `null`.
- [x] Validate returned keys and enum values against the registry.
- [x] Merge accepted values with `filter_sources[field] = "deepseek_description"`.
- [x] Run: `.venv/bin/python3 -m pytest -q tests/test_run_labeling.py`
- [x] Commit: `feat: fill null filters from listing description`

## Task 4: Kimi Null-Only Visual Enrichment

**Files:**
- Modify: `llm/shootout_vision.py`
- Modify: `labeling/run_labeling.py`
- Modify: `tests/test_run_labeling.py`
- Modify: `llm/gold_benchmark.py`

- [ ] Remove `salon_ozellikleri`.
- [ ] Add tests proving the Kimi prompt includes only null fields permitted for `image_vlm`.
- [ ] Add tests proving Kimi cannot overwrite scraper or DeepSeek values.
- [ ] Start with visually defensible fields: elevator, air conditioner, balcony status/type, kitchen type, shower cabin, bathtub, jacuzzi, parking, pool, garden/landscaping, playground, sports area, and view types.
- [ ] Keep image inference positive-only for booleans: clear evidence may produce `true`; absence produces `null`.
- [ ] Preserve `VISION_MAX_IMAGE_EDGE`; use `512` for the first validation run and `--vision-chunk-size 0` unless timeout evidence requires chunking.
- [ ] Run: `.venv/bin/python3 -m pytest -q tests/test_run_labeling.py tests/test_gold_helpers.py`
- [ ] Commit: `feat: fill visually supported null filters with kimi`

## Task 5: Dynamic Metadata and Hard Filters

**Files:**
- Modify: `indexing/composer.py`
- Modify: `retrieval/retriever.py`
- Modify: `llm/shootout.py`
- Modify: `tests/test_composer.py`
- Modify: `tests/test_retriever.py`

- [ ] Add tests for numeric, categorical, multi-enum, and boolean metadata.
- [ ] Flatten canonical fields into Chroma metadata using stable registry slugs.
- [ ] Update query-slot schema so DeepSeek extracts requested canonical fields.
- [ ] Build Chroma `where` conditions for every requested supported field.
- [ ] Require exact proven matches: queried `true` excludes `false` and `null`.
- [ ] Keep BGE-M3 + reranker for ranking after hard filtering.
- [ ] Run: `.venv/bin/python3 -m pytest -q tests/test_composer.py tests/test_retriever.py tests/test_llm_shootout.py`
- [ ] Commit: `feat: query canonical filters as hard constraints`

## Task 6: Re-scrape and Validation

**Files:**
- Modify after implementation: `docs/PROJECT.md`
- Modify after implementation: `docs/STATUS.md`
- Modify after implementation: `docs/HANDOFF.md`

- [ ] Run full tests: `.venv/bin/python3 -m pytest -q`
- [ ] Re-scrape Istanbul rental listings so newly captured structured fields such as balcony status are available.
- [ ] Re-run cleaner and print coverage by source: scraper info, property feature, DeepSeek, Kimi, unresolved null.
- [ ] Rebuild manual gold templates against the new registry before spending on APIs.
- [ ] Run a small DeepSeek pre-flight.
- [ ] Run a 16-listing Kimi pre-flight at `512px`, `--batch-size 20`, `--confidence-mode self`, `--vision-chunk-size 0`, and an explicit cost cap.
- [ ] Rebuild Chroma and evaluate retrieval using manually verified demo queries.
- [ ] Document measured accuracy, latency, cost, and unresolved null coverage.
- [ ] Commit: `docs: record canonical filter pipeline results`

## Acceptance Criteria

- One registry defines every supported filter and source rule.
- Scraper fills exact listing-info and property-feature facts without model calls.
- DeepSeek sees only title + description and fills only null fields.
- Kimi sees only images and fills only visually supported null fields.
- Later stages never overwrite earlier evidence.
- Requested filters are hard constraints; `null` does not match.
- `salon_ozellikleri` no longer exists.
- Re-scraped dataset coverage and retrieval quality are reported from files, not agent claims.
