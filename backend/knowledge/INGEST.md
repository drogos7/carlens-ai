# Knowledge ingestion (forums, manuals, OBD references)

CarLens can grow its local database from **public sources** you enable in config.

## 1. VIN decode (instant, no scraping)

For any valid 17-character VIN, the backend calls the free **NHTSA vPIC API** and caches the result in `vin_vehicles`:

- Make, model, year, trim, body style, engine hints
- Example: `WDDNG9FB3AA304827` → Mercedes-Benz S-Class, S400 Hybrid, 2010

Restart uvicorn after backend updates.

## 2. Forum / manual crawl (local snippets)

### Configure sources

Edit `knowledge/sources.mercedes.json`:

- Set `"enabled": true` only for URLs you are allowed to crawl
- The script checks **robots.txt** automatically
- Default seeds include [MBWorld](https://mbworld.org/) boards and [OBD-Codes.com](https://www.obd-codes.com/) references
- [BenzWorld](https://www.benzworld.org/) and [MB-Manual](https://mb-manual.com/) are **disabled** until you verify permission

Add specific forum **thread URLs** for best results (homepages alone contain few DTC details).

### Run ingest

From `backend/`:

```powershell
.\.venv\Scripts\python.exe knowledge\ingest_sources.py knowledge\sources.mercedes.json
```

This stores:

- Full page text in `source_documents`
- OBD code snippets in `code_mentions`
- VIN snippets in `vin_mentions`

Settings in JSON:

- `request_delay_seconds` — pause between requests (default 2s)
- `max_pages_per_source` — limit per seed URL
- `max_total_pages` — global cap

### Search ingested text

```http
GET /knowledge/sources/search?q=P0420
GET /knowledge/sources/search?q=WDDNG9FB3AA304827
```

## 3. Recommended workflow

1. Run ingest with a small enabled set first (2–3 URLs)
2. Verify `diagnostics.db` grew and search returns snippets
3. Add more Mercedes forum threads (OM651, DPF, W221, etc.)
4. Merge useful snippets into `brands/{slug}/codes/*.json` manually (see `BRANDS.md`)

## Legal note

Only crawl sites that allow it. Respect robots.txt, rate limits, and terms of service. Do not ingest login-only or paywalled content.
