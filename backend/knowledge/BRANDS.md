# Brand-organized knowledge libraries

Each vehicle brand has its own folder under `knowledge/brands/{slug}/`.

## Layout

```
knowledge/brands/
  mercedes-benz/
    brand.json          # slug, name, WMI prefixes, code patterns
    codes/
      generic.json      # OBD codes (array of entries)
      common.json
      catalog.json
    vins.json           # { vehicles, wmi_prefixes }
    sources.json        # web ingest config
  bmw/
    brand.json
    codes/              # add BMW code JSON files here
    vins.json
    sources.json
```

## Database

SQLite table `brands` registers each library. Diagnostic codes are stored with `brand_slug` (e.g. `mercedes-benz`, `bmw`).

API: `GET /brands` — list registered brands.

Lookup: `GET /codes/P0420?brand=mercedes-benz`

## Ingest (forums / manuals)

```powershell
cd backend
.\.venv\Scripts\python.exe knowledge\ingest_sources.py mercedes-benz
.\.venv\Scripts\python.exe knowledge\ingest_sources.py bmw
```

## Legacy files

Flat `seed_*.json` and `seed_vins.json` in `knowledge/` still work but are **deprecated**. Prefer `brands/{slug}/codes/` and `brands/{slug}/vins.json`.
