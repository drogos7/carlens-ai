"""
Brand-organized knowledge library loader.

Layout:
  knowledge/brands/{slug}/
    brand.json       — metadata (name, WMI prefixes, code patterns)
    codes/*.json     — diagnostic code arrays
    vins.json        — { vehicles, wmi_prefixes }
    sources.json     — ingest config (same shape as legacy sources.*.json)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parent
BRANDS_DIR = KNOWLEDGE_DIR / "brands"

# Legacy flat seeds (still loaded if present, for backward compatibility).
LEGACY_SEED_GLOB = "seed_*.json"
LEGACY_VINS_PATH = KNOWLEDGE_DIR / "seed_vins.json"
LEGACY_SOURCES_GLOB = "sources.*.json"


@dataclass(frozen=True)
class BrandConfig:
    slug: str
    name: str
    wmi_prefixes: tuple[str, ...] = ()
    obd_patterns: tuple[str, ...] = ("[PCBU][0-3][0-9A-F]{3}",)
    manufacturer_code_pattern: str | None = None
    sort_order: int = 0
    root: Path = field(default_factory=Path)

    @property
    def codes_dir(self) -> Path:
        return self.root / "codes"

    @property
    def vins_path(self) -> Path:
        return self.root / "vins.json"

    @property
    def sources_path(self) -> Path:
        return self.root / "sources.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_brand_config(brand_dir: Path) -> BrandConfig | None:
    manifest = brand_dir / "brand.json"
    if not manifest.exists():
        return None
    data = _read_json(manifest)
    slug = str(data.get("slug") or brand_dir.name).strip()
    patterns = data.get("code_patterns") or {}
    obd = patterns.get("obd")
    obd_patterns = tuple(obd) if isinstance(obd, list) else (obd,) if isinstance(obd, str) else ("[PCBU][0-3][0-9A-F]{3}",)
    mfg = patterns.get("manufacturer_hex")
    return BrandConfig(
        slug=slug,
        name=str(data.get("name") or slug),
        wmi_prefixes=tuple(data.get("wmi_prefixes") or ()),
        obd_patterns=obd_patterns,
        manufacturer_code_pattern=mfg if isinstance(mfg, str) else None,
        sort_order=int(data.get("sort_order") or 0),
        root=brand_dir,
    )


def discover_brands() -> list[BrandConfig]:
    if not BRANDS_DIR.is_dir():
        return []
    brands: list[BrandConfig] = []
    for child in sorted(BRANDS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        cfg = load_brand_config(child)
        if cfg:
            brands.append(cfg)
    brands.sort(key=lambda b: (b.sort_order, b.name))
    return brands


def iter_code_seed_files(brand: BrandConfig) -> list[Path]:
    paths: list[Path] = []
    codes_dir = brand.codes_dir
    if codes_dir.is_dir():
        paths.extend(sorted(codes_dir.glob("*.json")))
    single = brand.root / "codes.json"
    if single.is_file():
        paths.append(single)
    return paths


def load_code_items(brand: BrandConfig) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in iter_code_seed_files(brand):
        batch = _read_json(path)
        if not isinstance(batch, list):
            logger.warning("Skipping non-array code seed: %s", path)
            continue
        for raw in batch:
            if not isinstance(raw, dict) or "code" not in raw:
                continue
            item = dict(raw)
            item.setdefault("make", brand.name)
            item["brand_slug"] = brand.slug
            items.append(item)
    return items


def load_vins_payload(brand: BrandConfig) -> dict[str, list[Any]]:
    if not brand.vins_path.is_file():
        return {"vehicles": [], "wmi_prefixes": []}
    data = _read_json(brand.vins_path)
    if not isinstance(data, dict):
        return {"vehicles": [], "wmi_prefixes": []}
    return {
        "vehicles": list(data.get("vehicles") or []),
        "wmi_prefixes": list(data.get("wmi_prefixes") or []),
    }


def legacy_code_seed_paths() -> list[Path]:
    paths = sorted(KNOWLEDGE_DIR.glob(LEGACY_SEED_GLOB))
    return [p for p in paths if p.name != LEGACY_VINS_PATH.name]


def legacy_vins_payload() -> dict[str, list[Any]]:
    if not LEGACY_VINS_PATH.is_file():
        return {"vehicles": [], "wmi_prefixes": []}
    data = _read_json(LEGACY_VINS_PATH)
    return {
        "vehicles": list(data.get("vehicles") or []),
        "wmi_prefixes": list(data.get("wmi_prefixes") or []),
    }


def make_to_slug_map(brands: list[BrandConfig]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for brand in brands:
        mapping[brand.name.lower()] = brand.slug
        mapping[brand.slug.lower()] = brand.slug
    return mapping


def slug_for_make(make: str, brands: list[BrandConfig]) -> str | None:
    key = make.strip().lower()
    for brand in brands:
        if brand.name.lower() == key or brand.slug.lower() == key:
            return brand.slug
    return make_to_slug_map(brands).get(key)


def ensure_brand_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS brands (
            slug TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            wmi_prefixes TEXT NOT NULL DEFAULT '[]',
            code_patterns TEXT NOT NULL DEFAULT '{}',
            sort_order INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(diagnostic_codes)").fetchall()}
    if "brand_slug" not in cols:
        conn.execute(
            "ALTER TABLE diagnostic_codes ADD COLUMN brand_slug TEXT NOT NULL DEFAULT 'mercedes-benz'"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_diagnostic_codes_brand_slug ON diagnostic_codes(brand_slug)"
    )

    doc_cols = {row[1] for row in conn.execute("PRAGMA table_info(source_documents)").fetchall()}
    if doc_cols and "brand_slug" not in doc_cols:
        conn.execute(
            "ALTER TABLE source_documents ADD COLUMN brand_slug TEXT NOT NULL DEFAULT ''"
        )

    mention_cols = {row[1] for row in conn.execute("PRAGMA table_info(code_mentions)").fetchall()}
    if mention_cols and "brand_slug" not in mention_cols:
        conn.execute(
            "ALTER TABLE code_mentions ADD COLUMN brand_slug TEXT NOT NULL DEFAULT ''"
        )

    wmi_cols = {row[1] for row in conn.execute("PRAGMA table_info(vin_wmi)").fetchall()}
    if wmi_cols and "brand_slug" not in wmi_cols:
        conn.execute(
            "ALTER TABLE vin_wmi ADD COLUMN brand_slug TEXT NOT NULL DEFAULT ''"
        )


def upsert_brand_row(conn: sqlite3.Connection, brand: BrandConfig) -> None:
    conn.execute(
        """
        INSERT INTO brands (slug, name, wmi_prefixes, code_patterns, sort_order, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(slug) DO UPDATE SET
            name = excluded.name,
            wmi_prefixes = excluded.wmi_prefixes,
            code_patterns = excluded.code_patterns,
            sort_order = excluded.sort_order,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            brand.slug,
            brand.name,
            json.dumps(list(brand.wmi_prefixes)),
            json.dumps(
                {
                    "obd": list(brand.obd_patterns),
                    "manufacturer_hex": brand.manufacturer_code_pattern,
                }
            ),
            brand.sort_order,
        ),
    )


def seed_brands_into_db(
    conn: sqlite3.Connection,
    *,
    normalize_code: Any,
    encode_json: Any,
) -> list[BrandConfig]:
    """Load all brand libraries into SQLite. Returns discovered brands."""
    ensure_brand_tables(conn)
    brands = discover_brands()
    seen_code_keys: set[tuple[str, str, str]] = set()

    for brand in brands:
        upsert_brand_row(conn, brand)
        for item in load_code_items(brand):
            code = normalize_code(item["code"])
            engine = str(item.get("engine") or "Generic")
            key = (code, brand.slug, engine)
            if key in seen_code_keys:
                continue
            seen_code_keys.add(key)
            conn.execute(
                """
                INSERT OR REPLACE INTO diagnostic_codes (
                    code, make, brand_slug, engine, title, description, probable_causes,
                    symptoms, step_by_step_fix, difficulty, safety_warning, sources, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    code,
                    str(item.get("make") or brand.name),
                    brand.slug,
                    engine,
                    item["title"],
                    item["description"],
                    encode_json(item["probable_causes"]),
                    encode_json(item["symptoms"]),
                    encode_json(item["step_by_step_fix"]),
                    item["difficulty"],
                    item["safety_warning"],
                    encode_json(item["sources"]),
                ),
            )

        vins = load_vins_payload(brand)
        for item in vins["vehicles"]:
            conn.execute(
                """
                INSERT OR REPLACE INTO vin_vehicles (
                    vin, make, model, engine, year, body_style, trim, notes, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(item["vin"]).upper(),
                    str(item.get("make") or brand.name),
                    item["model"],
                    item["engine"],
                    item.get("year"),
                    item.get("body_style") or "",
                    item.get("trim") or "",
                    item.get("notes") or "",
                ),
            )
        for item in vins["wmi_prefixes"]:
            conn.execute(
                """
                INSERT OR REPLACE INTO vin_wmi (
                    wmi, make, country, manufacturer, notes, brand_slug, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(item["wmi"]).upper()[:3],
                    str(item.get("make") or brand.name),
                    item.get("country") or "",
                    item.get("manufacturer") or "",
                    item.get("notes") or "",
                    brand.slug,
                ),
            )

    has_brand_code_files = any(iter_code_seed_files(b) for b in brands)
    has_brand_vin_files = any(b.vins_path.is_file() for b in brands)

    # Legacy flat seeds (deprecated — use brands/{slug}/codes/ instead).
    if has_brand_code_files:
        logger.debug("Skipping legacy seed_*.json; brand code libraries present under brands/")
    elif legacy_code_seed_paths():
        logger.info("Loading legacy seed_*.json from knowledge/ root (migrate to brands/{slug}/codes/)")
    for seed_path in [] if has_brand_code_files else legacy_code_seed_paths():
        slug_guess = "mercedes-benz"
        if "bmw" in seed_path.name.lower():
            slug_guess = "bmw"
        for item in _read_json(seed_path):
            if not isinstance(item, dict):
                continue
            make = str(item.get("make") or "Mercedes-Benz")
            brand_slug = slug_for_make(make, brands) or slug_guess
            code = normalize_code(item["code"])
            engine = str(item.get("engine") or "Generic")
            key = (code, brand_slug, engine)
            if key in seen_code_keys:
                continue
            seen_code_keys.add(key)
            conn.execute(
                """
                INSERT OR REPLACE INTO diagnostic_codes (
                    code, make, brand_slug, engine, title, description, probable_causes,
                    symptoms, step_by_step_fix, difficulty, safety_warning, sources, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    code,
                    make,
                    brand_slug,
                    engine,
                    item["title"],
                    item["description"],
                    encode_json(item["probable_causes"]),
                    encode_json(item["symptoms"]),
                    encode_json(item["step_by_step_fix"]),
                    item["difficulty"],
                    item["safety_warning"],
                    encode_json(item["sources"]),
                ),
            )

    if brands and not has_brand_vin_files:
        legacy_vins = legacy_vins_payload()
        if legacy_vins["vehicles"] or legacy_vins["wmi_prefixes"]:
            logger.info("Loading legacy seed_vins.json (migrate to brands/{slug}/vins.json)")
        for item in legacy_vins["vehicles"]:
            make = str(item.get("make") or "")
            brand_slug = slug_for_make(make, brands) or "mercedes-benz"
            conn.execute(
                """
                INSERT OR REPLACE INTO vin_vehicles (
                    vin, make, model, engine, year, body_style, trim, notes, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(item["vin"]).upper(),
                    make,
                    item["model"],
                    item["engine"],
                    item.get("year"),
                    item.get("body_style") or "",
                    item.get("trim") or "",
                    item.get("notes") or "",
                ),
            )
        for item in legacy_vins["wmi_prefixes"]:
            make = str(item.get("make") or "")
            brand_slug = slug_for_make(make, brands) or "mercedes-benz"
            conn.execute(
                """
                INSERT OR REPLACE INTO vin_wmi (
                    wmi, make, country, manufacturer, notes, brand_slug, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(item["wmi"]).upper()[:3],
                    make,
                    item.get("country") or "",
                    item.get("manufacturer") or "",
                    item.get("notes") or "",
                    brand_slug,
                ),
            )

    conn.execute(
        """
        UPDATE diagnostic_codes
        SET brand_slug = 'mercedes-benz'
        WHERE brand_slug = '' OR brand_slug IS NULL
        """
    )
    if not brands:
        brands = _default_brands_fallback(conn)
    return brands


def _default_brands_fallback(conn: sqlite3.Connection) -> list[BrandConfig]:
    """If brands/ folder is empty, register a minimal Mercedes-Benz stub."""
    fallback = BrandConfig(
        slug="mercedes-benz",
        name="Mercedes-Benz",
        wmi_prefixes=("WDD", "WDB", "WDC", "WDF", "W1K", "W1N"),
        root=BRANDS_DIR / "mercedes-benz",
    )
    upsert_brand_row(conn, fallback)
    return [fallback]


def all_wmi_prefixes(brands: list[BrandConfig]) -> tuple[str, ...]:
    seen: list[str] = []
    for brand in brands:
        for prefix in brand.wmi_prefixes:
            if prefix not in seen:
                seen.append(prefix)
    return tuple(seen)
