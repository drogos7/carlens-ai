"""
NHTSA vPIC API — free official VIN decode (no scraping required).

Docs: https://vpic.nhtsa.dot.gov/api/
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

VPIC_DECODE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"


def fetch_nhtsa_decode(vin: str) -> dict[str, str] | None:
    url = VPIC_DECODE_URL.format(vin=vin.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "CarLensAI/1.0 (diagnostic tool)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None

    results = payload.get("Results")
    if not isinstance(results, list):
        return None

    fields: dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        variable = item.get("Variable")
        value = item.get("Value")
        if variable and value and str(value).strip() not in {"", "Not Applicable"}:
            fields[str(variable)] = str(value).strip()

    if not fields.get("Make"):
        return None
    return fields


def nhtsa_to_vehicle_fields(vin: str, fields: dict[str, str]) -> dict[str, str | int | None]:
    year_raw = fields.get("Model Year")
    year: int | None = None
    if year_raw and year_raw.isdigit():
        year = int(year_raw)

    engine_parts = [
        fields.get("Engine Model"),
        fields.get("Displacement (L)"),
        fields.get("Engine Configuration"),
        fields.get("Fuel Type - Primary"),
    ]
    engine = " / ".join(part for part in engine_parts if part) or "See NHTSA record"

    notes_parts = [
        f"Source: NHTSA vPIC ({VPIC_DECODE_URL.format(vin=vin.upper())})",
    ]
    if fields.get("Vehicle Type"):
        notes_parts.append(f"Type: {fields['Vehicle Type']}")
    if fields.get("Drive Type"):
        notes_parts.append(f"Drive: {fields['Drive Type']}")

    return {
        "vin": vin.upper(),
        "make": fields.get("Make", "").title(),
        "model": fields.get("Model", "Unknown model"),
        "engine": engine,
        "year": year,
        "body_style": fields.get("Body Class", ""),
        "trim": fields.get("Trim", "") or fields.get("Series", ""),
        "notes": " | ".join(notes_parts),
    }
