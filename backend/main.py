"""
FastAPI MVP: analyze diagnostic tool screen images and return a technical fix guide.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sqlite3
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from anthropic import AsyncAnthropic
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dtc_decode import decode_dtc
from knowledge.brand_registry import (
    BrandConfig,
    all_wmi_prefixes,
    discover_brands,
    seed_brands_into_db,
    slug_for_make,
)
from nhtsa_vin import fetch_nhtsa_decode, nhtsa_to_vehicle_fields
from vin_decode import decode_vin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_DB_PATH = KNOWLEDGE_DIR / "diagnostics.db"
VIN_LENGTH = 17
_BRANDS: list[BrandConfig] = []
VIN_CHARS = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")
    max_image_bytes: int = Field(default=15 * 1024 * 1024, alias="MAX_IMAGE_BYTES")  # 15 MB
    cors_origins: str = Field(
        default=(
            "http://localhost:8081,http://127.0.0.1:8081,"
            "http://localhost:19006,http://127.0.0.1:19006,"
            "http://localhost:8082,http://127.0.0.1:8082"
        ),
        alias="CORS_ORIGINS",
        description="Comma-separated origins allowed for browser clients (Expo web / dev).",
    )


settings = Settings()

_rapid_ocr_engine: Any = None


def _parse_cors_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class VinSegmentResponse(BaseModel):
    label: str
    positions: str
    value: str
    meaning: str


class VinDecodeResponse(BaseModel):
    vin: str
    wmi: str
    vds: str
    vis: str
    model_year: int | None = None
    check_digit: str
    check_digit_valid: bool | None = None
    summary: str
    segments: list[VinSegmentResponse]


class DtcSegmentResponse(BaseModel):
    label: str
    position: str
    value: str
    meaning: str


class DtcDecodeResponse(BaseModel):
    code: str
    is_standard_format: bool
    summary: str
    segments: list[DtcSegmentResponse]


class CodeDependency(BaseModel):
    """Relationship between two detected codes."""
    from_code: str
    to_code: str
    relation: Literal["root_cause", "symptom", "related", "same_system"]
    explanation: str


class DetectedCodeEntry(BaseModel):
    """One detected fault code with its DB lookup result (if available)."""
    code: str
    title: str | None = None
    description: str | None = None
    probable_causes: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    step_by_step_fix: list[str] = Field(default_factory=list)
    difficulty: Literal["Easy", "Medium", "Hard"] | None = None
    safety_warning: str | None = None
    in_database: bool = False
    brand_slug: str | None = None


class AnalyzeErrorResponse(BaseModel):
    """Structured response returned to API clients."""

    scan_type: Literal["dtc", "vin"] = Field(default="dtc", description="Whether the scan matched a DTC or a VIN.")
    detected_code: str = Field(description="Primary OBD-II code (dtc) or VIN string (vin).")
    detected_vin: str | None = Field(default=None, description="17-character VIN when scan_type is vin.")
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    vehicle_engine: str | None = None
    vehicle_year: int | None = None
    vin_decode: VinDecodeResponse | None = None
    dtc_decode: DtcDecodeResponse | None = None
    additional_codes: list[DetectedCodeEntry] = Field(default_factory=list)
    dependencies: list[CodeDependency] = Field(default_factory=list)
    probable_cause: str
    step_by_step_fix: list[str] = Field(default_factory=list)
    estimated_difficulty: Literal["Easy", "Medium", "Hard"]
    safety_warning: str

    @field_validator("step_by_step_fix", mode="before")
    @classmethod
    def ensure_steps(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split("\n") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip() for s in v if str(s).strip()]
        return []


class LocalScanDebugResponse(BaseModel):
    text: str
    detected_candidates: list[str]
    detected_vins: list[str] = Field(default_factory=list)


class VinVehicleResponse(BaseModel):
    vin: str
    make: str
    model: str
    engine: str
    year: int | None = None
    body_style: str = ""
    trim: str = ""
    notes: str = ""


class WmiPrefixResponse(BaseModel):
    wmi: str
    make: str
    country: str = ""
    manufacturer: str = ""
    notes: str = ""


class BrandResponse(BaseModel):
    slug: str
    name: str
    wmi_prefixes: list[str] = Field(default_factory=list)
    manufacturer_code_pattern: str | None = None


class DiagnosticCodeResponse(BaseModel):
    code: str
    make: str
    brand_slug: str | None = None
    engine: str
    title: str
    description: str
    probable_causes: list[str]
    symptoms: list[str]
    step_by_step_fix: list[str]
    difficulty: Literal["Easy", "Medium", "Hard"]
    safety_warning: str
    sources: list[str]


class KnowledgeSearchResponse(BaseModel):
    query: str
    count: int
    results: list[DiagnosticCodeResponse]


class LookupQueryBody(BaseModel):
    q: str = Field(description="Fault code or VIN to look up (OBD-II, BMW E+hex, or 17-char VIN).")


class SourceMentionResponse(BaseModel):
    code: str
    url: str
    title: str
    snippet: str


class VinMentionResponse(BaseModel):
    vin: str
    url: str
    title: str
    snippet: str


class SourceSearchResponse(BaseModel):
    query: str
    code_mentions: list[SourceMentionResponse]
    vin_mentions: list[VinMentionResponse]


class VisionExtraction(BaseModel):
    """Internal model for strict JSON from the vision model."""

    detected_code: str = ""
    car_model: str = ""
    error_description: str = ""
    extraction_confidence: Literal["high", "medium", "low", "none"] = "none"
    extraction_notes: str = ""
    analysis: AnalyzeErrorResponse


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


def _db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(KNOWLEDGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _encode_json(value: list[str]) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decode_json(value: str) -> list[str]:
    data = json.loads(value)
    return [str(item) for item in data] if isinstance(data, list) else []


def _row_to_diagnostic_code(row: sqlite3.Row) -> DiagnosticCodeResponse:
    keys = row.keys()
    brand_slug = str(row["brand_slug"]) if "brand_slug" in keys and row["brand_slug"] else None
    return DiagnosticCodeResponse(
        code=row["code"],
        make=row["make"],
        brand_slug=brand_slug,
        engine=row["engine"],
        title=row["title"],
        description=row["description"],
        probable_causes=_decode_json(row["probable_causes"]),
        symptoms=_decode_json(row["symptoms"]),
        step_by_step_fix=_decode_json(row["step_by_step_fix"]),
        difficulty=row["difficulty"],
        safety_warning=row["safety_warning"],
        sources=_decode_json(row["sources"]),
    )


def _vin_decode_to_response(vin: str) -> VinDecodeResponse | None:
    wmi = _lookup_wmi(vin[:3])
    decoded = decode_vin(
        vin,
        wmi_make=wmi.make if wmi else None,
        wmi_country=wmi.country if wmi else None,
        wmi_manufacturer=wmi.manufacturer if wmi else None,
    )
    if not decoded:
        return None
    return VinDecodeResponse(
        vin=decoded.vin,
        wmi=decoded.wmi,
        vds=decoded.vds,
        vis=decoded.vis,
        model_year=decoded.model_year,
        check_digit=decoded.check_digit,
        check_digit_valid=decoded.check_digit_valid,
        summary=decoded.summary,
        segments=[
            VinSegmentResponse(label=s.label, positions=s.positions, value=s.value, meaning=s.meaning)
            for s in decoded.segments
        ],
    )


def _dtc_decode_to_response(code: str) -> DtcDecodeResponse | None:
    decoded = decode_dtc(code)
    if not decoded:
        return None
    return DtcDecodeResponse(
        code=decoded.code,
        is_standard_format=decoded.is_standard_format,
        summary=decoded.summary,
        segments=[
            DtcSegmentResponse(label=s.label, position=s.position, value=s.value, meaning=s.meaning)
            for s in decoded.segments
        ],
    )


def _diagnostic_to_analyze_response(
    item: DiagnosticCodeResponse,
    all_codes: list[str] | None = None,
    brand_slug: str | None = None,
) -> AnalyzeErrorResponse:
    others = [c for c in (all_codes or []) if c != item.code]
    additional = [_build_detected_code_entry(c, brand_slug=brand_slug) for c in others]
    all_for_deps = [item.code, *others]
    dependencies = _analyze_dependencies(all_for_deps)

    lines = [item.title, item.description, *item.probable_causes]
    dep_notes = [d.explanation for d in dependencies if d.from_code == item.code or d.to_code == item.code]
    if dep_notes:
        lines.append("")
        lines.extend(dep_notes)

    steps = list(item.step_by_step_fix)
    root_causes = [d for d in dependencies if d.relation == "root_cause"]
    if root_causes:
        rc = root_causes[0]
        steps.insert(0, f"Priority: fix {rc.from_code} first — it may be the root cause of {rc.to_code}.")

    return AnalyzeErrorResponse(
        scan_type="dtc",
        detected_code=item.code,
        dtc_decode=_dtc_decode_to_response(item.code),
        additional_codes=additional,
        dependencies=dependencies,
        probable_cause="\n".join(line for line in lines if line),
        step_by_step_fix=steps,
        estimated_difficulty=item.difficulty,
        safety_warning=item.safety_warning,
    )


CODE_SYSTEM_GROUPS: dict[str, str] = {
    "0": "fuel_air",
    "1": "fuel_air",
    "2": "fuel_air",
    "3": "ignition",
    "4": "emissions",
    "5": "speed_idle",
    "6": "computer",
    "7": "transmission",
    "8": "transmission",
    "9": "transmission",
    "A": "hybrid",
    "B": "hybrid",
    "C": "hybrid",
}

BMW_SYSTEM_GROUPS: dict[str, str] = {
    "E12": "transmission",
    "E11": "network",
    "E114": "sensor",
    "E7C": "battery",
}

SYSTEM_LABELS: dict[str, str] = {
    "fuel_air": "Fuel / air metering",
    "ignition": "Ignition / misfire",
    "emissions": "Emissions control",
    "speed_idle": "Speed / idle control",
    "computer": "Computer / output",
    "transmission": "Transmission",
    "hybrid": "Hybrid / EV",
    "network": "Network / communication",
    "sensor": "Sensor / plausibility",
    "battery": "Battery management",
    "body": "Body systems",
    "chassis": "Chassis / ABS / ESP",
    "unknown": "Unknown system",
}


def _code_system_group(code: str) -> str:
    upper = code.upper()
    if upper.startswith("E") and len(upper) == 6:
        for prefix, group in BMW_SYSTEM_GROUPS.items():
            if upper.startswith(prefix):
                return group
        return "unknown"
    if len(upper) == 5 and upper[0] in "PCBU":
        category = upper[0]
        if category == "P":
            return CODE_SYSTEM_GROUPS.get(upper[2], "unknown")
        if category == "C":
            return "chassis"
        if category == "B":
            return "body"
        if category == "U":
            return "network"
    return "unknown"


def _analyze_dependencies(codes: list[str]) -> list[CodeDependency]:
    if len(codes) < 2:
        return []

    deps: list[CodeDependency] = []
    groups: dict[str, list[str]] = {}
    for code in codes:
        grp = _code_system_group(code)
        groups.setdefault(grp, []).append(code)

    network_codes = groups.get("network", [])
    non_network = [c for c in codes if c not in network_codes]

    if network_codes and non_network:
        for nc in network_codes:
            for other in non_network:
                deps.append(CodeDependency(
                    from_code=nc,
                    to_code=other,
                    relation="root_cause",
                    explanation=(
                        f"Communication fault {nc} can prevent modules from reporting correctly, "
                        f"causing {other} as a secondary symptom. Fix {nc} first."
                    ),
                ))

    for grp, grp_codes in groups.items():
        if grp == "network" or len(grp_codes) < 2:
            continue
        label = SYSTEM_LABELS.get(grp, grp)
        for i, a in enumerate(grp_codes):
            for b in grp_codes[i + 1:]:
                deps.append(CodeDependency(
                    from_code=a,
                    to_code=b,
                    relation="same_system",
                    explanation=f"Both {a} and {b} belong to {label} — likely a shared root cause.",
                ))

    battery_codes = groups.get("battery", [])
    sensor_codes = groups.get("sensor", [])
    if battery_codes and sensor_codes:
        for bc in battery_codes:
            for sc in sensor_codes:
                deps.append(CodeDependency(
                    from_code=bc,
                    to_code=sc,
                    relation="root_cause",
                    explanation=(
                        f"Battery/voltage fault {bc} can cause sensor plausibility errors like {sc}. "
                        "Check battery state first."
                    ),
                ))

    return deps


def _build_detected_code_entry(code: str, brand_slug: str | None = None) -> DetectedCodeEntry:
    item = None
    slugs = [brand_slug] if brand_slug else _brand_slugs_for_scan()
    for slug in slugs:
        item = _lookup_code(code, brand_slug=slug)
        if item:
            break
    if not item:
        for brand in _BRANDS:
            item = _lookup_code(code, make=brand.name)
            if item:
                break

    if item:
        return DetectedCodeEntry(
            code=item.code,
            title=item.title,
            description=item.description,
            probable_causes=item.probable_causes,
            symptoms=item.symptoms,
            step_by_step_fix=item.step_by_step_fix,
            difficulty=item.difficulty,
            safety_warning=item.safety_warning,
            in_database=True,
            brand_slug=item.brand_slug or brand_slug,
        )

    return DetectedCodeEntry(
        code=code,
        title=f"Code {code} — not yet in database",
        in_database=False,
        brand_slug=brand_slug,
    )


def _multi_code_response(primary: str, all_codes: list[str], brand_slug: str | None = None) -> AnalyzeErrorResponse:
    primary_entry = _build_detected_code_entry(primary, brand_slug=brand_slug)
    others = [c for c in all_codes if c != primary]
    additional = [_build_detected_code_entry(c, brand_slug=brand_slug) for c in others]
    dependencies = _analyze_dependencies(all_codes)

    if primary_entry.in_database:
        lines = [primary_entry.title or primary, primary_entry.description or ""]
        lines.extend(primary_entry.probable_causes)
    else:
        lines = [f"Fault code {primary} detected on the diagnostic display."]
        if not primary_entry.in_database:
            lines.append("This code is not in the local repair database yet.")

    if others:
        lines.append(f"Additional codes on screen: {', '.join(others)}.")

    dep_notes = []
    for dep in dependencies:
        if dep.from_code == primary or dep.to_code == primary:
            dep_notes.append(dep.explanation)
    if dep_notes:
        lines.append("")
        lines.extend(dep_notes)

    steps = list(primary_entry.step_by_step_fix) if primary_entry.step_by_step_fix else []
    if not steps:
        steps = [
            f"Look up {primary} in manufacturer documentation (ISTA for BMW, XENTRY for Mercedes).",
            "Address the root-cause code first if dependency analysis indicates one.",
            "Clear faults after repair and confirm they do not return after a drive cycle.",
        ]
    if dependencies:
        root_causes = [d for d in dependencies if d.relation == "root_cause"]
        if root_causes:
            rc = root_causes[0]
            steps.insert(0, f"Priority: fix {rc.from_code} first — it may be the root cause of {rc.to_code}.")

    return AnalyzeErrorResponse(
        scan_type="dtc",
        detected_code=primary,
        additional_codes=additional,
        dependencies=dependencies,
        probable_cause="\n".join(line for line in lines if line),
        step_by_step_fix=steps,
        estimated_difficulty=primary_entry.difficulty or "Medium",
        safety_warning=primary_entry.safety_warning or (
            "Multiple fault codes detected. Verify with manufacturer-level diagnostics "
            "before replacing major components."
        ),
    )


def _row_to_vin_vehicle(row: sqlite3.Row) -> VinVehicleResponse:
    year = row["year"]
    return VinVehicleResponse(
        vin=row["vin"],
        make=row["make"],
        model=row["model"],
        engine=row["engine"],
        year=int(year) if year is not None else None,
        body_style=row["body_style"] or "",
        trim=row["trim"] or "",
        notes=row["notes"] or "",
    )


def _vin_to_analyze_response(vehicle: VinVehicleResponse) -> AnalyzeErrorResponse:
    year = vehicle.year
    year_part = str(year) if year else "Unknown year"
    summary = f"{year_part} {vehicle.make} {vehicle.model}".strip()
    if vehicle.trim:
        summary += f" ({vehicle.trim})"
    details = [summary, f"Engine: {vehicle.engine}"]
    if vehicle.body_style:
        details.append(f"Body style: {vehicle.body_style}")
    if vehicle.notes:
        details.append(vehicle.notes)
    steps = [
        "Confirm this VIN on the vehicle data plate (door jamb or windshield base).",
        "Use the VIN for parts lookup, service history, and manufacturer TSB searches.",
        "Scan or enter an OBD-II fault code for this vehicle to get a step-by-step repair guide.",
    ]
    return AnalyzeErrorResponse(
        scan_type="vin",
        detected_code=vehicle.vin,
        detected_vin=vehicle.vin,
        vehicle_make=vehicle.make,
        vehicle_model=vehicle.model,
        vehicle_engine=vehicle.engine,
        vehicle_year=year,
        vin_decode=None,
        probable_cause="\n".join(details),
        step_by_step_fix=steps,
        estimated_difficulty="Easy",
        safety_warning=(
            "VIN identifies the vehicle configuration only. Verify the plate VIN matches before ordering "
            "parts or performing safety-related work."
        ),
    )


def _wmi_to_analyze_response(wmi: WmiPrefixResponse, vin: str) -> AnalyzeErrorResponse:
    normalized = _normalize_vin_chars(vin)[:VIN_LENGTH]
    details = [
        f"VIN {normalized}",
        f"Manufacturer: {wmi.make}",
        f"WMI prefix {wmi.wmi} — {wmi.manufacturer or wmi.make}",
    ]
    if wmi.country:
        details.append(f"Country of origin: {wmi.country}")
    if wmi.notes:
        details.append(wmi.notes)
    details.append(
        "Full vehicle details (model, engine, year) are not in the local database yet for this exact VIN."
    )
    return AnalyzeErrorResponse(
        scan_type="vin",
        detected_code=normalized,
        detected_vin=normalized,
        vehicle_make=wmi.make,
        vehicle_year=None,
        vin_decode=None,
        probable_cause="\n".join(details),
        step_by_step_fix=[
            "Photograph the full 17-character VIN clearly if any characters were misread.",
            "Add this VIN to the local database to unlock model, engine, and year.",
            "Scan an OBD-II fault code screen for repair steps.",
        ],
        estimated_difficulty="Easy",
        safety_warning="WMI decode is approximate. Confirm the full VIN on the vehicle before relying on it for parts.",
    )


def _normalize_code(code: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", code).upper()
    if len(cleaned) >= 2:
        cleaned = cleaned[0] + cleaned[1:].replace("O", "0").replace("I", "1").replace("L", "1")
    return cleaned


def _normalize_vin_chars(value: str) -> str:
    """Normalize OCR text toward valid VIN alphabet (no I, O, Q)."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return cleaned.replace("O", "0").replace("I", "1").replace("Q", "0")


VIN_BLACKLIST_FRAGMENTS = (
    "CHECK",
    "ENGINE",
    "MEMOSC",
    "CANOBD",
    "ERROR",
    "DISPLAY",
    "SCAN",
    "BOSS",
    "PARK",
    "ORIG",
    "1G1NA",
    "CODES",
    "ACTIVE",
    "ENV1",
    "ENV2",
    "KMERR",
    "DTCE",
)

def _known_wmi_prefixes() -> tuple[str, ...]:
    if _BRANDS:
        return all_wmi_prefixes(_BRANDS)
    return (
        "WDD",
        "WDB",
        "WDC",
        "WDF",
        "W1K",
        "W1N",
        "WBA",
        "WBS",
        "WBX",
        "WBY",
        "WBM",
    )


def _brand_slugs_for_scan() -> list[str]:
    if _BRANDS:
        return [b.slug for b in _BRANDS]
    return ["mercedes-benz", "bmw"]


def _infer_brand_slug_from_scan(text: str) -> str | None:
    if _manufacturer_codes_from_text(text):
        return "bmw"
    upper = text.upper()
    for brand in _BRANDS:
        if brand.name.upper() in upper or brand.slug.replace("-", " ").upper() in upper:
            return brand.slug
    return None

# BMW manufacturer-specific hex codes (E + 5 hex digits), common on iDrive fault lists.
BMW_MFG_CODE_PATTERN = re.compile(r"\bE[0-9A-F]{5}\b", re.IGNORECASE)


def _is_valid_vin(vin: str) -> bool:
    return len(vin) == VIN_LENGTH and bool(VIN_CHARS.fullmatch(vin))


def _is_plausible_vin(vin: str) -> bool:
    if not _is_valid_vin(vin):
        return False
    if any(fragment in vin for fragment in VIN_BLACKLIST_FRAGMENTS):
        return False
    if vin[8] not in "0123456789X":
        return False
    if not re.match(r"^[A-HJ-NPR-Z0-9]{3}", vin[:3]):
        return False
    return True


def _compact_alnum(text: str) -> str:
    return _normalize_vin_chars(text)


def _vin_windows_from_compact(compact: str) -> list[str]:
    windows: list[str] = []
    if len(compact) < VIN_LENGTH:
        return windows
    for index in range(len(compact) - VIN_LENGTH + 1):
        window = compact[index : index + VIN_LENGTH]
        if _is_plausible_vin(window) and window not in windows:
            windows.append(window)
    return windows


def _text_has_vin_label(text: str) -> bool:
    return bool(re.search(r"\bVIN\b", text, re.IGNORECASE))


def _vin_from_labeled_text(text: str) -> str | None:
    for match in re.finditer(r"VIN\s*[:#-]?\s*([A-Za-z0-9\s-]{15,24})", text, re.IGNORECASE):
        candidate = _normalize_vin_chars(match.group(1))[:VIN_LENGTH]
        if _is_plausible_vin(candidate):
            return candidate
    return None


def _candidate_vins_from_text(text: str) -> list[str]:
    vins: list[str] = []

    labeled = _vin_from_labeled_text(text)
    if labeled:
        vins.append(labeled)
        return _rank_vin_candidates(vins)

    compact = _compact_alnum(text)

    if _text_has_vin_label(text):
        for match in re.finditer(r"VIN", text, re.IGNORECASE):
            tail = text[match.end() : match.end() + 28]
            fragment = _normalize_vin_chars(tail)
            if len(fragment) >= VIN_LENGTH:
                candidate = fragment[:VIN_LENGTH]
                if _is_plausible_vin(candidate) and candidate not in vins:
                    vins.append(candidate)
            slice_compact = _compact_alnum(tail)[:24]
            for window in _vin_windows_from_compact(slice_compact):
                if window not in vins:
                    vins.append(window)
    else:
        for prefix in _known_wmi_prefixes():
            index = compact.find(prefix)
            if index < 0:
                continue
            slice_compact = compact[max(0, index) : index + VIN_LENGTH + 8]
            for window in _vin_windows_from_compact(slice_compact):
                if window not in vins:
                    vins.append(window)
            if vins:
                break

    return _rank_vin_candidates(vins[:24])


def _manufacturer_codes_from_text(text: str) -> list[str]:
    codes: list[str] = []
    for match in BMW_MFG_CODE_PATTERN.finditer(text):
        code = match.group(0).upper()
        if code not in codes:
            codes.append(code)
    return codes


def _parse_bmw_manufacturer_hex_manual(raw: str) -> str | None:
    """Normalize manual entry to E + 5 hex (BMW iDrive hex codes)."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if len(cleaned) != 6 or cleaned[0] != "E":
        return None
    normalized = cleaned[0] + cleaned[1:].replace("O", "0").replace("I", "1").replace("L", "1")
    if re.fullmatch(r"E[0-9A-F]{5}", normalized):
        return normalized
    return None


def _lookup_code_cross_brand(code: str, make: str | None = None) -> DiagnosticCodeResponse | None:
    """Same search order as OCR scan: brand_slug iteration, optional make hint, then all known makes."""
    for slug in _brand_slugs_for_scan():
        hit = _lookup_code(code, brand_slug=slug)
        if hit:
            return hit
    if make:
        hit = _lookup_code(code, make=make)
        if hit:
            return hit
    for brand in _BRANDS:
        hit = _lookup_code(code, make=brand.name)
        if hit:
            return hit
    return None


def _text_indicates_fault_code_screen(text: str) -> bool:
    upper = text.upper()
    screen_markers = (
        "ERROR CODE",
        "ERROR CODES",
        "FAULT CODE",
        "NO.DTC",
        "NO DTC",
        "ENV1",
        "ENV2",
        "ACTIVE",
    )
    if any(marker in upper for marker in screen_markers):
        return True
    if _candidate_codes_from_text(text):
        return True
    if _manufacturer_codes_from_text(text):
        return True
    return False


def _rank_vin_candidates(candidates: list[str]) -> list[str]:
    unique = list(dict.fromkeys(candidates))[:24]
    if not unique:
        return []

    scored: list[tuple[int, str]] = []
    for vin in unique:
        points = 0
        if _lookup_vin(vin):
            points += 100
        elif _lookup_wmi(vin[:3]):
            points += 40
        if re.match(r"^W[DBDCFKN1]", vin):
            points += 15
        if any(fragment in vin for fragment in VIN_BLACKLIST_FRAGMENTS):
            points -= 80
        if points > 0 or _is_plausible_vin(vin):
            scored.append((points, vin))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [vin for _, vin in scored]


def _candidate_is_inside_vin(candidate: str, text: str, vin_windows: list[str] | None = None) -> bool:
    normalized = _normalize_code(candidate)
    if len(normalized) != 5:
        return False
    windows = vin_windows if vin_windows is not None else _candidate_vins_from_text(text)
    for vin in windows:
        if normalized in vin:
            return True
    compact = _compact_alnum(text)
    for index in range(max(0, len(compact) - VIN_LENGTH + 1)):
        window = compact[index : index + VIN_LENGTH]
        if normalized in window and len(window) >= VIN_LENGTH:
            return True
    return False


def _upsert_vin_vehicle(data: dict[str, str | int | None]) -> VinVehicleResponse:
    with _db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO vin_vehicles (
                vin, make, model, engine, year, body_style, trim, notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(data["vin"]),
                str(data["make"]),
                str(data["model"]),
                str(data["engine"]),
                data.get("year"),
                str(data.get("body_style") or ""),
                str(data.get("trim") or ""),
                str(data.get("notes") or ""),
            ),
        )
        conn.commit()
    vehicle = _lookup_vin(str(data["vin"]))
    if not vehicle:
        raise RuntimeError("Failed to persist VIN vehicle record.")
    return vehicle


def _lookup_vin(vin: str) -> VinVehicleResponse | None:
    normalized = _normalize_vin_chars(vin)[:VIN_LENGTH]
    if not _is_plausible_vin(normalized):
        return None
    with _db_connection() as conn:
        row = conn.execute("SELECT * FROM vin_vehicles WHERE vin = ?", (normalized,)).fetchone()
    return _row_to_vin_vehicle(row) if row else None


def _fetch_and_cache_nhtsa_vin(vin: str) -> VinVehicleResponse | None:
    normalized = _normalize_vin_chars(vin)[:VIN_LENGTH]
    if not _is_plausible_vin(normalized):
        return None
    fields = fetch_nhtsa_decode(normalized)
    if not fields:
        return None
    payload = nhtsa_to_vehicle_fields(normalized, fields)
    return _upsert_vin_vehicle(payload)


def _lookup_wmi(wmi: str) -> WmiPrefixResponse | None:
    prefix = _normalize_vin_chars(wmi)[:3]
    if len(prefix) != 3:
        return None
    with _db_connection() as conn:
        row = conn.execute("SELECT * FROM vin_wmi WHERE wmi = ?", (prefix,)).fetchone()
    if not row:
        return None
    return WmiPrefixResponse(
        wmi=row["wmi"],
        make=row["make"],
        country=row["country"] or "",
        manufacturer=row["manufacturer"] or "",
        notes=row["notes"] or "",
    )


def _detect_vin_from_text(text: str) -> tuple[str | None, list[str]]:
    candidates = _candidate_vins_from_text(text)
    for vin in candidates:
        if _lookup_vin(vin):
            return vin, candidates
    return (candidates[0], candidates) if candidates else (None, candidates)


def _resolve_vin_scan(text: str) -> AnalyzeErrorResponse | None:
    has_vin_label = _text_has_vin_label(text)
    if _text_indicates_fault_code_screen(text) and not has_vin_label:
        return None

    vin, candidates = _detect_vin_from_text(text)
    if not vin and not candidates and not has_vin_label:
        return None

    primary = vin or (candidates[0] if candidates else None)
    if primary:
        normalized_primary = _normalize_vin_chars(primary)[:VIN_LENGTH]
        vehicle = _lookup_vin(normalized_primary)
        if vehicle:
            return _vin_to_analyze_response(vehicle)

        if _is_plausible_vin(normalized_primary):
            vehicle = _fetch_and_cache_nhtsa_vin(normalized_primary)
            if vehicle:
                return _vin_to_analyze_response(vehicle)

            wmi = _lookup_wmi(normalized_primary[:3])
            if wmi:
                return _wmi_to_analyze_response(wmi, normalized_primary)

    if has_vin_label or candidates:
        partial = primary or "unknown"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": (
                    f"Detected VIN context ({partial}), but this VIN is not in the local vehicle database yet."
                ),
                "detected_vins": candidates,
                "ocr_text": text,
                "hints": [
                    "Ensure all 17 VIN characters are visible and in focus.",
                    "VIN is usually on the door jamb sticker or bottom of the windshield.",
                    "Fragments like letters inside a VIN are not OBD fault codes.",
                ],
            },
        )
    return None


def _candidate_codes_from_text(text: str, *, vin_windows: list[str] | None = None) -> list[str]:
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    candidates: list[str] = []
    # DTCs are P/C/B/U + digit 0-3 + 3 hex-like chars. This rejects UI text like "B0SSC".
    pattern = re.compile(r"[PCBU][0-3OIL][0-9A-FIO]{3}")
    for source in (text.upper(), compact):
        for match in pattern.findall(source):
            code = _normalize_code(match)
            if len(code) == 5 and code not in candidates and not _candidate_is_inside_vin(code, text, vin_windows):
                candidates.append(code)
    return candidates


def _fuzzy_candidate_codes_from_text(text: str, *, vin_windows: list[str] | None = None) -> list[str]:
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    candidates: list[str] = []
    # Invalid-looking P/C/B/U words are used only to correct to known DB codes.
    fuzzy_pattern = re.compile(r"[PCBU][0-9A-Z]{4}")
    for source in (text.upper(), compact):
        for match in fuzzy_pattern.findall(source):
            code = _normalize_code(match)
            if len(code) == 5 and code not in candidates and not _candidate_is_inside_vin(code, text, vin_windows):
                candidates.append(code)
    return candidates


def _known_codes() -> list[str]:
    with _db_connection() as conn:
        rows = conn.execute("SELECT DISTINCT code FROM diagnostic_codes ORDER BY code").fetchall()
    return [str(row["code"]) for row in rows]


def _known_code_matches_from_text(text: str) -> list[str]:
    normalized_text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    matches: list[str] = []
    for code in _known_codes():
        variants = {
            code,
            code.replace("0", "O"),
            code.replace("1", "I"),
            code.replace("1", "L"),
        }
        if any(variant in normalized_text for variant in variants) and code not in matches:
            matches.append(code)
    return matches


def _code_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    similar_pairs = {
        frozenset(("0", "8")),
        frozenset(("0", "6")),
        frozenset(("2", "8")),
        frozenset(("2", "Z")),
        frozenset(("0", "D")),
        frozenset(("1", "I")),
        frozenset(("1", "L")),
        frozenset(("5", "S")),
        frozenset(("4", "A")),
    }
    distance = 0
    for left, right in zip(a, b):
        if left == right:
            continue
        distance += 1 if frozenset((left, right)) in similar_pairs else 2
    return distance


def _fuzzy_known_code_matches(candidates: list[str]) -> list[str]:
    known = _known_codes()
    matches: list[str] = []
    for candidate in candidates:
        if candidate in known:
            matches.append(candidate)
            continue
        same_family = [code for code in known if code[0] == candidate[0]]
        ranked = sorted((_code_distance(candidate, code), code) for code in same_family)
        if ranked and ranked[0][0] <= 3 and ranked[0][1] not in matches:
            matches.append(ranked[0][1])
    return matches


def _image_variants(image_bytes: bytes, suffix: str) -> list[tuple[str, bytes]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [("original", image_bytes)]

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return [("original", image_bytes)]

    max_side = max(img.shape[:2])
    variants: list[tuple[str, bytes]] = [("original", image_bytes)]
    encode_ext = ".png" if suffix.lower() == ".png" else ".jpg"

    if max_side > 1600:
        scale = 1600 / max_side
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(encode_ext, resized)
        if ok:
            variants[0] = ("original", encoded.tobytes())
        return variants

    scale = 2 if max_side < 1200 else 1.5
    enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(encode_ext, enlarged)
    if ok:
        variants.append(("enlarged", encoded.tobytes()))
    return variants


def _get_rapid_ocr_engine() -> Any:
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Local OCR is not installed. Run: pip install -r requirements.txt in backend.",
            ) from exc
        _rapid_ocr_engine = RapidOCR()
    return _rapid_ocr_engine


def _run_rapidocr(image_bytes: bytes, suffix: str) -> str:
    engine = _get_rapid_ocr_engine()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
    try:
        result, _ = engine(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result:
        return ""

    parts: list[str] = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1]
            score = item[2] if len(item) >= 3 else 1
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 1
            if isinstance(text, str) and text.strip() and score_value >= 0.25:
                parts.append(text.strip())
    return "\n".join(parts)


def _ocr_has_usable_signal(text: str) -> bool:
    if _vin_from_labeled_text(text):
        return True
    if _text_has_vin_label(text):
        return True
    if _known_code_matches_from_text(text):
        return True
    if _candidate_codes_from_text(text):
        return True
    if _manufacturer_codes_from_text(text):
        return True
    return False


def _extract_text_with_local_ocr(image_bytes: bytes, suffix: str = ".jpg") -> str:
    texts: list[str] = []
    for name, variant_bytes in _image_variants(image_bytes, suffix):
        variant_text = _run_rapidocr(variant_bytes, suffix)
        if variant_text.strip():
            texts.append(f"[{name}]\n{variant_text}")
        if _ocr_has_usable_signal(variant_text):
            break
    return "\n\n".join(texts)


def _detect_known_code_from_text(text: str) -> tuple[str | None, list[str]]:
    if _text_has_vin_label(text):
        return None, []

    vin_windows = _candidate_vins_from_text(text)
    mfg_codes = _manufacturer_codes_from_text(text)
    known_matches = _known_code_matches_from_text(text)
    strict_candidates = _candidate_codes_from_text(text, vin_windows=vin_windows)
    fuzzy_candidates = _fuzzy_candidate_codes_from_text(text, vin_windows=vin_windows)
    candidates = [*known_matches]
    for candidate in mfg_codes:
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in strict_candidates:
        if candidate not in candidates:
            candidates.append(candidate)
    for match in _fuzzy_known_code_matches([*strict_candidates, *fuzzy_candidates]):
        if match not in candidates:
            candidates.insert(0, match)
    scan_slugs = _brand_slugs_for_scan()
    inferred = _infer_brand_slug_from_scan(text)
    if inferred:
        scan_slugs = [inferred, *[slug for slug in scan_slugs if slug != inferred]]
    for candidate in candidates:
        for slug in scan_slugs:
            hit = _lookup_code(candidate, brand_slug=slug)
            if hit:
                return candidate, candidates
        for brand in _BRANDS:
            if _lookup_code(candidate, make=brand.name):
                return candidate, candidates
    if mfg_codes:
        return mfg_codes[0], candidates
    return (strict_candidates[0], candidates) if strict_candidates else (None, candidates)


def _init_knowledge_db() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with _db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostic_codes (
                code TEXT NOT NULL,
                make TEXT NOT NULL,
                engine TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                probable_causes TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                step_by_step_fix TEXT NOT NULL,
                difficulty TEXT NOT NULL CHECK (difficulty IN ('Easy', 'Medium', 'Hard')),
                safety_warning TEXT NOT NULL,
                sources TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (code, make, engine)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS code_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                source_document_id INTEGER NOT NULL,
                snippet TEXT NOT NULL,
                FOREIGN KEY (source_document_id) REFERENCES source_documents(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT NOT NULL,
                source_document_id INTEGER NOT NULL,
                snippet TEXT NOT NULL,
                FOREIGN KEY (source_document_id) REFERENCES source_documents(id)
            )
            """
        )
        global _BRANDS
        _BRANDS = seed_brands_into_db(
            conn,
            normalize_code=_normalize_code,
            encode_json=_encode_json,
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_vehicles (
                vin TEXT PRIMARY KEY,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                engine TEXT NOT NULL,
                year INTEGER,
                body_style TEXT NOT NULL DEFAULT '',
                trim TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_wmi (
                wmi TEXT PRIMARY KEY,
                make TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                manufacturer TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _lookup_code(
    code: str,
    make: str | None = None,
    engine: str | None = None,
    brand_slug: str | None = None,
) -> DiagnosticCodeResponse | None:
    normalized = _normalize_code(code)
    clauses = ["code = ?"]
    params: list[str] = [normalized]
    if brand_slug:
        clauses.append("brand_slug = ?")
        params.append(brand_slug)
    elif make:
        clauses.append("LOWER(make) LIKE LOWER(?)")
        params.append(f"%{make}%")
    if engine:
        clauses.append("LOWER(engine) = LOWER(?)")
        params.append(engine)
    sql = (
        "SELECT * FROM diagnostic_codes WHERE "
        + " AND ".join(clauses)
        + " ORDER BY CASE WHEN engine = 'OM651' THEN 0 WHEN engine = 'Generic' THEN 1 ELSE 2 END LIMIT 1"
    )
    with _db_connection() as conn:
        row = conn.execute(sql, params).fetchone()
    return _row_to_diagnostic_code(row) if row else None


def _search_codes(
    query: str,
    make: str | None = None,
    engine: str | None = None,
    brand_slug: str | None = None,
) -> list[DiagnosticCodeResponse]:
    q = query.strip()
    normalized_code = _normalize_code(q)
    like = f"%{q}%"
    clauses = [
        "(code LIKE ? OR LOWER(title) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?) "
        "OR LOWER(probable_causes) LIKE LOWER(?) OR LOWER(symptoms) LIKE LOWER(?))"
    ]
    params: list[str] = [f"%{normalized_code}%", like, like, like, like]
    if brand_slug:
        clauses.append("brand_slug = ?")
        params.append(brand_slug)
    elif make:
        clauses.append("LOWER(make) LIKE LOWER(?)")
        params.append(f"%{make}%")
    if engine:
        clauses.append("LOWER(engine) = LOWER(?)")
        params.append(engine)
    sql = (
        "SELECT * FROM diagnostic_codes WHERE "
        + " AND ".join(clauses)
        + " ORDER BY code, CASE WHEN engine = 'OM651' THEN 0 WHEN engine = 'Generic' THEN 1 ELSE 2 END LIMIT 25"
    )
    with _db_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_diagnostic_code(row) for row in rows]


def _source_mentions_for_code(code: str) -> list[SourceMentionResponse]:
    normalized = _normalize_code(code)
    with _db_connection() as conn:
        rows = conn.execute(
            """
            SELECT cm.code, sd.url, sd.title, cm.snippet
            FROM code_mentions cm
            JOIN source_documents sd ON sd.id = cm.source_document_id
            WHERE cm.code = ?
            ORDER BY sd.fetched_at DESC
            LIMIT 25
            """,
            (normalized,),
        ).fetchall()
    return [
        SourceMentionResponse(code=row["code"], url=row["url"], title=row["title"], snippet=row["snippet"])
        for row in rows
    ]


def _search_knowledge_sources(query: str, limit: int = 20) -> SourceSearchResponse:
    q = query.strip()
    like = f"%{q}%"
    normalized_code = _normalize_code(q)
    normalized_vin = _normalize_vin_chars(q)[:VIN_LENGTH] if len(q) >= 11 else q.upper()

    with _db_connection() as conn:
        code_rows = conn.execute(
            """
            SELECT cm.code, sd.url, sd.title, cm.snippet
            FROM code_mentions cm
            JOIN source_documents sd ON sd.id = cm.source_document_id
            WHERE cm.code LIKE ? OR LOWER(cm.snippet) LIKE LOWER(?) OR LOWER(sd.content) LIKE LOWER(?)
            ORDER BY sd.fetched_at DESC
            LIMIT ?
            """,
            (f"%{normalized_code}%", like, like, limit),
        ).fetchall()
        vin_rows = conn.execute(
            """
            SELECT vm.vin, sd.url, sd.title, vm.snippet
            FROM vin_mentions vm
            JOIN source_documents sd ON sd.id = vm.source_document_id
            WHERE vm.vin LIKE ? OR LOWER(vm.snippet) LIKE LOWER(?) OR LOWER(sd.content) LIKE LOWER(?)
            ORDER BY sd.fetched_at DESC
            LIMIT ?
            """,
            (f"%{normalized_vin}%", like, like, limit),
        ).fetchall()

    return SourceSearchResponse(
        query=q,
        code_mentions=[
            SourceMentionResponse(code=row["code"], url=row["url"], title=row["title"], snippet=row["snippet"])
            for row in code_rows
        ],
        vin_mentions=[
            VinMentionResponse(vin=row["vin"], url=row["url"], title=row["title"], snippet=row["snippet"])
            for row in vin_rows
        ],
    )


def _build_system_prompt() -> str:
    return """You are an expert automotive diagnostician helping mechanics and DIY users.

TASK:
1) Read the diagnostic tool screen from the image (OCR). Extract:
   - OBD-II / manufacturer trouble code(s). Normalize codes: fix common OCR confusions
     (e.g. letter O vs digit 0, I vs 1). Valid generic OBD-II powertrain codes often match P[0-9A-Z]{4}.
   - Vehicle identification visible on the screen (make/model/year if present).
   - The error / freeze-frame description text if visible.

2) If the image is too blurred, cropped, glare-heavy, or no code can be determined, set extraction_confidence to "none"
   or "low", leave detected_code empty, and still populate analysis with best-effort guidance explaining what is missing.

3) KNOWLEDGE PRIORITY — Mercedes-Benz OM651 (4-cylinder diesel) niche:
   When the vehicle appears to be a Mercedes with OM651 (or unclear diesel Mercedes of relevant era), prioritize in
   probable_cause and steps (when consistent with the code):
   - EGR system faults (EGR valve, EGR temperature/pressure sensors, carbon fouling)
   - DPF / exhaust: differential pressure sensor, soot loading, forced regeneration caveats
   - Swirl flap / intake tuning issues where relevant
   - Timing chain stretch / camshaft correlation symptoms when codes or descriptions suggest it
   Still ground everything in the extracted code and visible text; do not invent a code.

4) SAFETY: If the fault may involve brakes, steering, SRS/airbags, fuel leaks, or high-voltage (hybrid), include explicit
   safety_warning text. Otherwise give a general safe-workshop reminder.

5) DIFFICULTY: Choose estimated_difficulty as Easy, Medium, or Hard based on typical DIY access, tools, and risk.

OUTPUT: Return a single JSON object matching the provided JSON schema exactly. No markdown fences."""


def _normalize_media_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";")[0].strip().lower()


def _validate_image(content_type: str | None, size: int) -> None:
    mt = _normalize_media_type(content_type)
    if mt not in ALLOWED_IMAGE_TYPES and mt != "image/jpg":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type {content_type!r}. Use JPEG, PNG, or WebP.",
        )
    if size > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds max size of {settings.max_image_bytes} bytes.",
        )


async def _call_openai_vision(image_bytes: bytes, mime_type: str) -> VisionExtraction:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured.",
        )
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        completion = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this diagnostic screen image. Reply with minified JSON only, matching:\n"
                                '{"detected_code":"","car_model":"","error_description":"",'
                                '"extraction_confidence":"high|medium|low|none","extraction_notes":"",'
                                '"analysis":{"detected_code":"","probable_cause":"",'
                                '"step_by_step_fix":[],"estimated_difficulty":"Easy|Medium|Hard",'
                                '"safety_warning":""}}'
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 — surface upstream errors cleanly
        logger.exception("OpenAI vision call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vision provider error: {exc!s}",
        ) from exc

    raw = completion.choices[0].message.content or "{}"
    return _parse_vision_payload(raw)


async def _call_anthropic_vision(image_bytes: bytes, mime_type: str) -> VisionExtraction:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY is not configured.",
        )
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    try:
        message = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=_build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime_type, "data": b64},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Return ONLY minified JSON matching this shape:\n"
                                '{"detected_code":"","car_model":"","error_description":"",'
                                '"extraction_confidence":"high|medium|low|none","extraction_notes":"",'
                                '"analysis":{"detected_code":"","probable_cause":"",'
                                '"step_by_step_fix":[],"estimated_difficulty":"Easy|Medium|Hard",'
                                '"safety_warning":""}}\n'
                                "No markdown."
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.exception("Anthropic vision call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vision provider error: {exc!s}",
        ) from exc

    parts: list[str] = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    raw = "".join(parts).strip()
    return _parse_vision_payload(raw)


def _parse_vision_payload(raw: str) -> VisionExtraction:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model JSON: %s", raw[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vision model returned invalid JSON.",
        ) from exc
    try:
        return VisionExtraction.model_validate(data)
    except Exception as exc:
        logger.error("Vision payload validation failed: %s", data)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vision model JSON failed validation: {exc!s}",
        ) from exc


def _merge_detected_code(vision: VisionExtraction) -> AnalyzeErrorResponse:
    """Prefer top-level extracted code; fall back to analysis.detected_code."""
    code = (vision.detected_code or vision.analysis.detected_code or "").strip().upper()
    analysis = vision.analysis.model_copy(update={"detected_code": code})
    return analysis


def _ensure_code_present(vision: VisionExtraction, merged: AnalyzeErrorResponse) -> None:
    if merged.detected_code.strip():
        return
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "message": "Could not read a diagnostic trouble code from the image.",
            "hints": [
                "Retake the photo with sharper focus and no glare.",
                "Include the full code line and any description text.",
                "Move closer so characters are legible.",
            ],
            "extraction_confidence": vision.extraction_confidence,
            "extraction_notes": vision.extraction_notes,
        },
    )


app = FastAPI(title="Diagnostic Vision API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    _init_knowledge_db()


def _vision_api_configured() -> bool:
    if settings.llm_provider == LLMProvider.OPENAI:
        return bool(settings.openai_api_key and settings.openai_api_key.strip())
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        return bool(settings.anthropic_api_key and settings.anthropic_api_key.strip())
    return False


@app.get("/brands", response_model=list[BrandResponse])
async def list_brands() -> list[BrandResponse]:
    """Registered vehicle brands and their knowledge libraries."""
    if _BRANDS:
        return [
            BrandResponse(
                slug=brand.slug,
                name=brand.name,
                wmi_prefixes=list(brand.wmi_prefixes),
                manufacturer_code_pattern=brand.manufacturer_code_pattern,
            )
            for brand in _BRANDS
        ]
    with _db_connection() as conn:
        rows = conn.execute(
            "SELECT slug, name, wmi_prefixes, code_patterns FROM brands ORDER BY sort_order, name"
        ).fetchall()
    results: list[BrandResponse] = []
    for row in rows:
        patterns = json.loads(row["code_patterns"] or "{}")
        results.append(
            BrandResponse(
                slug=row["slug"],
                name=row["name"],
                wmi_prefixes=json.loads(row["wmi_prefixes"] or "[]"),
                manufacturer_code_pattern=patterns.get("manufacturer_hex"),
            )
        )
    return results


@app.get("/health")
async def health() -> dict[str, str | bool]:
    configured = _vision_api_configured()
    return {
        "status": "ok",
        "vision_configured": configured,
        "llm_provider": settings.llm_provider.value,
        "lookup_supports_bmw_hex": True,
        "message": (
            "Vision API ready."
            if configured
            else "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in backend/.env and restart uvicorn."
        ),
    }


@app.get("/codes/{code}", response_model=DiagnosticCodeResponse)
async def get_code(
    code: str,
    make: str | None = "Mercedes-Benz",
    brand: str | None = None,
    engine: str | None = None,
) -> DiagnosticCodeResponse:
    result = _lookup_code(
        code,
        make=None if brand else make,
        brand_slug=brand,
        engine=engine,
    )
    if result:
        return result
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No knowledge entry found for code {_normalize_code(code)}.",
    )


@app.get("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    q: str,
    make: str | None = "Mercedes-Benz",
    brand: str | None = None,
    engine: str | None = None,
) -> KnowledgeSearchResponse:
    results = _search_codes(
        q,
        make=None if brand else make,
        brand_slug=brand,
        engine=engine,
    )
    return KnowledgeSearchResponse(query=q, count=len(results), results=results)


@app.get("/codes/{code}/sources", response_model=list[SourceMentionResponse])
async def get_code_sources(code: str) -> list[SourceMentionResponse]:
    return _source_mentions_for_code(code)


def _lookup_manual_impl(raw_input: str, make: str | None = None) -> AnalyzeErrorResponse:
    raw = raw_input.strip()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enter an OBD-II code (e.g. P0420), a BMW hex code (e.g. E12C11), or a 17-character VIN.",
        )

    normalized_vin = _normalize_vin_chars(raw)[:VIN_LENGTH]
    if len(normalized_vin) == VIN_LENGTH and _is_plausible_vin(normalized_vin):
        vehicle = _lookup_vin(normalized_vin)
        if vehicle:
            return _vin_to_analyze_response(vehicle)
        vehicle = _fetch_and_cache_nhtsa_vin(normalized_vin)
        if vehicle:
            return _vin_to_analyze_response(vehicle)
        wmi = _lookup_wmi(normalized_vin[:3])
        if wmi:
            return _wmi_to_analyze_response(wmi, normalized_vin)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid VIN: {raw!r}",
        )

    mfg_hex = _parse_bmw_manufacturer_hex_manual(raw)
    if mfg_hex:
        item = _lookup_code_cross_brand(mfg_hex, make=make)
        brand_slug = "bmw"
        if item:
            slug = item.brand_slug or slug_for_make(item.make, _BRANDS) or brand_slug
            return _diagnostic_to_analyze_response(item, all_codes=[mfg_hex], brand_slug=slug)
        return _multi_code_response(mfg_hex, [mfg_hex], brand_slug=brand_slug)

    code = _normalize_code(raw)
    if len(code) != 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Enter a 5-character OBD code (e.g. P0420), a BMW hex code (e.g. E12C11), "
                "or a full 17-character VIN."
            ),
        )

    item = _lookup_code_cross_brand(code, make=make)
    if item:
        slug = item.brand_slug or slug_for_make(item.make, _BRANDS)
        return _diagnostic_to_analyze_response(item, all_codes=[code], brand_slug=slug)

    dtc_decoded = _dtc_decode_to_response(code)
    if dtc_decoded:
        return AnalyzeErrorResponse(
            scan_type="dtc",
            detected_code=code,
            dtc_decode=dtc_decoded,
            probable_cause="\n".join(
                [
                    f"{code} is not in the repair database yet.",
                    dtc_decoded.summary,
                ]
            ),
            step_by_step_fix=[
                "Use the code structure above to verify the scanner read the correct fault.",
                "Search manufacturer TSBs and forums for this code on your vehicle.",
                f"Add {code} to the local database to unlock step-by-step repair instructions.",
            ],
            estimated_difficulty="Medium",
            safety_warning="Follow workshop safety practices when testing or repairing this system.",
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Could not interpret {raw!r} as an OBD-II code, BMW hex code, or VIN.",
    )


@app.get("/lookup", response_model=AnalyzeErrorResponse)
async def lookup_manual_get(q: str, make: str | None = None) -> AnalyzeErrorResponse:
    """
    Look up a fault code or VIN entered manually.
    Accepts 5-character OBD-II (P/C/B/U), BMW 6-character hex codes (E + 5 hex), or 17-character VIN.
    When make is omitted, all brand libraries are searched (same behaviour as OCR scan).

    Prefer `POST /lookup` with JSON `{"q": "E12C11"}` from the Expo app — avoids CDN/proxy stripping query params on some setups.
    """
    return _lookup_manual_impl(q, make)


@app.post("/lookup", response_model=AnalyzeErrorResponse)
async def lookup_manual_post(body: LookupQueryBody, make: str | None = None) -> AnalyzeErrorResponse:
    """Same as GET /lookup; JSON body avoids caching issues."""
    return _lookup_manual_impl(body.q, make)


@app.get("/vins/{vin}/decode", response_model=VinDecodeResponse)
async def decode_vin_endpoint(vin: str) -> VinDecodeResponse:
    normalized = _normalize_vin_chars(vin)[:VIN_LENGTH]
    decoded = _vin_decode_to_response(normalized)
    if not decoded:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid VIN length or characters: {vin!r}",
        )
    return decoded


@app.get("/codes/{code}/decode", response_model=DtcDecodeResponse)
async def decode_code_endpoint(code: str) -> DtcDecodeResponse:
    decoded = _dtc_decode_to_response(_normalize_code(code))
    if not decoded:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Not a valid 5-character OBD-II code: {code!r}",
        )
    return decoded


@app.get("/vins/{vin}", response_model=VinVehicleResponse)
async def get_vin(vin: str) -> VinVehicleResponse:
    normalized = _normalize_vin_chars(vin)[:VIN_LENGTH]
    item = _lookup_vin(normalized)
    if not item and _is_plausible_vin(normalized):
        item = _fetch_and_cache_nhtsa_vin(normalized)
    if not item:
        wmi = _lookup_wmi(normalized)
        if wmi and _is_plausible_vin(normalized):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": f"VIN {normalized} is not in the database, but WMI {wmi.wmi} maps to {wmi.make}.",
                    "wmi": wmi.model_dump(),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No vehicle record found for VIN {normalized}.",
        )
    return item


@app.get("/knowledge/sources/search", response_model=SourceSearchResponse)
async def search_ingested_sources(q: str) -> SourceSearchResponse:
    if not q.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query parameter q is required.")
    return _search_knowledge_sources(q)


@app.post("/scan-error-local", response_model=AnalyzeErrorResponse)
async def scan_error_local(image: UploadFile = File(..., description="Diagnostic screen photo")) -> AnalyzeErrorResponse:
    """
    OCR image locally, detect an OBD code, then return the matching local knowledge-base guide.
    This endpoint does not call OpenAI/Anthropic.
    """
    content = await image.read()
    _validate_image(image.content_type, len(content))
    mime = _normalize_media_type(image.content_type)
    suffix = ".png" if mime == "image/png" else ".webp" if mime == "image/webp" else ".jpg"
    text = _extract_text_with_local_ocr(content, suffix=suffix)

    vin_result = _resolve_vin_scan(text)
    if vin_result is not None:
        return vin_result

    code, candidates = _detect_known_code_from_text(text)

    if not code:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "OCR did not find an OBD-style code in the image.",
                "ocr_text": text,
                "hints": [
                    "Crop closer to the scanner display where the code is visible.",
                    "Avoid glare and blur.",
                    "For small handheld scanner screens, fill the frame with the green display.",
                ],
            },
        )

    inferred = _infer_brand_slug_from_scan(text)
    all_detected = candidates if candidates else [code]
    if code not in all_detected:
        all_detected = [code, *all_detected]

    item = None
    lookup_slugs = [inferred] if inferred else []
    lookup_slugs.extend(slug for slug in _brand_slugs_for_scan() if slug not in lookup_slugs)
    for slug in lookup_slugs:
        item = _lookup_code(code, brand_slug=slug)
        if item:
            break
    if not item:
        for brand in _BRANDS:
            item = _lookup_code(code, make=brand.name)
            if item:
                break

    if item:
        return _diagnostic_to_analyze_response(item, all_codes=all_detected, brand_slug=inferred)

    mfg_codes = _manufacturer_codes_from_text(text)
    if code in mfg_codes or code.startswith("E"):
        return _multi_code_response(code, mfg_codes or all_detected, brand_slug=inferred)

    if len(all_detected) > 1:
        return _multi_code_response(code, all_detected, brand_slug=inferred)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "message": f"OCR found {code}, but the code is not in the local DB yet.",
            "detected_candidates": candidates,
            "ocr_text": text,
            "dtc_decode": (
                (decoded := _dtc_decode_to_response(code)).model_dump() if decoded else None
            ),
        },
    )


@app.post("/scan-error-local/debug", response_model=LocalScanDebugResponse)
async def scan_error_local_debug(image: UploadFile = File(...)) -> LocalScanDebugResponse:
    content = await image.read()
    _validate_image(image.content_type, len(content))
    mime = _normalize_media_type(image.content_type)
    suffix = ".png" if mime == "image/png" else ".webp" if mime == "image/webp" else ".jpg"
    text = _extract_text_with_local_ocr(content, suffix=suffix)
    vin_windows = _candidate_vins_from_text(text)
    return LocalScanDebugResponse(
        text=text,
        detected_candidates=_candidate_codes_from_text(text, vin_windows=vin_windows),
        detected_vins=vin_windows,
    )


@app.post("/analyze-error", response_model=AnalyzeErrorResponse)
async def analyze_error(image: UploadFile = File(..., description="Diagnostic screen photo")) -> AnalyzeErrorResponse:
    """
    Accept a diagnostic tool screen image and return a structured technical fix guide.
    """
    content = await image.read()
    _validate_image(image.content_type, len(content))

    mime = _normalize_media_type(image.content_type) or "image/jpeg"
    if mime == "image/jpg":
        mime = "image/jpeg"

    if settings.llm_provider == LLMProvider.OPENAI:
        vision = await _call_openai_vision(content, mime)
    elif settings.llm_provider == LLMProvider.ANTHROPIC:
        vision = await _call_anthropic_vision(content, mime)
    else:
        raise HTTPException(status_code=500, detail="Unsupported LLM_PROVIDER.")

    merged = _merge_detected_code(vision)
    _ensure_code_present(vision, merged)
    return merged
