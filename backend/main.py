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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_DB_PATH = KNOWLEDGE_DIR / "diagnostics.db"
SEED_CODES_PATH = KNOWLEDGE_DIR / "seed_codes.json"
SEED_CODES_GLOB = "seed_*.json"


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


def _parse_cors_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class AnalyzeErrorResponse(BaseModel):
    """Structured response returned to API clients."""

    detected_code: str = Field(description="Primary OBD-II code, normalized (e.g. P0401).")
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


class DiagnosticCodeResponse(BaseModel):
    code: str
    make: str
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


class SourceMentionResponse(BaseModel):
    code: str
    url: str
    title: str
    snippet: str


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
    return DiagnosticCodeResponse(
        code=row["code"],
        make=row["make"],
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


def _diagnostic_to_analyze_response(item: DiagnosticCodeResponse) -> AnalyzeErrorResponse:
    return AnalyzeErrorResponse(
        detected_code=item.code,
        probable_cause="\n".join([item.title, item.description, *item.probable_causes]),
        step_by_step_fix=item.step_by_step_fix,
        estimated_difficulty=item.difficulty,
        safety_warning=item.safety_warning,
    )


def _normalize_code(code: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", code).upper()
    if len(cleaned) >= 2:
        cleaned = cleaned[0] + cleaned[1:].replace("O", "0").replace("I", "1").replace("L", "1")
    return cleaned


def _candidate_codes_from_text(text: str) -> list[str]:
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    candidates: list[str] = []
    # DTCs are P/C/B/U + digit 0-3 + 3 hex-like chars. This rejects UI text like "B0SSC".
    pattern = re.compile(r"[PCBU][0-3OIL][0-9A-FIO]{3}")
    for source in (text.upper(), compact):
        for match in pattern.findall(source):
            code = _normalize_code(match)
            if len(code) == 5 and code not in candidates:
                candidates.append(code)
    return candidates


def _extract_text_with_local_ocr(image_bytes: bytes, suffix: str = ".jpg") -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local OCR is not installed. Run: pip install -r requirements.txt in backend.",
        ) from exc

    engine = RapidOCR()
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
            if isinstance(text, str) and text.strip() and score_value >= 0.35:
                parts.append(text.strip())
    return "\n".join(parts)


def _detect_known_code_from_text(text: str) -> tuple[str | None, list[str]]:
    candidates = _candidate_codes_from_text(text)
    for candidate in candidates:
        if _lookup_code(candidate, make="Mercedes-Benz"):
            return candidate, candidates
    return (candidates[0], candidates) if candidates else (None, candidates)


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
        seed_paths = sorted(KNOWLEDGE_DIR.glob(SEED_CODES_GLOB))
        if not seed_paths:
            logger.warning("Knowledge seed files missing: %s", KNOWLEDGE_DIR / SEED_CODES_GLOB)
            return
        for seed_path in seed_paths:
            seed_items = json.loads(seed_path.read_text(encoding="utf-8"))
            for item in seed_items:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO diagnostic_codes (
                        code, make, engine, title, description, probable_causes, symptoms,
                        step_by_step_fix, difficulty, safety_warning, sources, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        _normalize_code(item["code"]),
                        item["make"],
                        item["engine"],
                        item["title"],
                        item["description"],
                        _encode_json(item["probable_causes"]),
                        _encode_json(item["symptoms"]),
                        _encode_json(item["step_by_step_fix"]),
                        item["difficulty"],
                        item["safety_warning"],
                        _encode_json(item["sources"]),
                    ),
                )
        conn.commit()


def _lookup_code(code: str, make: str | None = None, engine: str | None = None) -> DiagnosticCodeResponse | None:
    normalized = _normalize_code(code)
    clauses = ["code = ?"]
    params: list[str] = [normalized]
    if make:
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


def _search_codes(query: str, make: str | None = None, engine: str | None = None) -> list[DiagnosticCodeResponse]:
    q = query.strip()
    normalized_code = _normalize_code(q)
    like = f"%{q}%"
    clauses = [
        "(code LIKE ? OR LOWER(title) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?) "
        "OR LOWER(probable_causes) LIKE LOWER(?) OR LOWER(symptoms) LIKE LOWER(?))"
    ]
    params: list[str] = [f"%{normalized_code}%", like, like, like, like]
    if make:
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


@app.get("/health")
async def health() -> dict[str, str | bool]:
    configured = _vision_api_configured()
    return {
        "status": "ok",
        "vision_configured": configured,
        "llm_provider": settings.llm_provider.value,
        "message": (
            "Vision API ready."
            if configured
            else "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in backend/.env and restart uvicorn."
        ),
    }


@app.get("/codes/{code}", response_model=DiagnosticCodeResponse)
async def get_code(code: str, make: str | None = "Mercedes-Benz", engine: str | None = None) -> DiagnosticCodeResponse:
    result = _lookup_code(code, make=make, engine=engine)
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
    engine: str | None = None,
) -> KnowledgeSearchResponse:
    results = _search_codes(q, make=make, engine=engine)
    return KnowledgeSearchResponse(query=q, count=len(results), results=results)


@app.get("/codes/{code}/sources", response_model=list[SourceMentionResponse])
async def get_code_sources(code: str) -> list[SourceMentionResponse]:
    return _source_mentions_for_code(code)


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

    item = _lookup_code(code, make="Mercedes-Benz")
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"OCR found {code}, but the code is not in the local DB yet.",
                "detected_candidates": candidates,
                "ocr_text": text,
            },
        )
    return _diagnostic_to_analyze_response(item)


@app.post("/scan-error-local/debug", response_model=LocalScanDebugResponse)
async def scan_error_local_debug(image: UploadFile = File(...)) -> LocalScanDebugResponse:
    content = await image.read()
    _validate_image(image.content_type, len(content))
    mime = _normalize_media_type(image.content_type)
    suffix = ".png" if mime == "image/png" else ".webp" if mime == "image/webp" else ".jpg"
    text = _extract_text_with_local_ocr(content, suffix=suffix)
    return LocalScanDebugResponse(text=text, detected_candidates=_candidate_codes_from_text(text))


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
