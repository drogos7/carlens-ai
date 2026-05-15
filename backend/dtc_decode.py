"""
OBD-II Diagnostic Trouble Code (DTC) structure decoder — 5-character codes.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PREFIX = {
    "P": "Powertrain (engine & transmission)",
    "B": "Body (climate, airbags, lighting, etc.)",
    "C": "Chassis (ABS, steering, suspension)",
    "U": "Network / communication between modules",
}

CODE_TYPE = {
    "0": "Generic (SAE/ISO standardized)",
    "1": "Manufacturer-specific",
}

SUBSYSTEM_DIGIT_3 = {
    "1": "Fuel & air metering",
    "2": "Fuel & air metering (injector circuit)",
    "3": "Ignition system / misfire",
    "4": "Auxiliary emissions controls",
    "5": "Vehicle speed & idle control",
    "6": "Computer / output circuit",
    "7": "Transmission",
    "8": "Transmission",
    "9": "Transmission (manufacturer-specific)",
}


@dataclass
class DtcSegment:
    label: str
    position: str
    value: str
    meaning: str


@dataclass
class DtcDecodeResult:
    code: str
    is_standard_format: bool
    segments: list[DtcSegment]
    summary: str


def decode_dtc(code: str) -> DtcDecodeResult | None:
    cleaned = "".join(ch for ch in code.upper() if ch.isalnum())
    if len(cleaned) != 5:
        return None

    prefix = cleaned[0]
    code_type = cleaned[1]
    subsystem = cleaned[2]
    fault_id = cleaned[3:5]

    if prefix not in SYSTEM_PREFIX:
        return None

    is_standard = code_type in CODE_TYPE
    segments = [
        DtcSegment("System", "1", prefix, SYSTEM_PREFIX[prefix]),
        DtcSegment(
            "Code type",
            "2",
            code_type,
            CODE_TYPE.get(code_type, "Non-standard digit — may be manufacturer-specific or invalid"),
        ),
        DtcSegment(
            "Subsystem",
            "3",
            subsystem,
            SUBSYSTEM_DIGIT_3.get(subsystem, "Manufacturer-specific subsystem"),
        ),
        DtcSegment(
            "Fault description",
            "4–5",
            fault_id,
            f"Specific fault identifier {fault_id} (see repair database for exact definition).",
        ),
    ]

    summary = f"{prefix}{code_type}{subsystem}{fault_id}: {SYSTEM_PREFIX[prefix]}"
    if code_type == "0" and subsystem == "3":
        summary += " — ignition / misfire family"
    if not is_standard:
        summary += " (non-standard type digit — verify code on scanner)"

    return DtcDecodeResult(
        code=cleaned,
        is_standard_format=is_standard,
        segments=segments,
        summary=summary,
    )
