"""
VIN (Vehicle Identification Number) decoder — WMI, VDS, VIS sections.

Based on ISO 3779 / NHTSA standard 17-character VIN layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

VIN_LENGTH = 17

# Model year codes (position 10) — repeats every 30 years from 1980.
VIN_YEAR_CYCLES: dict[str, list[int]] = {
    "A": [1980, 2010],
    "B": [1981, 2011],
    "C": [1982, 2012],
    "D": [1983, 2013],
    "E": [1984, 2014],
    "F": [1985, 2015],
    "G": [1986, 2016],
    "H": [1987, 2017],
    "J": [1988, 2018],
    "K": [1989, 2019],
    "L": [1990, 2020],
    "M": [1991, 2021],
    "N": [1992, 2022],
    "P": [1993, 2023],
    "R": [1994, 2024],
    "S": [1995, 2025],
    "T": [1996, 2026],
    "V": [1997, 2027],
    "W": [1998, 2028],
    "X": [1999, 2029],
    "Y": [2000, 2030],
    "1": [2001, 2031],
    "2": [2002, 2032],
    "3": [2003, 2033],
    "4": [2004, 2034],
    "5": [2005, 2035],
    "6": [2006, 2036],
    "7": [2007, 2037],
    "8": [2008, 2038],
    "9": [2009, 2039],
}

# First digit — country / region of origin (simplified, common codes).
COUNTRY_DIGIT_1: dict[str, str] = {
    "1": "United States",
    "2": "Canada",
    "3": "Mexico",
    "4": "United States",
    "5": "United States",
    "6": "Australia",
    "7": "New Zealand",
    "8": "Argentina",
    "9": "Brazil",
    "J": "Japan",
    "K": "South Korea",
    "L": "China",
    "S": "United Kingdom",
    "T": "Switzerland",
    "V": "France / Spain (varies)",
    "W": "Germany",
    "X": "Russia / Netherlands (varies)",
    "Y": "Sweden / Finland (varies)",
    "Z": "Italy",
}

# Second digit — manufacturer region hints when known.
MANUFACTURER_REGION_DIGIT_2: dict[str, str] = {
    "A": "Audi / VAG region codes",
    "B": "BMW",
    "D": "Mercedes-Benz (Germany)",
    "F": "Ford (USA)",
    "G": "General Motors (USA)",
    "H": "Honda",
    "J": "Japan domestic",
    "N": "Nissan",
    "T": "Toyota",
    "V": "Volkswagen",
    "W": "Volkswagen / VAG (Germany)",
}

VIN_TRANSLITERATION = {
    **{str(i): i for i in range(10)},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}

VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


@dataclass
class VinSegment:
    label: str
    positions: str
    value: str
    meaning: str


@dataclass
class VinDecodeResult:
    vin: str
    wmi: str
    vds: str
    vis: str
    model_year: int | None
    check_digit: str
    check_digit_valid: bool | None
    segments: list[VinSegment]
    summary: str


def normalize_vin(vin: str) -> str:
    cleaned = "".join(ch for ch in vin.upper() if ch.isalnum())
    return cleaned.replace("O", "0").replace("I", "1").replace("Q", "0")[:VIN_LENGTH]


def decode_model_year(pos10: str, reference_year: int | None = None) -> int | None:
    ref = reference_year if reference_year is not None else date.today().year
    cycles = VIN_YEAR_CYCLES.get(pos10.upper())
    if not cycles:
        return None
    eligible = [year for year in cycles if year >= 1981]
    if not eligible:
        return None
    return min(eligible, key=lambda year: abs(year - ref))


def verify_check_digit(vin: str) -> bool | None:
    if len(vin) != VIN_LENGTH:
        return None
    total = 0
    for index, char in enumerate(vin):
        if index == 8:
            continue
        value = VIN_TRANSLITERATION.get(char)
        if value is None:
            return None
        total += value * VIN_WEIGHTS[index]
    remainder = total % 11
    expected = "X" if remainder == 10 else str(remainder)
    return vin[8] == expected


def decode_vin(
    vin: str,
    *,
    reference_year: int | None = None,
    wmi_make: str | None = None,
    wmi_country: str | None = None,
    wmi_manufacturer: str | None = None,
) -> VinDecodeResult | None:
    normalized = normalize_vin(vin)
    if len(normalized) != VIN_LENGTH:
        return None

    wmi = normalized[0:3]
    vds = normalized[3:9]
    vis = normalized[9:17]
    model_year = decode_model_year(normalized[9], reference_year)
    check_valid = verify_check_digit(normalized)

    country = COUNTRY_DIGIT_1.get(normalized[0], "Region code — consult manufacturer")
    region2 = MANUFACTURER_REGION_DIGIT_2.get(normalized[1], "Manufacturer / plant region code")
    division = f"Vehicle type / division code: {normalized[2]}"

    wmi_meaning = f"WMI {wmi}"
    if wmi_make:
        wmi_meaning = f"{wmi_make} ({wmi})"
        if wmi_manufacturer:
            wmi_meaning += f" — {wmi_manufacturer}"
        if wmi_country:
            wmi_meaning += f", {wmi_country}"

    segments: list[VinSegment] = [
        VinSegment(
            label="World Manufacturer Identifier (WMI)",
            positions="1–3",
            value=wmi,
            meaning=wmi_meaning,
        ),
        VinSegment(
            label="Country / final assembly region",
            positions="1",
            value=normalized[0],
            meaning=country,
        ),
        VinSegment(
            label="Manufacturer & region",
            positions="2",
            value=normalized[1],
            meaning=region2,
        ),
        VinSegment(
            label="Vehicle type / division",
            positions="3",
            value=normalized[2],
            meaning=division,
        ),
        VinSegment(
            label="Vehicle Descriptor Section (VDS)",
            positions="4–9",
            value=vds,
            meaning=(
                "Model, body, restraint, transmission, and engine codes "
                f"({normalized[3:8]}); manufacturer-specific."
            ),
        ),
        VinSegment(
            label="Check digit",
            positions="9",
            value=normalized[8],
            meaning=(
                "Valid check digit (ISO 3779)"
                if check_valid is True
                else "Check digit mismatch — verify VIN for typos/OCR errors"
                if check_valid is False
                else "Check digit could not be verified"
            ),
        ),
        VinSegment(
            label="Model year",
            positions="10",
            value=normalized[9],
            meaning=str(model_year) if model_year else "Unknown model year code",
        ),
        VinSegment(
            label="Vehicle Identifier Section (VIS)",
            positions="10–17",
            value=vis,
            meaning="Plant code, serial number, and production sequence (unique to this vehicle).",
        ),
        VinSegment(
            label="Plant / serial",
            positions="11–17",
            value=normalized[10:17],
            meaning="Assembly plant and unique serial number.",
        ),
    ]

    year_text = str(model_year) if model_year else "unknown year"
    summary = f"VIN decodes to {year_text}"
    if wmi_make:
        summary += f", {wmi_make}"
    summary += f" (WMI {wmi})."

    return VinDecodeResult(
        vin=normalized,
        wmi=wmi,
        vds=vds,
        vis=vis,
        model_year=model_year,
        check_digit=normalized[8],
        check_digit_valid=check_valid,
        segments=segments,
        summary=summary,
    )
