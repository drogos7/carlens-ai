"""One-off builder: parse Mercedes code list into seed_mb_catalog.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW = """
P1856 – Gear selector repair [SOLVED]
P0138 Mercedes: Heated Oxygen Sensor (HO2S) Circuit High Voltage Bank 1 Sensor 2
P0460 Mercedes: Fuel Level Sensor Circuit
P0150 Mercedes: Heated Oxygen Sensor (HO2S) Circuit Closed Loop (CL) Performance Bank 2 Sensor 1
P0226 Mercedes: APP Sensor 3 Circuit Performance
P0221 Mercedes: APP (Throttle Position) Sensor 2 Circuit Performance
P0217 Mercedes: Engine Overtemp Condition
P0234 Mercedes: Turbocharger Engine Overboost Condition
P0170 Mercedes: Fuel Trim Bank 1
P0151 Mercedes: Heated Oxygen Sensor (HO2S) Circuit Low Voltage Bank 2 Sensor 1
P0121 Mercedes: TP (Throttle position)Sensor Circuit Insufficient Activity
P0483 Mercedes: Cooling Fan Rationality Check Malfunction
P0620 Mercedes: Generator Control Circuit Malfunction
P0014 Mercedes: Camshaft B position timing over advanced or system performance (Bank 1)
P0560 Mercedes: System Voltage
P0229 Mercedes: Throttle Position Sensor 3 Intermittent
P0335 Mercedes: CKP Sensor A Circuit Performance
P0605 Mercedes: Control Module Programming Read Only Memory (ROM)
P0475 Mercedes: Exhaust Pressure Control Valve Malfunction
P0179 Mercedes: Fuel Composition Sensor Circuit High Voltage
P0533 Mercedes: Air Conditioning (A/C) Refrigerant Pressure Sensor Circuit High Voltage
P0191 Mercedes: Fuel Rail Pressure Sensor Circuit Performance
P0015 Mercedes: Camshaft B position timing over retarded (Bank 1)
P0718 Mercedes: Input/Turbine Speed Sensor Circuit Intermittent in automatic transmission
P0190 Mercedes: Fuel Rail Pressure Sensor Circuit
P1335 Mercedes: CKP Circuit
P0200 Mercedes: Injector Control Circuit
P1105 Mercedes: Secondary Vacuum Sensor Circuit
P0711 Mercedes: TFT Sensor Circuit Range/Performance
P1234 Mercedes: Injector Circuit Cylinder 5 Intermittent
P0356 Mercedes: Ignition Coil 6 Control Circuit
P0044 Mercedes: H02S heater control circuit High (Bank 1 sensor 3)
P0702 Mercedes: Transmission Control System Electrical
P1350 Mercedes: Ignition Control System
P0104 Mercedes: Mass Air Flow Circuit Intermittent
P0085 Mercedes: Exhaust valve control solenoid circuit low (Bank 2)
P0238 Mercedes: Turbocharger Boost Sensor Circuit High Voltage
P0721 Mercedes: Output Speed Sensor Range/Performance in automatic transmission
P0740 Mercedes: TCC Enable Solenoid Circuit Electrical
P0056 Mercedes: Heated Oxygen Sensor (HO2S) Heater Circuit Bank 2 Sensor 2
P0068 Mercedes: Throttle Body Airflow Performance (PCM)
P0198 Mercedes: Engine Oil Temperature Sensor High Voltage
P0522 Mercedes: Engine Oil Pressure Sensor/Switch Circuit Low Voltage
P0480 Mercedes: Cooling Fan Relay 1 Control Circuit
P0038 Mercedes: H02S heater control circuit high (Bank 1 sensor 2)
P0100 Mercedes: MAF Sensor Circuit Insufficient Activity
P0062 Mercedes: H02S heater control circuit (Bank 2 sensor 3)
P0101 Mercedes: Mass Air Flow (MAF) Sensor Performance
P0067 Mercedes: Air assisted injector control circuit high
P0745 Mercedes: Pressure Control Solenoid Malfunction
"""


def guess_difficulty(title: str) -> str:
    lower = title.lower()
    if any(x in lower for x in ("transmission", "turbo", "camshaft", "injector control", "gear selector")):
        return "Hard"
    if any(x in lower for x in ("sensor", "circuit", "voltage", "heater", "maf", "ho2s")):
        return "Medium"
    return "Easy"


def guess_steps(code: str, title: str) -> list[str]:
    lower = title.lower()
    steps = [
        f"Confirm {code} with a scan tool; note freeze-frame and readiness monitors.",
        "Inspect related wiring harness, connectors, and grounds for corrosion or damage.",
    ]
    if "oxygen" in lower or "ho2s" in lower or "o2" in lower:
        steps += [
            "Check sensor heater fuse and voltage supply at the sensor connector.",
            "Measure sensor signal and heater resistance against specification.",
            "Replace the affected O2 sensor if out of spec; clear codes and verify closed-loop fuel control.",
        ]
    elif "transmission" in lower or "tcc" in lower or "turbine" in lower or "gear selector" in lower:
        steps += [
            "Check transmission fluid level and condition.",
            "Inspect speed sensor wiring and transmission control module connections.",
            "Consult transmission-specific repair data before internal repairs.",
        ]
    elif "fuel" in lower or "injector" in lower or "rail pressure" in lower:
        steps += [
            "Verify fuel pressure and volume under load.",
            "Inspect fuel pump relay, filter, and injector wiring.",
            "Test or replace faulty fuel delivery or injector components as indicated.",
        ]
    elif "camshaft" in lower or "timing" in lower:
        steps += [
            "Inspect camshaft adjuster solenoids and oil supply to phasers.",
            "Check timing chain stretch indicators and correlation values with a scan tool.",
            "Repair timing components if correlation is out of specification.",
        ]
    elif "turbo" in lower or "boost" in lower:
        steps += [
            "Inspect turbo hoses and intercooler for leaks.",
            "Check boost pressure sensor and wastegate/actuator operation.",
            "Verify charge air system integrity after repairs.",
        ]
    else:
        steps += [
            "Test the suspected component circuit per manufacturer wiring diagrams.",
            "Repair or replace the faulty part; clear codes and road-test.",
        ]
    return steps


def parse_entries() -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    for line in RAW.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(P[0-9A-Z]{4})\s*(?:Mercedes:|:|–|-)\s*(.+)$", line, re.I)
        if not match:
            continue
        code = match.group(1).upper()
        title = match.group(2).strip()
        if code in seen:
            continue
        seen.add(code)
        entries.append(
            {
                "code": code,
                "make": "Mercedes-Benz",
                "engine": "Generic",
                "title": title.rstrip("."),
                "description": f"Mercedes-Benz diagnostic trouble code {code}: {title}",
                "probable_causes": [
                    title,
                    "Wiring, connector, or sensor/actuator fault in the related circuit",
                    "Component wear or contamination common on higher-mileage vehicles",
                ],
                "symptoms": [
                    "Check engine light (MIL) illuminated",
                    "Possible limp mode or reduced performance depending on fault",
                    "Related system performance fault flags in scan data",
                ],
                "step_by_step_fix": guess_steps(code, title),
                "difficulty": guess_difficulty(title),
                "safety_warning": (
                    "Disconnect the battery when working on airbag or SRS-related circuits. "
                    "Use proper jack stands and allow the engine to cool before exhaust or cooling system work."
                ),
                "sources": ["Mercedes-Benz code catalog (imported reference list)"],
            }
        )
    return entries


if __name__ == "__main__":
    out = Path(__file__).parent / "brands" / "mercedes-benz" / "codes" / "catalog.json"
    data = parse_entries()
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(data)} codes to {out}")
