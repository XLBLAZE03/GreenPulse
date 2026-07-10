import os
import io
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

DEFAULT_EQUIPMENT_FILE = os.path.join(
    DATA_DIR,
    "INEOS_Daily_CO2_Dataset_2010_2024_updated.xlsx"
)

ACTIVE_DF = None
ACTIVE_FILE_NAME = "Default INEOS dataset"


EQUIPMENT_MAP = {
    "Boiler": {
        "emission_col": "boiler",
        "energy_col": "boiler_energy KWh",
        "section": "Utilities / Steam Generation"
    },
    "Furnace": {
        "emission_col": "furance",
        "energy_col": "furnace_energy KWh",
        "section": "Thermal Processing"
    },
    "Reformer": {
        "emission_col": "reformer",
        "energy_col": "reforemer_energy KWh",
        "section": "Process Conversion"
    },
    "Other Equipment": {
        "emission_col": "other_Equpiments",
        "energy_col": "other_equipment_KWh_KWh",
        "section": "Auxiliary Equipment"
    }
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Handle common spelling/spacing issues
    rename_map = {
        "boiler ": "boiler",
        "boiler": "boiler",
        "furnace": "furance",
        "reformer_energy KWh": "reforemer_energy KWh",
    }

    df = df.rename(columns=rename_map)
    return df


def _read_excel(file_obj_or_path) -> pd.DataFrame:
    df = pd.read_excel(file_obj_or_path, sheet_name=0)
    df = _clean_columns(df)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "CO2_Emissions_tCO2e",
        "Total_GHG_tCO2e",
        "boiler",
        "furance",
        "reformer",
        "other_Equpiments",
        "boiler_energy KWh",
        "reforemer_energy KWh",
        "furnace_energy KWh",
        "other_equipment_KWh_KWh",
        "Operating_Hours",
        "Production_Index",
        "Steam_tonnes",
        "Natural_Gas_GJ",
        "KWh_MWh",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    required = []
    for eq, cfg in EQUIPMENT_MAP.items():
        required.extend([cfg["emission_col"], cfg["energy_col"]])

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required equipment columns: {missing}")

    return df


def load_default_dataset():
    global ACTIVE_DF, ACTIVE_FILE_NAME

    if ACTIVE_DF is not None:
        return ACTIVE_DF

    if not os.path.exists(DEFAULT_EQUIPMENT_FILE):
        raise FileNotFoundError(
            "Default INEOS equipment file not found. "
            "Place it at data/INEOS_Daily_CO2_Dataset_2010_2024_updated.xlsx"
        )

    ACTIVE_DF = _read_excel(DEFAULT_EQUIPMENT_FILE)
    ACTIVE_FILE_NAME = "Default INEOS dataset"
    return ACTIVE_DF


def set_uploaded_dataset(file_storage):
    global ACTIVE_DF, ACTIVE_FILE_NAME

    ACTIVE_DF = _read_excel(file_storage)
    ACTIVE_FILE_NAME = file_storage.filename or "Uploaded INEOS dataset"

    return {
        "status": "success",
        "file_name": ACTIVE_FILE_NAME,
        "records": int(len(ACTIVE_DF)),
        "columns": list(ACTIVE_DF.columns)
    }


def get_active_dataset():
    return load_default_dataset()


def equipment_summary():
    df = get_active_dataset()

    rows = []
    total_equipment_emissions = 0
    total_equipment_energy = 0

    for equipment, cfg in EQUIPMENT_MAP.items():
        emission_col = cfg["emission_col"]
        energy_col = cfg["energy_col"]

        total_emission = float(df[emission_col].sum())
        total_energy = float(df[energy_col].sum())
        latest_emission = float(df[emission_col].iloc[-1])
        latest_energy = float(df[energy_col].iloc[-1])

        total_equipment_emissions += total_emission
        total_equipment_energy += total_energy

        intensity = total_emission / total_energy if total_energy else 0

        rows.append({
            "equipment": equipment,
            "section": cfg["section"],
            "total_emissions_tCO2e": round(total_emission, 2),
            "latest_emissions_tCO2e": round(latest_emission, 2),
            "total_energy_kWh": round(total_energy, 2),
            "latest_energy_kWh": round(latest_energy, 2),
            "emission_intensity_tCO2e_per_kWh": round(intensity, 6),
        })

    for row in rows:
        row["emission_share_pct"] = round(
            row["total_emissions_tCO2e"] / total_equipment_emissions * 100,
            2
        ) if total_equipment_emissions else 0

    root_cause = max(rows, key=lambda x: x["total_emissions_tCO2e"])

    return {
        "active_file": ACTIVE_FILE_NAME,
        "records": int(len(df)),
        "date_range": _date_range(df),
        "total_equipment_emissions_tCO2e": round(total_equipment_emissions, 2),
        "total_equipment_energy_kWh": round(total_equipment_energy, 2),
        "equipment": rows,
        "root_cause": {
            "equipment": root_cause["equipment"],
            "section": root_cause["section"],
            "reason": (
                f"{root_cause['equipment']} is the largest contributor with "
                f"{root_cause['total_emissions_tCO2e']} tCO₂e, contributing "
                f"{root_cause['emission_share_pct']}% of equipment-attributed emissions."
            )
        }
    }


def section_summary():
    summary = equipment_summary()
    section_rows = {}

    for row in summary["equipment"]:
        section = row["section"]

        if section not in section_rows:
            section_rows[section] = {
                "section": section,
                "total_emissions_tCO2e": 0,
                "total_energy_kWh": 0,
                "equipment": []
            }

        section_rows[section]["total_emissions_tCO2e"] += row["total_emissions_tCO2e"]
        section_rows[section]["total_energy_kWh"] += row["total_energy_kWh"]
        section_rows[section]["equipment"].append(row["equipment"])

    final = []

    for section, data in section_rows.items():
        final.append({
            "section": section,
            "equipment": ", ".join(data["equipment"]),
            "total_emissions_tCO2e": round(data["total_emissions_tCO2e"], 2),
            "total_energy_kWh": round(data["total_energy_kWh"], 2),
        })

    return {
        "active_file": ACTIVE_FILE_NAME,
        "sections": final
    }


def _date_range(df):
    if "Date" not in df.columns:
        return "Unknown"

    valid = df["Date"].dropna()

    if valid.empty:
        return "Unknown"

    return f"{valid.min().date()} to {valid.max().date()}"


def answer_equipment_prompt(message: str):
    msg = message.lower()
    summary = equipment_summary()
    sections = section_summary()

    if "report" in msg or "pdf" in msg:
        return {
            "handled": True,
            "reply": "I can generate the equipment-wise emissions and root-cause report. Click the Download PDF Report button.",
            "intent": "report_request"
        }

    if "root" in msg or "cause" in msg or "highest" in msg or "maximum" in msg or "most" in msg:
        rc = summary["root_cause"]
        return {
            "handled": True,
            "reply": f"Root cause: {rc['reason']}",
            "intent": "root_cause"
        }

    if "section" in msg:
        lines = ["Section-wise emissions summary:"]
        for s in sections["sections"]:
            lines.append(
                f"- {s['section']}: {s['total_emissions_tCO2e']} tCO₂e, "
                f"{s['total_energy_kWh']} kWh, equipment: {s['equipment']}"
            )

        return {
            "handled": True,
            "reply": "\n".join(lines),
            "intent": "section_summary"
        }

    if "energy" in msg:
        lines = ["Equipment-wise energy consumption:"]
        for e in summary["equipment"]:
            lines.append(
                f"- {e['equipment']}: {e['total_energy_kWh']} kWh "
                f"({e['section']})"
            )

        return {
            "handled": True,
            "reply": "\n".join(lines),
            "intent": "energy_summary"
        }

    if "equipment" in msg or "boiler" in msg or "furnace" in msg or "reformer" in msg or "emission" in msg:
        lines = [
            f"Equipment-wise CO₂ generation from {summary['active_file']}:"
        ]

        for e in summary["equipment"]:
            lines.append(
                f"- {e['equipment']}: {e['total_emissions_tCO2e']} tCO₂e "
                f"({e['emission_share_pct']}%), energy {e['total_energy_kWh']} kWh"
            )

        return {
            "handled": True,
            "reply": "\n".join(lines),
            "intent": "equipment_summary"
        }

    return {
        "handled": False,
        "reply": "",
        "intent": "not_equipment_query"
    }


def generate_pdf_report():
    summary = equipment_summary()
    sections = section_summary()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("INEOS Equipment CO2 Emissions & Root Cause Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Active file: {summary['active_file']}", styles["Normal"]))
    story.append(Paragraph(f"Date range: {summary['date_range']}", styles["Normal"]))
    story.append(Paragraph(f"Records analysed: {summary['records']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        f"Total equipment-attributed emissions are "
        f"{summary['total_equipment_emissions_tCO2e']} tCO2e. "
        f"Total equipment energy consumption is "
        f"{summary['total_equipment_energy_kWh']} kWh.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Root Cause Analysis", styles["Heading2"]))
    story.append(Paragraph(summary["root_cause"]["reason"], styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Equipment-wise CO2 and Energy Summary", styles["Heading2"]))

    equipment_table = [["Equipment", "Section", "CO2 tCO2e", "Energy kWh", "Share %"]]

    for e in summary["equipment"]:
        equipment_table.append([
            e["equipment"],
            e["section"],
            e["total_emissions_tCO2e"],
            e["total_energy_kWh"],
            e["emission_share_pct"]
        ])

    table = Table(equipment_table, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f5132")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Section-wise Emissions", styles["Heading2"]))

    section_table = [["Section", "Equipment", "CO2 tCO2e", "Energy kWh"]]

    for s in sections["sections"]:
        section_table.append([
            s["section"],
            s["equipment"],
            s["total_emissions_tCO2e"],
            s["total_energy_kWh"]
        ])

    s_table = Table(section_table, repeatRows=1)
    s_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4D2F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ]))

    story.append(s_table)

    doc.build(story)

    buffer.seek(0)
    return buffer