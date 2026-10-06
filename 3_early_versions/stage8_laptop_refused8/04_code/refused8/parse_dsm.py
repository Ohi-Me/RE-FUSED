"""Parser for Grid-India DSM rate files (NLDC), December 2018 – present.

Two regimes are published:
  R1 "ChargesForDeviationForDDMMYY.csv" (CERC DSM Regulations 2014 with the Fourth Amendment 2018, until the DSM
     Regulations 2022 took effect): a frequency-band schedule of charges for deviation by bid area and the daily
     "Average ACP/UMCP (paisa/kWh)" discovered at the power exchanges, by bid area.
  R2 "DSM_DD-MM-YYYY.xlsx/.csv" (DSM Regulations 2022 and 2024): 96 blocks of the normal rate of charges for deviation by
     bid area, followed by block-wise weighted-average DAM ACP, RTM ACP (and, in later files, HP-DAM and derived
     sections) and the ancillary net cost.
Output rows (long): date, block (1–96, or 0 for daily values), bid_area (A1…W3, IR, UMCP/MCP), series, value in ₹/MWh
(paise/kWh × 10). Series: normal_rate, dam_acp (DAM incl. GDAM, and HP-DAM where stated), dam_gdam_acp (DAM+GDAM
repeated without HP-DAM in 2024+ files), rtm_acp, hpdam (HP-DAM seller rate), sum_third (1/3 DAM + 1/3 RTM + 1/3
ancillary), ancillary_net_cost, reference_rate_daily (2024+ daily reference charge rate), acp_daily (R1 daily average
ACP/UMCP), charge_at_50_00 (R1 charge for the 50.00–49.99 Hz band, the rate around nominal frequency).
Note: under the 2024 Regulations block values of 1000 paise/kWh (₹10,000/MWh) are the price cap.
"""
import datetime as dt
import io
import re

import numpy as np
import pandas as pd

AREAS = ["A1", "A2", "E1", "E2", "N1", "N2", "N3", "S1", "S2", "S3", "W1", "W2", "W3"]


def _date_from_name(name):
    m = re.search(r"ChargesForDeviationFor(\d{2})(\d{2})(\d{2})", name)
    if m:
        return dt.date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", name)
    if m:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _num(x):
    try:
        v = float(str(x).replace(",", "").strip())
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def parse_r1_csv(raw, name):
    text = raw.decode("latin-1")
    date = None
    m = re.search(r"applicable for:\s*(\d{2})-(\d{2})-(\d{4})", text, re.I)
    if m:
        date = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    date = date or _date_from_name(name)
    rows = []
    lines = [l.replace("\t", "") for l in text.splitlines()]
    header = next((i for i, l in enumerate(lines) if l.startswith("Below,Not Below")), None)
    if header is not None:
        cols = [c.strip() for c in lines[header].split(",")]
        for l in lines[header + 1:]:
            parts = [p.strip() for p in l.split(",")]
            if len(parts) >= 16 and parts[0] == "50.00" and parts[1] == "49.99":
                for a, v in zip(cols[2:15], parts[2:15]):
                    rows.append(dict(date=date, block=0, bid_area=a, series="charge_at_50_00", value=_num(v) * 10 if _num(v) is not None else None))
    for l in lines:
        if l.startswith("Average ACP/UMCP (paisa/kWh),"):
            parts = [p.strip() for p in l.split(",")]
            vals = parts[2:16]
            for a, v in zip(AREAS + ["IR"], vals):
                if _num(v) is not None:
                    rows.append(dict(date=date, block=0, bid_area=a, series="acp_daily", value=_num(v) * 10))
            break
    return rows


SECTION_PATTERNS = [
    ("normal_rate", r"^Normal Rate of Charges"),
    ("dam_acp", r"weighted average\s+ACP\s+DAM"),
    ("rtm_acp", r"weighted average\s+ACP\s+RTM"),
    ("hpdam", r"HP-DA|HPDAM|High Price"),
    ("sum_third", r"^Sum of \[1/3"),
    ("ancillary_net_cost", r"^Ancillary net Cost"),
]


def parse_r2_table(df, name):
    """df: header=None frame of an R2 xlsx/csv."""
    cells = df.astype(object).where(pd.notna(df), None).values.tolist()
    date = None
    for r in cells[:2]:
        s = " ".join(str(x) for x in r if x is not None)
        m = re.search(r"Applicable for:?\s*(\d{2})-(\d{2})-(\d{4})", s, re.I)
        if m:
            date = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    date = date or _date_from_name(name)
    rows, series, seen = [], None, set()
    header, ref_header = None, None
    for r in cells:
        first = " ".join(str(x) for x in r[:3] if x is not None).strip()
        if re.search(r"daily weighted average ACP.*Reference Charge Rate", first, re.I):
            series, header, ref_header = "reference_rate_daily", None, "pending"
            continue
        if ref_header == "pending" and r and "A1" in [str(x).strip() for x in r]:
            ref_header = [str(x).strip() if x is not None else "" for x in r]
            continue
        if isinstance(ref_header, list):
            for a, v in zip(ref_header, r):
                a = "IR" if a.startswith("IR") else a
                if (a in AREAS or a == "IR") and _num(v) is not None:
                    rows.append(dict(date=date, block=0, bid_area=a, series="reference_rate_daily", value=_num(v) * 10))
            ref_header, series = None, None
            continue
        hit = next((s for s, rx in SECTION_PATTERNS if re.search(rx, first, re.I)), None)
        if hit:
            if hit == "dam_acp" and "dam_acp" in seen:
                hit = "dam_gdam_acp"  # 2024 files repeat the DAM table without HP-DAM
            series, header = (hit if hit not in seen else None), None
            if hit == "ancillary_net_cost":
                header = ["Block", "Time", "ANC"]
            seen.add(hit)
            continue
        if r and str(r[0]).strip() == "Block":
            header = [str(x).strip() if x is not None else "" for x in r]
            if len(header) > 2 and header[2].startswith("Ancillary"):
                series, header = "ancillary_net_cost", ["Block", "Time", "ANC"]
            continue
        if series and header and r and _num(r[0]) is not None and 1 <= _num(r[0]) <= 96:
            blk = int(_num(r[0]))
            for c in range(2, min(len(header), len(r))):
                a = header[c]
                if a == "ANC":
                    a = "ALL"
                elif a.startswith("IR"):
                    a = "IR"
                elif a.startswith("UMCP") or a.startswith("MCP"):
                    a = "MCP"
                elif a not in AREAS:
                    continue
                v = _num(r[c])
                if v is not None:
                    rows.append(dict(date=date, block=blk, bid_area=a, series=series, value=v * 10))
    return rows


def parse_file(raw, name):
    low = name.lower()
    if "chargesfordeviation" in low and low.endswith(".csv"):
        return "R1", parse_r1_csv(raw, name)
    if low.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(raw), header=None, engine="openpyxl" if low.endswith("x") else None)
        except Exception:
            return "unreadable", []
        return "R2", parse_r2_table(df, name)
    if low.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw), header=None, encoding="latin-1", on_bad_lines="skip",
                             names=list(range(20)))
        except Exception:
            return "unreadable", []
        return "R2", parse_r2_table(df, name)
    return "skip", []
