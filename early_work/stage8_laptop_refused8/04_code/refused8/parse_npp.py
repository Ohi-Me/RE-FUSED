"""Parser for CEA National Power Portal Daily Generation Report, sub-report 2 (unit-wise generation), 2018 – present.

Hierarchy in the sheet: REGION → (REGION TOTAL) → STATE → (STATE TOTAL) → "SECTOR:" → "TYPE:" → station → "Unit" rows.
Column positions differ between years, so they are located from the header text:
  MONITORED CAP. (MW) | TODAY'S PROGRAM (MU) | TODAY'S ACTUAL (MU) | COAL STOCK IN DAYS | CAP. UNDER OUTAGE (MW)
Outputs:
  stations  date, region, state_raw, sector, type, station, cap_mw, prog_gwh, act_gwh, coal_days, outage_mw, units,
            units_out
  states    date, region, state_raw, cap_mw, prog_gwh, act_gwh, outage_mw   (the report's own STATE TOTAL rows)
"""
import datetime as dt
import re

import numpy as np
import pandas as pd

REGION_NAMES = {"NORTHERN", "WESTERN", "SOUTHERN", "EASTERN", "NORTH EASTERN", "NORTH-EASTERN", "ISLANDS", "OTHERS"}
TYPE_NAMES = {"THERMAL", "THER (GT)", "THER (DG)", "NUCLEAR", "HYDRO"}


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def _texts(row):
    return [str(v).strip() for v in row if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip()]


def parse_dgr2(path):
    x = pd.read_excel(path, header=None)
    cells = x.astype(object).where(pd.notna(x), None).values.tolist()
    date = None
    for r in cells[:3]:
        s = " ".join(_texts(r))
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s) or re.search(r"(\d{2})-([A-Za-z]{3})-(\d{4})", s)
        if m:
            try:
                date = dt.datetime.strptime("/".join(m.groups()), "%d/%m/%Y" if "/" in m.group(0) else "%d/%b/%Y").date()
            except ValueError:
                pass
            break
    if date is None:  # Jul–Dec 2018 files carry the report date as a datetime cell in the header block
        date = next((v.date() for r in cells[:6] for v in r if isinstance(v, dt.datetime)), None)
    col = {}
    for r in cells[:6]:
        for j, v in enumerate(r):
            s = re.sub(r"\s+", " ", str(v).upper()) if v is not None else ""
            if "MONITORED" in s and "cap" not in col:
                col["cap"] = j
            elif "TODAY'S PROGRAM" in s or "TODAYS PROGRAM" in s:
                col["prog"] = j
            elif "TODAY'S ACTUAL" in s or "TODAYS ACTUAL" in s:
                col["act"] = j
            elif "COAL STOCK" in s:
                col["coal"] = j
            elif "UNDER" in s and "OUTAGE" in s:
                col["out"] = j
    if date is None or not {"cap", "prog", "act"} <= set(col):
        return None
    region = state = sector = typ = None
    stations, states = [], []
    cur = None

    def val(r, k):
        return _num(r[col[k]]) if k in col and col[k] < len(r) else None

    for i, r in enumerate(cells[5:], start=5):
        t = _texts(r)
        if not t:
            continue
        head = t[0].upper()
        if _num(t[0]) is not None:  # unlabelled subtotal row (label cells blank, first text is a value)
            continue
        has_vals = val(r, "cap") is not None
        if head == "REGION TOTAL" or head == "ALL INDIA TOTAL" or head.startswith("GRAND TOTAL"):
            continue
        if head == "STATE TOTAL":
            states.append(dict(date=date, region=region, state_raw=state, cap_mw=val(r, "cap"), prog_gwh=val(r, "prog"),
                               act_gwh=val(r, "act"), outage_mw=val(r, "out")))
            continue
        if head.startswith("SECTOR"):
            sector = next((s for s in t[1:] if _num(s) is None), None) or (t[0].split(":", 1)[-1].strip() or None)
            continue
        if head.startswith("TYPE"):
            typ = next((s for s in t[1:] if _num(s) is None), None) or (t[0].split(":", 1)[-1].strip() or None)
            continue
        # some files (e.g. 16 Aug 2021) print sector and type as unprefixed subtotal rows with values
        if head.endswith(" SECTOR"):
            sector = t[0]
            continue
        if head in TYPE_NAMES:
            typ = t[0]
            continue
        if head == "UNIT":
            if cur is not None:
                cur["units"] += 1
                if (val(r, "out") or 0) > 0:
                    cur["units_out"] += 1
            continue
        if not has_vals:
            nxt = next((_texts(c) for c in cells[i + 1:i + 3] if _texts(c)), [""])
            if nxt and nxt[0].upper() == "REGION TOTAL" or head in REGION_NAMES:
                region, state = t[0].title(), None
            elif nxt and nxt[0].upper() == "STATE TOTAL":
                state = t[0]
            continue
        cur = dict(date=date, region=region, state_raw=state, sector=sector, type=typ, station=t[0],
                   cap_mw=val(r, "cap"), prog_gwh=val(r, "prog"), act_gwh=val(r, "act"), coal_days=val(r, "coal"),
                   outage_mw=val(r, "out"), units=0, units_out=0)
        stations.append(cur)
    return {"stations": stations, "states": states}
