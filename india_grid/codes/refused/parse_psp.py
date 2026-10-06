"""Parser for Grid-India (NLDC) Daily Power Supply Position (PSP) reports, PDF (2017–2022) and XLS (2023–).

The report for "previous day" D is issued on D+1 ("Date of Reporting"). All outputs are keyed by the data date D.
Units: MU = GWh (1 MU = 10^6 kWh). Outputs (one dict of lists per file):

  state   date, state_raw, region, max_demand_mw, peak_shortage_mw, energy_met_gwh, drawal_schedule_gwh, od_ud_gwh,
          max_od_mw, max_ud_mw, energy_shortage_gwh
  region  date, region (NR, WR, SR, ER, NER, ALL), section, item, value
  freq    date, fvi, pct_lt_49_7, pct_49_7_49_8, pct_49_8_49_9, pct_lt_49_9, pct_49_9_50_05, pct_gt_50_05

PDF strategy: words with coordinates are clustered into lines; numbers are assigned to table columns by x-position
so that values that drift onto a neighbouring text line are still attributed to the right row. Non-PSP files that
the website lists under the PSP category (e.g. frequency charts) are detected and skipped.
"""
import datetime as dt
import re

import numpy as np

NUM = re.compile(r"^[-−]?\d+(?:\.\d+)?$")
NA = re.compile(r"^-{1,}$|^[—–]+$|^NA$", re.I)
REGIONS = ["NR", "WR", "SR", "ER", "NER"]
STATE_COLS = ["max_demand_mw", "peak_shortage_mw", "energy_met_gwh", "drawal_schedule_gwh", "od_ud_gwh", "max_od_mw",
              "energy_shortage_gwh"]
FREQ_COLS = ["fvi", "pct_lt_49_7", "pct_49_7_49_8", "pct_49_8_49_9", "pct_lt_49_9", "pct_49_9_50_05", "pct_gt_50_05"]
REGION_ITEMS = [  # (section, item, label regex)
    ("A", "evening_peak_demand_mw", r"^Demand Met.*(Evening|Peak)"),
    ("A", "peak_shortage_mw", r"^(Peak Shortage|Sho[r]?[a]?t[a]?ge \(MW\) during Evening)"),
    ("A", "energy_met_gwh", r"^Energy Met"),
    ("A", "hydro_gen_gwh", r"^Hydro Gen"),
    ("A", "wind_gen_gwh", r"^Wind Gen"),
    ("A", "solar_gen_gwh", r"^Solar Gen"),
    ("A", "energy_shortage_gwh", r"^Energy Shortage"),
    ("A", "max_demand_mw", r"^Maximum Demand Met"),
    ("E", "ir_schedule_gwh", r"^Schedule\s*\(MU\)"),
    ("E", "ir_actual_gwh", r"^Actual\s*\(MU\)"),
    ("E", "ir_odud_gwh", r"^O/D/U/D"),
    ("F", "outage_central_mw", r"^Central Sector"),
    ("F", "outage_state_mw", r"^State Sector"),
    ("F", "outage_total_mw", r"^Total$"),
    ("G", "gen_coal_gwh", r"^Coal$"),
    ("G", "gen_lignite_gwh", r"^Lignite$"),
    ("G", "gen_coal_lignite_gwh", r"^Thermal \(Coal & Lignite\)"),
    ("G", "gen_hydro_gwh", r"^Hydro$"),
    ("G", "gen_nuclear_gwh", r"^Nuclear$"),
    ("G", "gen_gas_gwh", r"^Gas, Naptha & Diesel"),
    ("G", "gen_res_gwh", r"^RES \(Wind, Solar"),
    ("G", "gen_total_gwh", r"^Total$"),
]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def to_num(tok):
    tok = tok.replace("−", "-").replace(",", "").strip()
    if NUM.match(tok):
        return float(tok)
    return None


def parse_report_date(text):
    """'Date of Reporting: 16-Jul-2020' / '1-Jan-18' / '01-Aug-2024' -> data date (reporting date - 1 day)."""
    m = re.search(r"(\d{1,2})[-\s]([A-Za-z]{3})[A-Za-z]*[-\s](\d{2,4})", text)
    if not m:
        return None
    d, mon, y = int(m.group(1)), MONTHS.get(m.group(2).lower()[:3]), int(m.group(3))
    if mon is None:
        return None
    y = y + 2000 if y < 100 else y
    try:
        return dt.date(y, mon, d) - dt.timedelta(days=1)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------------------- PDF
def _lines(page, tol=3.5):
    ws = sorted(page.get_text("words"), key=lambda w: (w[1] + w[3]) / 2)
    out = []
    for w in ws:
        yc = (w[1] + w[3]) / 2
        if out and abs(out[-1]["y"] - yc) <= tol:
            out[-1]["w"].append(w)
            out[-1]["y"] = np.mean([(v[1] + v[3]) / 2 for v in out[-1]["w"]])
        else:
            out.append({"y": yc, "w": [w]})
    for L in out:
        L["w"].sort(key=lambda w: w[0])
        L["text"] = " ".join(w[4] for w in L["w"])
    return out


def _split_label_values(words):
    """Leading non-numeric words form the label; the rest are value tokens (numbers or NA dashes) with x-centres."""
    label, vals, depth = [], [], 0
    for w in words:
        t = w[4]
        opens, closes = t.count("("), t.count(")")
        inside = depth > 0 or opens > closes
        depth = max(0, depth + opens - closes)
        if not vals and (inside or (to_num(t) is None and not NA.match(t))):
            label.append(t)  # numbers inside parentheses, e.g. "(at 2000 hrs; from RLDCs)", belong to the label
        else:
            vals.append(((w[0] + w[2]) / 2, to_num(t)))
    return " ".join(label), vals


def _column_centres(rows, k):
    """x-centres of k columns from rows that have exactly k value tokens."""
    good = [np.array([x for x, _ in v]) for v in rows if len(v) == k]
    if len(good) < 3:
        return None
    return np.median(np.vstack(good), axis=0)


def _assign(vals, centres):
    out = [None] * len(centres)
    for x, v in vals:
        j = int(np.argmin(np.abs(centres - x)))
        if abs(centres[j] - x) < 40 and out[j] is None:
            out[j] = v
    return out


def _letter_date(doc):
    """Data date implied by the cover letter ('दिनांक: 27th Feb 2019' → 26 Feb 2019), or None.
    The table page's 'Date of Reporting' is occasionally not updated from the previous day's report (e.g. the report
    issued on 27 Feb 2019), so the build uses this second date to resolve two different reports claiming one date."""
    for p in list(doc)[:2]:
        t = " ".join(p.get_text().split())
        m = re.search(r"(?:ांक|Dated?)\s*:?\s*(\d{1,2})(?:st|nd|rd|th)?[\s.\-/]*([A-Za-z]{3})[A-Za-z]*[\s.,\-/]*(\d{4})", t)
        if m and MONTHS.get(m.group(2).lower()[:3]):
            try:
                return dt.date(int(m.group(3)), MONTHS[m.group(2).lower()[:3]], int(m.group(1))) - dt.timedelta(days=1)
            except ValueError:
                return None
    return None


def parse_pdf(path, state_aliases):
    import pymupdf
    doc = pymupdf.open(path)
    page = None
    for p in doc:
        if "Power Supply Position in States" in p.get_text():
            page = p
            break
    if page is None:
        return None
    L = _lines(page)
    full = "\n".join(x["text"] for x in L)
    date = None
    for x in L:
        if "Date of Reporting" in x["text"] or re.search(r"\d{1,2}-[A-Za-z]{3}-\d{2,4}", x["text"]):
            date = parse_report_date(x["text"].split("Reporting")[-1])
            if date:
                break
    if date is None:
        return None
    ys = {k: next((x["y"] for x in L if x["text"].startswith(k)), None) for k in
          ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]}
    res = {"state": [], "region": [], "freq": [], "meta": {"table_date": date, "letter_date": _letter_date(doc)}}

    # ---- C: states (between C. and D.)
    yC, yD = ys["C."], ys["D."] or 1e9
    segment = [x for x in L if yC is not None and yC < x["y"] < yD]
    parsed = []
    for x in segment:
        lab, vals = _split_label_values(x["w"])
        lab = re.sub(r"^(Region\s*)+", "", lab).strip()
        lab = re.sub(r"^(NR|WR|SR|ER|NER)\s+", "", lab).strip()
        parsed.append({"y": x["y"], "label": lab, "vals": vals})
    centres = _column_centres([p["vals"] for p in parsed if p["label"]], 7)
    if centres is not None:
        orphans = [p for p in parsed if not p["label"] and p["vals"]]
        for p in parsed:
            key = state_aliases(p["label"])
            if not key:
                continue
            vals = list(p["vals"])
            if len(vals) < 7:  # adopt drifted numbers from label-less lines within 9 pt
                for o in orphans:
                    if abs(o["y"] - p["y"]) <= 9:
                        vals += o["vals"]
            row = _assign(vals, centres)
            res["state"].append(dict(date=date, state_raw=p["label"], **dict(zip(STATE_COLS, row)), max_ud_mw=None))

    # ---- B: frequency (All India row)
    for x in L:
        if x["text"].startswith("All India") and ys["B."] and ys["C."] and ys["B."] < x["y"] < ys["C."]:
            _, vals = _split_label_values(x["w"])
            nums = [v for _, v in vals]
            if len(nums) >= 7:
                res["freq"].append(dict(date=date, **dict(zip(FREQ_COLS, nums[:7]))))

    # ---- A, E, F, G: regional tables with 6 (or 7 with % share) columns
    def section_rows(y0, y1, sec):
        rows = [x for x in L if y0 is not None and y0 < x["y"] < (y1 or 1e9)]
        items = [(s, i, rx) for s, i, rx in REGION_ITEMS if s == sec]
        for idx, x in enumerate(rows):
            lab, vals = _split_label_values(x["w"])
            for s, item, rx in items:
                if re.search(rx, lab.strip(), re.I):
                    if len(vals) < 6:  # numbers printed on the adjacent line (multi-line label)
                        for nb in rows[max(0, idx - 1): idx + 2]:
                            if nb is not x:
                                l2, v2 = _split_label_values(nb["w"])
                                if not l2 and len(v2) >= 6 and abs(nb["y"] - x["y"]) <= 12:
                                    vals = v2
                    nums = [v for _, v in vals]
                    if len(nums) >= 6:
                        for reg, v in zip(REGIONS + ["ALL"], nums[:6]):
                            res["region"].append(dict(date=date, region=reg, section=s, item=item, value=v))
                    break

    section_rows(ys["A."], ys["B."], "A")
    section_rows(ys["E."], ys["F."], "E")
    section_rows(ys["F."], ys["G."], "F")
    section_rows(ys["G."], ys["H."], "G")
    return res


# --------------------------------------------------------------------------------------------------------- XLS
def _cell(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return float(v)
    s = str(v).strip()
    if s in ("", "nan") or NA.match(s):
        return None
    return to_num(s)


_XLS_HEADER = [  # (field, regex on the column's concatenated header text), checked in this order
    ("max_od_mw", r"max\s*\.?\s*OD"),
    ("od_ud_gwh", r"OD\s*\(\+\)\s*/\s*UD"),
    ("energy_shortage_gwh", r"Energy\s+Shortage"),
    ("peak_shortage_mw", r"Shortage\s+during|Shortage.*maximum"),
    ("energy_met_gwh", r"Energy\s+Met"),
    ("drawal_schedule_gwh", r"Drawal|Schedule"),
    ("max_demand_mw", r"Max\s*\.?\s*Demand"),
]


def _xls_state_columns(section_rows, state_aliases):
    """Column index of each State field from the header lines of section C, or None if not all are found."""
    head = []
    for r in section_rows:
        if any(isinstance(v, str) and state_aliases(v.strip()) for v in r[:2]):
            break
        head.append(r)
    ncol = max((len(r) for r in head), default=0)
    text = {k: " ".join(str(r[k]) for r in head if k < len(r) and r[k] is not None) for k in range(ncol)}
    text = {k: re.sub(r"\s+", " ", t) for k, t in text.items() if t.strip()}
    out = {}
    for field, rx in _XLS_HEADER:
        for k, t in sorted(text.items()):
            if k not in out.values() and re.search(rx, t, re.I):
                out[field] = k
                break
    return out if set(out) == set(STATE_COLS) else None


def parse_xls(path, state_aliases):
    import pandas as pd
    x = pd.read_excel(path, sheet_name=0, header=None)
    rows = [[None if (isinstance(v, float) and np.isnan(v)) else v for v in r] for r in x.values.tolist()]
    flat = [" ".join(str(v) for v in r if v is not None) for r in rows]
    date = None
    for s in flat[:6]:
        if "Reporting" in s:
            date = parse_report_date(s.split("Reporting")[-1])
    if date is None:
        return None
    res = {"state": [], "region": [], "freq": []}
    sec_idx = {}
    for i, s in enumerate(flat):
        m = re.match(r"^\s*([A-H])\.\s", s)
        if m and m.group(1) not in sec_idx:
            sec_idx[m.group(1)] = i
    iC, iD = sec_idx.get("C"), sec_idx.get("D", len(rows))
    colmap = _xls_state_columns(rows[(iC or 0) + 1: iD], state_aliases)
    for r in rows[(iC or 0) + 1: iD]:
        texts = [str(v).strip() for v in r[:2] if v is not None and to_num(str(v)) is None]
        name = None
        for t in reversed(texts):
            if state_aliases(t):
                name = t
                break
        if not name:
            continue
        j = max(k for k in range(len(r)) if r[k] is not None and str(r[k]).strip() == name)
        if colmap:  # header-based positions (robust to empty spacer columns, e.g. 21–22 Feb 2023)
            vals = [r[colmap[c]] if colmap[c] < len(r) else None for c in STATE_COLS]
        else:
            vals = r[j + 1: j + 8]
        maxod, maxud = vals[5] if len(vals) > 5 else None, None
        if isinstance(maxod, str) and "/" in maxod:
            a, b = maxod.split("/", 1)
            maxod, maxud = _cell(a), _cell(b)
        else:
            maxod = _cell(maxod)
        row = [_cell(v) for v in vals[:5]] + [maxod] + [_cell(vals[6]) if len(vals) > 6 else None]
        res["state"].append(dict(date=date, state_raw=name, **dict(zip(STATE_COLS, row)), max_ud_mw=maxud))
    iB = sec_idx.get("B")
    if iB is not None:
        for r in rows[iB: iB + 4]:
            if r and str(r[0]).strip() == "All India":
                nums = [_cell(v) for v in r[1:] if _cell(v) is not None]
                if len(nums) >= 7:
                    res["freq"].append(dict(date=date, **dict(zip(FREQ_COLS, nums[:7]))))
    bounds = {"A": ("A", "B"), "E": ("E", "F"), "F": ("F", "G"), "G": ("G", "H")}
    for sec, (a, b) in bounds.items():
        if a not in sec_idx:
            continue
        for r in rows[sec_idx[a] + 1: sec_idx.get(b, len(rows))]:
            label = str(r[0]).strip() if r and r[0] is not None else ""
            for s, item, rx in REGION_ITEMS:
                if s == sec and re.search(rx, label, re.I):
                    nums = [_cell(v) if not (isinstance(v, str) and NA.match(v.strip())) else np.nan for v in r[1:]]
                    nums = [v for v in nums if v is None or not isinstance(v, str)]
                    vals = [v for v in nums if v is not None]
                    if len(vals) >= 6:
                        for reg, v in zip(REGIONS + ["ALL"], vals[:6]):
                            res["region"].append(dict(date=date, region=reg, section=s, item=item,
                                                      value=None if (isinstance(v, float) and np.isnan(v)) else v))
                    break
    return res
