"""Parser for the CEA Daily Renewable Generation Report (Renewable Project Monitoring Division), Govt. of India.

Figures are MU net (= GWh). State rows give today's wind, solar, other RES (biomass, bagasse, small hydro, others)
and total, followed by the cumulative values for the month.

A report contains up to three tables, whose sheet names and order change from file to file:
  scope "all"   Daily Renewable Generation Report / (All India) / (State+ISGS): all RE located in the State,
                including inter-State generating stations (ISGS). Present in every report.
  scope "ctrl"  Daily Renewable Generation Report (State Control Area): RE embedded in the State's own control area
                (excludes ISGS). Present from May 2020.
  (ISGS)        station-wise ISGS list: not used here.
Tables are classified by the title printed inside the sheet/page. Because some files carry swapped titles, the
classification is checked by magnitude: State+ISGS totals can never be below State-control totals, so when two State
tables are found the one with the larger All-India sum is "all". Disagreements between title and magnitude are
reported in `title_swapped`. If both State tables carry identical figures (one table uploaded twice) the scope cannot be
established and no rows are returned (`title_swapped` = "duplicate_tables").
Early reports (May–Oct 2019) are region-level only and yield no State rows. Formats: XLSX (2020-12 onward), PDF.
Output: dict(date, title_swapped, rows=[dict(scope, state_raw, wind_gwh, solar_gwh, other_re_gwh, total_re_gwh,
        layout)]).
"""
import datetime as dt
import re

import numpy as np
import pandas as pd

NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
AGG = re.compile(r"Region|क्षेत्र|All India|Total|कुल", re.I)


def _date_from(text, fname):
    m = re.search(r"(\d{1,2})[\s-]+([A-Za-z]{3})[a-z]*[\s-]+(\d{4}|\d{2})\b", text)
    if m:
        y = m.group(3) if len(m.group(3)) == 4 else "20" + m.group(3)
        try:
            return dt.datetime.strptime(f"{m.group(1)} {m.group(2)} {y}", "%d %b %Y").date()
        except ValueError:
            pass
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", fname)
    return dt.date(*map(int, m.groups())) if m else None


def _kind(title_text):
    t = " ".join(str(title_text).split())
    if re.search(r"\(\s*ISGS\s*\)", t):
        return "isgs"
    if re.search(r"State\s*Control\s*Area", t, re.I):
        return "ctrl"
    if re.search(r"Monthly", t, re.I):
        return "skip"
    if re.search(r"Daily\s*Renewable", t, re.I):
        return "all"
    return None


def _english(name):
    s = str(name).replace("\n", " ")
    if "/" in s:
        s = s.split("/")[-1]
    return re.sub(r"\s+", " ", s).strip()


def _record(name, nums):
    """Map a row's numbers to fields. Standard layout: wind, solar, other, total (then month-to-date values). Some
    files omit the 'other RES' column (wind, solar, total, ...); that layout is recognised when wind + solar = total.
    Rows matching neither layout are returned with NaN values and `layout` = "inconsistent"."""
    n = [np.nan if v is None else v for v in nums]
    if len(n) >= 4 and abs(np.nansum(n[:3]) - n[3]) <= 0.05:
        return dict(state_raw=name, wind_gwh=n[0], solar_gwh=n[1], other_re_gwh=n[2], total_re_gwh=n[3], layout="std")
    if len(n) >= 3 and abs(np.nansum(n[:2]) - n[2]) <= 0.05:
        return dict(state_raw=name, wind_gwh=n[0], solar_gwh=n[1], other_re_gwh=n[2] - np.nansum(n[:2]),
                    total_re_gwh=n[2], layout="no_other_col")
    return dict(state_raw=name, wind_gwh=np.nan, solar_gwh=np.nan, other_re_gwh=np.nan, total_re_gwh=np.nan,
                layout="inconsistent")


def _rows_from_cells(cells):
    out, seen = [], set()
    for r in cells:
        k = next((j for j, v in enumerate(r) if isinstance(v, str) and "/" in v), None)
        if k is None:
            continue
        name = _english(r[k])
        if name in seen:
            continue
        nums = []
        for v in r[k + 1:]:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        if len(nums) >= 4 and not AGG.search(name) and not re.search(r"State|Wind|Solar", name, re.I):
            seen.add(name)
            out.append(_record(name, nums))
    return out


def _rows_from_lines(lines):
    out, seen, i = [], set(), 0
    while i < len(lines):
        l = lines[i]
        if "/" in l and not NUM.match(l) and not re.search(r"Figures|Report|Division|Authority|State/|Energy", l):
            name, j = _english(l), i + 1
            while j < len(lines) and not NUM.match(lines[j]) and "/" not in lines[j] and len(name) < 60:
                name = (name + " " + lines[j]).strip()
                j += 1
            nums = []
            while j < len(lines) and (NUM.match(lines[j]) or lines[j] == "-") and len(nums) < 8:
                nums.append(np.nan if lines[j] == "-" else float(lines[j]))
                j += 1
            if len(nums) >= 4 and not AGG.search(name) and name not in seen:
                seen.add(name)
                out.append(_record(name, nums))
            i = j
        else:
            i += 1
    return out


def _assign(tables):
    """tables: list of (title_kind, rows). Returns ({scope: rows}, title_swapped)."""
    st = [(k, rows) for k, rows in tables if k not in ("isgs", "skip") and len(rows) >= 10]
    if not st:
        return {}, False
    if len(st) == 1:
        k, rows = st[0]
        return {"ctrl" if k == "ctrl" else "all": rows}, False
    tot = [np.nansum([r["total_re_gwh"] for r in rows]) for _, rows in st[:2]]
    if abs(tot[0] - tot[1]) < 0.05:  # the same table uploaded twice (e.g. 10 Feb 2022): scope cannot be established
        return {}, "duplicate_tables"
    big, small = (st[0], st[1]) if tot[0] >= tot[1] else (st[1], st[0])
    swapped = big[0] == "ctrl" or small[0] != "ctrl"
    return {"all": big[1], "ctrl": small[1]}, swapped


def _pack(date, scoped, swapped):
    rows = [dict(scope=s, **r) for s, rr in scoped.items() for r in rr]
    return dict(date=date, title_swapped=swapped, rows=rows)


def parse_xlsx(path):
    xl = pd.ExcelFile(path)
    tables, date = [], None
    for sh in xl.sheet_names:
        x = pd.read_excel(path, sheet_name=sh, header=None)
        cells = x.astype(object).where(pd.notna(x), None).values.tolist()
        head = " ".join(str(v) for r in cells[:6] for v in r if v is not None)
        k = _kind(head)
        if k == "all" and date is None or (date is None and k == "ctrl"):
            date = _date_from(head, str(path))
        tables.append((k, _rows_from_cells(cells)))
    return _pack(date, *_assign(tables))


def parse_pdf(path):
    import pymupdf
    doc = pymupdf.open(path)
    groups, date, cur = [], None, None
    for pg in doc:
        lines = [l.strip() for l in pg.get_text().splitlines() if l.strip()]
        k = _kind(" ".join(lines[:15]))
        if date is None and k in ("all", "ctrl"):
            date = _date_from(" ".join(lines[:40]), str(path))
        if k is None and cur is not None:  # continuation page of the previous table
            groups[-1][1].extend(lines)
            continue
        cur = k or "all"
        groups.append((cur, list(lines)))
    if date is None:  # title printed as an image or unreadable glyphs: use the listing date in the file name
        date = _date_from("", str(path))
    return _pack(date, *_assign([(k, _rows_from_lines(lines)) for k, lines in groups]))
