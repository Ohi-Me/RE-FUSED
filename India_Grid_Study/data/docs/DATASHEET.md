# Datasheet — RE-FUSED India State-Day Power System Panel (O1)

Version 1.0, 14 Sep 2026. Structure follows *Datasheets for Datasets* (Gebru et al., 2021).
Content hash: `data/processed/CHECKSUM.txt` (row-order-independent SHA-256 of the panel; reproduced exactly on
re-build).

## Motivation

* **Purpose.** A provenance-tracked, leakage-aware daily panel of Indian State control areas that joins demand,
  scheduled and actual drawal (deviation), conventional and renewable generation, bid-area prices, system stress,
  carbon intensity and weather. It supports proposal objectives O2 (forecasting with calibrated uncertainty),
  O3 (context fusion and carbon layer), O4 (context-aware deviation assessment) and O5 (risk-aware scheduling).
* **Gap addressed.** Public Indian power data are scattered across PDF/XLS reports of Grid-India, CEA and IMD, with
  changing layouts and no joined State-level record of deviation, prices, generation and carbon. The pilot dataset
  (RE-FUSED-0) used non-government aggregators, had a fully missing consumption field, and had no deviation or official
  price signal.

## Composition

| Item | Value |
|---|---|
| Unit | Grid-India State control area × calendar day |
| Entities | 34 (30 States/UTs, DNH & DD merged, Chandigarh, Puducherry, DVC); 18 pilot-continuity entities flagged |
| Period | 1 Apr 2018 – 31 Aug 2026 (3,075 days) |
| Rows × columns | 104,550 × 107 (10.17 million non-missing values) |
| Entity-days with energy met | 100,156 |
| Splits | train 62,084 (FY2018-19 … FY2022-23); validation 12,444 (FY2023-24); development test 12,410 (FY2024-25); reserved 17,612 (1 Apr 2025 – 31 Aug 2026, blinded) |
| Blocks | demand (4), deviation (6), conventional generation (25), renewables (8), price (7), carbon (5), system stress (11), regional/national (14), weather (6), calendar (4), availability (3), QC flags (8), keys (6) |

Every column is described in `FIELD_REGISTRY.csv` (source, unit, availability lag, note, missing share by split).

**Targets and outcomes present:** energy met, max demand, actual drawal, OD/UD and its absolute value, max OD/UD,
energy shortage, conventional generation (total and by fuel), RE generation (State + ISGS and State control area),
bid-area DAM ACP and DSM normal rate, all-India frequency stress.

**Missing data (train–development test, share of cells):** price 18.5% (DSM files begin 19 Dec 2018), renewables 27.4%
(CEA State reports begin 1 Jul 2019), deviation 20.9% (max UD printed only from 2023), conventional generation 3.1%
(NPP gap 13 Mar – 31 May 2020), demand 5.1% (129 PSP dates unavailable on the Grid-India website). In the reserved
period, renewables (CEA publication ended 18 Nov 2025) and weather (IMD gridded data end 31 Dec 2025) are largely
missing; demand, deviation, generation and price are complete.

**Known errors in the sources** are not corrected but excluded in the panel and flagged: 6 energy-met, 16 max-demand,
18 energy-shortage and 2 OD/UD values; 2 frequency days; 13 State-days whose generation exceeds capacity × 24 h × 1.15;
145 State-days containing an isolated station typo. Details: `QC_REPORT.md`.

## Collection

* **Sources (official only):** Grid-India daily PSP reports and DSM rate files; CEA National Power Portal Daily
  Generation Report (sub-report 2); CEA Daily Renewable Generation Report; CEA CO2 Baseline Database v22.0; IMD Pune
  gridded rainfall and temperature. URLs, retrieval times and SHA-256 of all 18,000+ files: `raw/*/MANIFEST.csv`;
  summary: `SOURCE_REGISTRY.md`.
* **Timeframe of collection:** 14 Sep 2026, by scripted HTTP requests to public pages (no login).
* **Publication timing measured** (used for availability lags): PSP median 10.6 h after the end of the data day
  (90% within 37 h); NPP generation d+1 evening (later at weekends); CEA RE d+1 to d+2; Grid-India DSM/ACP files uploaded
  in batches 8–11 days after the date; CO2 factors after the financial year.

## Preprocessing

Deterministic parsers (`04_code/refused/parse_*.py`) and build steps (`04_code/scripts/build/b01…b07`):

1. **PSP:** coordinate-based PDF table reading and header-mapped XLS reading. The data date is the "Date of
   Reporting" − 1; cover-letter dates resolve collisions. DD and DNH are summed.
2. **DSM:** three regulatory layouts (R1/R2/R3) are harmonised to ₹/MWh by bid area.
3. **NPP:** station rows are aggregated by State and fuel. Station sums reproduce the report's State totals exactly.
4. **CEA RE:** tables are classified by in-sheet title and magnitude into State+ISGS and State-control scopes.
   Layouts without an "other RES" column are recognised.
5. **Carbon:** CEA emissions / CEA net generation × net-to-gross ratio from complete years gives a factor on NPP
   gross generation, by State × fuel × FY. Factors are lagged FY−2 (operational) and FY−1 (accounting); a
   consumption-based intensity uses net drawal at the regional intensity.
6. **Weather:** IMD grid cells at 1–4 load centres per State; CDD24 and HDD18.
7. **QC:** plausibility rules set values to missing with flags. Targets are never imputed; structural zeros are
   written for fuels without capacity.

Raw files are retained unchanged; the full pipeline re-runs from raw files.

## Uses

* **Intended:** day-ahead to 3-day forecasting under realistic information sets (`avail_lag_days`), probabilistic
  calibration studies, deviation-cost and risk analyses, carbon accounting at State level, and policy analysis of
  regulatory regimes (DSM Regulations 2022/2024).
* **Not suitable for:** intra-day (block-level) State deviation analysis (daily net values only); plant-level dispatch
  or emissions; settlement or legal use; conclusions about generation "belonging" to a State (NPP generation is
  location-based and includes central stations that supply other States).
* **Leakage guidance:** a model issuing a forecast for day t may use a column only up to date t − 1 − `avail_lag_days`
  relative to its issue day. Ex-post columns (`co2_*_acc`, IMD final gridded weather) are marked in the registry.

## Distribution

Code, manifests and derived tables are to be released with attribution to Grid-India, CEA and IMD. Raw report files
are not redistributed until written permission is obtained (the sources' websites do not state an open licence);
anyone can re-download them from the recorded URLs and verify the hashes.

## Maintenance

Rebuild after new downloads with `b01 → b07`. Parsers log every file's status, and the checksum changes with any
content change. Layout changes of the sources (seen in 2018, 2020, 2021, 2023 and 2024) must be checked against
the validation list in `SOURCE_REGISTRY.md` §3 before a new version is released.
