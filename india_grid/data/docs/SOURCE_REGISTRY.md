# RE-FUSED Source Registry (O1 provenance)

Version 1.0 — 14 Sep 2026. Every raw file was downloaded by `04_code/scripts/acquire/*.py`. Each source folder in
`03_data/raw/<source>/` has a `MANIFEST.csv` listing URL, local path, HTTP status, bytes, SHA-256 and UTC retrieval
time for every request, including unavailable files. Interim tables are rebuilt from the raw files by
`04_code/scripts/build/b01…b06`.

Scope rule: only official Government of India sources are used (Grid-India, a Government of India enterprise under
the Ministry of Power; CEA and its National Power Portal; IMD). No aggregator, exchange or third-party files enter
the RE-FUSED panel. The pilot (RE-FUSED-0) folder is kept as an exact copy for cross-checks only
(`raw/legacy_refused0/MANIFEST_LEGACY.csv`: 162 files, 816.9 MB, SHA-256 per file).

## 1. Inventory

| # | Source (publisher) | Product | Access | Raw files (ok / not available) | Size | Data-date coverage | Interim table |
|---|---|---|---|---|---|---|---|
| G1 | Grid-India (NLDC) | Daily Power Supply Position (PSP) report | Public file API `webapi.grid-india.in/api/v1/file` → `webcdn.grid-india.in` | 3,448 / 5 | 2.67 GB | 2017-05-08 → 2026-09-13; 3,284 dates | `psp_state_day`, `psp_region_day`, `psp_freq_day` |
| G2 | Grid-India (NLDC) | DSM / deviation price files (normal rate, DAM/RTM ACP, reference rates by bid area) | Same API, FileType `F_PG00403` | 404 / 0 | 87.6 MB | 2018-12-19 → 2026-09-06; 2,818 dates | `dsm_block`, `dsm_daily` |
| C1 | CEA – National Power Portal | Daily Generation Report, sub-report 2 (unit-wise programme/actual, capacity, outage, coal stock days) | `npp.gov.in/public-reports/cea/daily/dgr/…` | 3,022 / 74 | 1.05 GB | 2018-03-22 → 2026-09-11; 3,019 usable dates | `npp_station_day`, `npp_state_day` |
| C2 | CEA – NPP | DGR sub-report 6 (hydro reservoirs) | same | 2,762 / 334 | 54.5 MB | 2018-03-23 → 2026-09-11 | (not yet parsed) |
| C3 | CEA – NPP | DGR sub-report 10 (planned/forced maintenance) | same | 3,018 / 78 | 166.5 MB | 2018-03-23 → 2026-09-11 | (not yet parsed) |
| C4 | CEA – NPP | Daily coal stock report | `npp.gov.in/public-reports/cea/daily/fuel/…` | 2,837 / 2,555* | 278.4 MB | 2018-09-25 → 2026-09-11 | (not needed: coal-stock days per station are in C1) |
| C5 | CEA (RE Project Monitoring Division) | Daily Renewable Generation Report | `cea.nic.in` listing (`admin-ajax.php?action=getpostsfordatatables`) | 2,361 / 8 | 382.3 MB | 2019-07-01 → 2025-11-18; 2,285 dates | `cea_re_state_day` |
| C6 | CEA (Thermal Performance Evaluation) | CO2 Baseline Database v19–v22 + user guides | `cea.nic.in` | 48 / 0 | 68.0 MB | FY2017-18 → FY2025-26 (v22.0) | `co2_ef_state_fuel_fy` |
| I1 | IMD Pune | Gridded daily Tmax, Tmin (1°) and rainfall (0.25°) | POST forms `imdpune.gov.in/cmpg/Griddata/*.php` | 27 / 0 | 254.2 MB | 2017-01-01 → 2025-12-31 | `imd_state_day` |

\*C4: each date is tried as `.xls` and `.xlsx`; one of the two is expected to be 404.
Total raw official data: **5.01 GB**, plus the 816.9 MB pilot copy.

## 2. Variables used in the RE-FUSED panel

| Block | Variable | Source | Unit | Notes |
|---|---|---|---|---|
| Demand | energy met, max demand met, peak shortage, energy shortage | G1 §C | GWh, MW | State control areas (34 entities) |
| Deviation (O4/O5 target) | drawal schedule, OD(+)/UD(−), max OD, max UD | G1 §C | GWh, MW | daily net; max UD printed only from 2023 (93.5% missing before) |
| System stress | FVI, % time in frequency bands | G1 §B | index, % | All-India |
| Regional | energy met, hydro/wind/solar, inter-regional schedule/actual/OD, outages, source-wise generation | G1 §A, E, F, G | GWh, MW | |
| Price signal | normal DSM rate, DAM/RTM ACP, reference rate, share of blocks at cap | G2 | ₹/MWh | bid-area resolved; regimes R1 (≤ 2022-12-04), R2, R3 (≥ 2024-09-16) |
| Conventional generation | programme, actual, capacity, outage by fuel; coal-stock days (capacity-weighted), share of coal capacity < 7 days | C1 | GWh, MW, days | gross generation, stations ≥ 25 MW monitored by CEA |
| RE generation | wind, solar, other, total — `all` (State + ISGS) and `ctrl` (State control area, from 2020-06-01) | C5 | GWh | |
| Carbon | emission factor by State × fuel × FY (net and NPP-gross basis) | C6 + C1 | t CO2/MWh | FY−2 for operational features, FY−1 for accounting |
| Weather | Tmax, Tmin, rain, CDD24, HDD18 at State load centres | I1 | °C, mm | ends 2025-12-31 |

## 3. Validation performed

| Check | Result |
|---|---|
| G1 PDF vs XLS parse of the same report (2023-03-15) | identical State values |
| G1 date assignment | data date = "Date of Reporting" − 1. Two reports whose table page kept the previous day's date (e.g. issued 27 Feb 2019) were moved to the date implied by their cover letter; 0 duplicate uploads used twice |
| C1 station rows vs the report's own STATE TOTAL rows | max absolute difference 4.5 × 10⁻¹³ GWh on all 3,019 dates |
| C1 State totals vs pilot India Data Portal copy (76,167 state-days, 2018-03-22 → 2025-04-22) | 100% within 1%; 99.1% within 0.01 GWh |
| C5 State+ISGS totals vs pilot copy (61,504 state-days) | 99.8% within 1%; remaining differences are listing-date errors in the pilot copy (e.g. a "30 Sept" report listed on 30 Oct 2021) |
| C5 row arithmetic (wind + solar + other = total) | 46 `all` and 23 `ctrl` rows fail in both known layouts and are set to missing |
| C6 national coal factor | 0.98 t/MWh net; 0.91–0.95 t/MWh on NPP gross basis (net/gross ratio 0.927, consistent with auxiliary consumption) |
| I1 orientation and climatology | Delhi Tmax Jan 19.4 °C / May 39.5 °C; Mumbai July 2023 rainfall 1,532 mm; sea cells missing |
| Captive-portal guard | responses from a local network login page are never stored (checked: 0 such files in any manifest) |

## 4. Known gaps and limitations (to be reported in the data paper)

1. **G1 PSP, 129 missing data dates in 2018-04-01 → 2026-08-31 (4.2%)**, mostly single days in 2018–2020. 90 are
   caused by the website itself, whose file API points several report dates to one shared file (72 dates share
   `download-manager-files/upload.pdf`, uploaded 11 Apr 2025). The rest are unavailable files (30–31 Dec 2024 return
   HTTP 404 in both XLS and PDF), uploads of another day's report, or reports without the State table. Missing target
   days are excluded from losses and scores, never imputed.
2. **C1 NPP DGR2 is unavailable for 13 Mar – 31 May 2020** (COVID-19 lockdown period) and 29–31 Oct 2019 (shifted
   layout without total rows); 77 dates in the study window.
3. **C5 CEA RE State data end on 2025-11-18** and begin 2019-07-01 (earlier reports are region-level). One file
   (10 Feb 2022) contains the same table twice, so its scope cannot be established and it is not used. For the
   reserved confirmatory period after 2025-11-18, RE generation is available only at regional level from G1 §A and
   as the NPP "programme" of conventional plants; the design treats State RE as an optional input.
4. **G1 deviation is a daily net figure.** Intra-day over- and under-drawal offset each other; max OD/UD (MW) is the
   only intra-day severity signal, and max UD is printed only from 2023.
5. **I1 weather ends 2025-12-31** and is represented by load-centre grid cells, not area- or population-weighted
   State means.
6. **Prices:** power-exchange price files are not government publications; the only price signals used are those
   published by Grid-India in the DSM files (DAM/RTM ACP by bid area and DSM normal rates).
7. **C6 factors** are annual averages by State and fuel (no unit dispatch); operational features use the factor of
   FY−2 because each database version is published after its financial year ends.

## 5. Terms of use and redistribution

All files were downloaded without login from public pages of Government of India websites. The websites' own policy
pages do not grant an explicit open licence: the CEA site footer states "All Rights Reserved", and the NPP copyright
policy refers to permission to reproduce material. Until written permission is obtained, the release plan is:

* publish code, manifests (URL + SHA-256) and the rebuild pipeline, so anyone can re-download and verify every file;
* publish derived State-day tables with full attribution to Grid-India, CEA and IMD;
* do not redistribute raw report files.

Recommended before submitting a data descriptor: request redistribution permission from CEA and Grid-India by
e-mail (a decision for the authors).

## 6. Citation strings (to be used in manuscripts)

* Grid-India (Grid Controller of India Limited), National Load Despatch Centre. *Daily Power Supply Position Report*,
  2017–2026. https://grid-india.in (accessed 14 Sep 2026).
* Grid-India, NLDC. *Deviation Settlement Mechanism rates*, 2018–2026. https://grid-india.in (accessed 14 Sep 2026).
* Central Electricity Authority, National Power Portal. *Daily Generation Report*, 2018–2026. https://npp.gov.in
  (accessed 14 Sep 2026).
* Central Electricity Authority. *Daily Renewable Generation Report*, 2019–2025. https://cea.nic.in (accessed
  14 Sep 2026).
* Central Electricity Authority. *CO2 Baseline Database for the Indian Power Sector, Version 22.0*. https://cea.nic.in
  (accessed 14 Sep 2026).
* Pai, D. S. et al. / India Meteorological Department, Pune. *Gridded daily rainfall (0.25°) and temperature (1°)
  data*, 2017–2025. https://www.imdpune.gov.in (accessed 14 Sep 2026). Exact dataset references to be checked
  against the IMD data page before submission.
