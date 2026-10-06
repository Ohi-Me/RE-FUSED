# RE-FUSED panel QC report

Panel: 104,550 rows × 107 columns; 34 entities; 2018-04-01 → 2026-08-31. Reserved period (≥ 2025-04-01): counts only.

## Plausibility exclusions (values set to missing)

| rule | rows/days |
|---|---|
| qc_energy_met | 6 |
| qc_max_demand | 16 |
| qc_energy_shortage | 18 |
| qc_od_ud | 2 |
| qc_frequency_days | 2 |
| qc_price_values | 0 |
| qc_npp_capacity | 13 |
| qc_npp_station_typo | 145 |
| qc_weather_tmax_lt_tmin | 0 |

## Source availability by split (share of entity-days)

| split | PSP | NPP | CEA RE |
|---|---|---|---|
| train | 0.930 | 0.958 | 0.713 |
| validation | 1.000 | 1.000 | 0.968 |
| dev_test | 0.995 | 1.000 | 0.969 |
| reserved | 1.000 | 1.000 | 0.433 |

## Cross-source check (train–dev_test only)

Ratio (NPP conventional gross + CEA RE + PSP actual drawal) / PSP energy met, median by entity. Values above 1 are expected where in-State generation is gross and part of it is scheduled to other States through ISGS or bilateral contracts; values far from 1 identify entities whose generation is not represented by the in-State reports (for use as a documented limitation, not as an exclusion). Days with energy met ≤ 5 GWh are omitted, so small north-eastern States have few or no rows; DVC is negative because its stations are listed under West Bengal and Jharkhand in the NPP report.

| entity | median ratio | n |
|---|---|---|
| AP | 1.45 | 1894 |
| AR | 5.4 | 1 |
| AS | 1.56 | 1897 |
| BR | 2.23 | 1896 |
| CG | 4.56 | 1896 |
| CH | 1.0 | 711 |
| DL | 1.0 | 1897 |
| DNHDD | 1.0 | 1898 |
| DVC | -0.6 | 474 |
| GA | 0.96 | 1895 |
| GJ | 1.28 | 1895 |
| HP | 3.13 | 1882 |
| HR | 1.14 | 1897 |
| JH | 3.56 | 1898 |
| JK | 1.62 | 1897 |
| KA | 1.31 | 1894 |
| KL | 0.98 | 1893 |
| MH | 1.15 | 1894 |
| ML | 0.97 | 1729 |
| MP | 2.22 | 1891 |
| NL | 0.25 | 1 |
| OD | 2.09 | 1895 |
| PB | 1.03 | 1861 |
| PY | 1.02 | 1884 |
| RJ | 1.33 | 1893 |
| TG | 1.25 | 1895 |
| TN | 1.47 | 1886 |
| TR | 4.02 | 599 |
| UK | 1.5 | 1892 |
| UP | 1.5 | 1878 |
| WB | 1.75 | 1894 |

## Missingness by block (share of cells missing)

| block | train–dev_test | reserved |
|---|---|---|
| availability | 0.000 | 0.000 |
| calendar | 0.000 | 0.000 |
| carbon | 0.126 | 0.154 |
| conventional generation | 0.031 | 0.000 |
| demand | 0.051 | 0.000 |
| deviation | 0.209 | 0.103 |
| key | 0.000 | 0.000 |
| price | 0.185 | 0.000 |
| qc | 0.000 | 0.000 |
| regional | 0.078 | 0.027 |
| renewables | 0.274 | 0.567 |
| system stress | 0.191 | 0.147 |
| weather | 0.000 | 0.469 |

## Longest consecutive gap in energy met (train–dev_test)

AP: 3, AR: 3, AS: 3, BR: 3, CG: 3, CH: 3, DL: 3, DNHDD: 3, DVC: 3, GA: 3, GJ: 3, HP: 3, HR: 3, JH: 3, JK: 3, KA: 3, KL: 3, MH: 3, ML: 3, MN: 3, MP: 3, MZ: 3, NL: 3, OD: 3, PB: 3, PY: 3, RJ: 3, SK: 3, TG: 3, TN: 3, TR: 3, UK: 3, UP: 3, WB: 3
