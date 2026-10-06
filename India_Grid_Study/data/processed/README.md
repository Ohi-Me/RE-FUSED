# processed

| File | What it is |
|---|---|
| `refused_state_day.parquet` | 34 State control areas × 3,075 days (1 Apr 2018 – 31 Aug 2026), 107 columns |
| `refused_entities.csv` | The 34 control areas and their codes |
| `CHECKSUM.txt` | Content hash of the panel (independent of row order) |
| `refused_allindia_15min.parquet` | All-India 15-minute SCADA values, 4 Nov 2024 – 13 Sep 2026, with quality flags |
| `refused_allindia_hourly.parquet` | Hourly means of the 15-minute series |
| `refused_allindia_hourly_prices.parquet` | Hourly national day-ahead and real-time market clearing prices |

Every column is explained in `../docs/FIELD_REGISTRY.csv`. Licence: CC BY 4.0.
