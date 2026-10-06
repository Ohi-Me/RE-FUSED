# data

| Folder | What it holds |
|---|---|
| `processed/` | The final datasets: State-day panel, all-India 15-minute and hourly series, hourly market prices, `CHECKSUM.txt` |
| `interim/` | Each source parsed into a table before joining, with parse logs |
| `docs/` | Datasheet, field registry (source, unit and publication lag of every column), quality reports, validation |
| `raw/` | One `MANIFEST.csv` per source: URL, download time, size and SHA-256 of every original report. The report files themselves are not redistributed; `../codes/scripts/acquire/` downloads them again |
