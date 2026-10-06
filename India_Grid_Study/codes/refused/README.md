# refused (library)

| Module | What it does |
|---|---|
| `fetch.py` | Polite, logged downloads with SHA-256 manifests |
| `parse_psp.py`, `parse_dsm.py`, `parse_npp.py`, `parse_cea_re.py` | Layout-aware parsers of the official reports |
| `entities.py` | The 34 State control areas and their name variants |
| `o2data.py`, `windows.py` | Forecast targets, features and time windows, respecting publication lags |
| `gates.py`, `selection.py`, `probabilistic.py` | Gated combination, model selection, calibrated intervals |
| `stats.py` | Block bootstrap, multiple-testing corrections |
| `guard.py` | Refuses held-out data until the pre-registration is frozen |
| `dev2.py`, `paths.py`, `report.py` | Round-2 helpers, folder layout, run reports |
