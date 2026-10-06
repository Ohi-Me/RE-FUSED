# codes

| Folder | What it holds |
|---|---|
| `refused/` | The library: report parsers, panel builder, targets, models, gating, scheduling, statistics, the blinding guard |
| `scripts/` | One script per pipeline step (see `scripts/README.md`) |
| `design/` | Research design, development protocols, the frozen pre-registration and its hash, deviations |
| `tests/` | The leakage test: no feature may use data before it was published |

Everything is run through `../run_all.py`.
