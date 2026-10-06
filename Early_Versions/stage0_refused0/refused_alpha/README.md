# RE-FUSED-Alpha
## Carbon-Uncertainty-Efficient Forecasting & Optimization — Alpha Release

**India 18-State Electricity Grid | 2017–2025 | 47,286 state-day rows**

---

### Five Contributions (N1–N5)

| # | Contribution | V2 | Alpha Upgrade |
|---|---|---|---|
| N1 | PA-LMP | Daily deviation signal | **PA-LMP_t (daily-resolution counterfactual)** + monsoon RE credit + 6-day rolling CVaR proxy |
| N2 | MCAG | 3-way ablation | **5-way ablation** + MCAG-Regime (soft 3-head routing) + cross-test vs TFT-Carbon |
| N3 | GCAL | Daily CFC | CFC counterfactual from daily CI (7-day variance proxy) + Carbon Forecast Error (CFE) metric |
| N4 | Settlement Counterfactual | Not present | **New** — CERC-motivated PA-LMP_t formulation (daily resolution) + regime-coupled DRO-λ dispatch (λ_t sourced live from the `MCAG_Regime` router; 5th PPO arm, replaces the earlier static vol-scaled-λ) |
| N5 | Arch Comparison | 3-model flat | **New** — staged best-of-3 (BiLSTM → TFT → PatchTST) + scale-free metrics (MASE/RMSSE) |
| N6 | Action-Conditioned Dispatch Sim | Static replay only | **New** — `IndiaGridSim` transitions now respond to the agent's own past actions (decaying imprint on `grid_stress_index`/`carbon_budget_pressure`/`coal_outage_stress`), replacing the pure historical-replay dynamics |

> **Resolution honesty:** all data is daily state-day. Legacy "5-min"/"4-hr" names from V2 survive only as column labels (`rolling_cvar4hr` = 6-day rolling q90; `ci_intraday_variance` = 7-day rolling σ). **No sub-daily results are claimed anywhere.**

---

### Architecture Comparison (N5)

**Stage 2 — BiLSTM best-of-3:**
- `BiLSTM-Base` (V2 replication anchor)
- `BiLSTM-Deep` (3-layer + skip connections, captures monsoon regimes)
- `BiLSTM-Attn` (Bahdanau attention — publishable as attention heatmap)

**Stage 3 — TFT best-of-3:**
- `TFT-Lite` (V2 replication anchor)
- `TFT-Full` (proper GRN + static covariate encoder for 18-state heterogeneity)
- `TFT-Carbon` (carbon as static context — N2 vs N4 cross-test: static vs dynamic)

**Stage 4 — PatchTST+MCAG best-of-3:**
- `MCAG-Real` (V2 replication anchor)
- `MCAG-Regime` (soft 3-head routing: normal / coal-critical / monsoon)
- `MCAG-5Min` (legacy name — 7-day carbon-variance gate, N4 integration)

**Decision criterion:**
- **Headline (paper):** MASE / RMSSE on the observable targets (`total_generation_mwh`, `avg_market_price`) vs the naive / seasonal-naive anchors in `results/tables/baseline_metrics.csv`, with Diebold–Mariano (HLN) significance between finalists (Cell 3d).
- **Legacy (V2 comparability only — not a headline metric):**
```
WtScore = 0.35×palmp_MAPE + 0.20×GSI_MAPE + 0.20×price_MAPE + 0.15×logcarbon_MAPE + 0.10×gen_MAPE
```
  Arbitrary weights over heterogeneous targets, dominated by the synthetic `palmp`; retained only so stage selection stays V2-comparable.

---

### Settlement Counterfactual Proposal (N4) — daily resolution

PA-LMP_t formula (daily-resolution counterfactual):
```
PA-LMP_t = LMP_t + ν·CI_norm_t + φ·1[p_t > CVaR90_4hr]·v_norm_t
           + ψ·GSI_norm_t + ω·FCFS_norm_t - ε·MonsoonRECredit_t
```

New components vs V2 (legacy V2 names kept; all are daily-resolution proxies):
- `rolling_cvar4hr` — 6-day trailing q90 proxy of a 4-hr CVaR window
- `monsoon_re_credit` — RE over-delivery credit, reduces PA-LMP_t during monsoon
- `ci_intraday_variance` — σ(CI) over a 7-day window → MCAG-Regime gate input
- `pa_lmp_t` — settlement price signal (daily counterfactual), feeds vol-scaled-λ DRO dispatch (λ is a single fixed constant per run, sized once from train-set PA-LMP_t/price volatility ratio — not time-varying)

---

### New Alpha Metrics

| Metric | Formula | Use |
|---|---|---|
| **MASE / RMSSE** | error scaled vs seasonal-naive (m=7) | **Headline forecast skill (zero-safe, scale-free)** |
| sMAPE | symmetric MAPE, denom \|y\|+\|ŷ\| | Secondary zero-safe point metric |
| Diebold–Mariano (HLN) | equal-accuracy test on loss differentials | Significance between finalists |
| Conditional MAPE | MAPE on coal_critical=1 rows | Tail-risk performance |
| Directional Accuracy (DA) | % correct next-day price direction | Sign-change skill (daily) |
| Carbon Forecast Error (CFE) | mean\|exp(log_carbon_hat) − exp(log_carbon)\| | GCAL accuracy |
| Weighted Score | 35/20/20/15/10 % combination | Legacy V2 architecture selection only |

---

### Directory Structure
```
REFUSED_Ready/
├── notebook/
│   └── REFUSED_Alpha_Final.ipynb   ← Main notebook (36 cells, incl. leakage-fix Cell 1.1b)
├── refused_fixes/                  ← Audit-fix package (imported by the notebook)
│   ├── causal_features.py        ← Train-only refit of leaky features + leakage_scan
│   ├── metrics.py                ← MASE/RMSSE/sMAPE, pinball/CRPS, Diebold–Mariano
│   ├── baselines.py              ← naive / seasonal-naive / moving-average anchors
│   └── units.py                  ← unit registry & rename map, imputation report
├── run_audit_diagnostics.py      ← reproduces leakage/units/imputation evidence
├── run_baselines.py              ← reproduces baseline_metrics.csv
├── LEAKAGE_AUDIT.md              ← full audit + measured results (§7)
├── data/
│   ├── full_dataset.csv          ← 47,286 state-day rows, 18 states
│   ├── RL_TRAIN_DATA.csv         ← Training split (2017–2023)
│   └── RL_TEST_DATA.csv          ← Test split (2024–2025)
├── results/
│   ├── figures/                  ← S1–S6 + master PNG (auto-generated)
│   ├── tables/                   ← metrics_summary.json, baseline_metrics.csv, leakage_scan.csv, …
│   └── outputs/                  ← diagnostic summaries
└── related_works/                ← Literature notes from V2
```

---

### Hardware & Runtime
- **Target GPU:** Tesla T4 (CUDA 13.0, 15,360 MiB, Driver 580.82.07)
- **Full CPU fallback** included — no GPU required for code to run
- **Estimated runtime:** ~25–40 min on T4 (15 epochs × 8 models + 3 PPO runs)

---

### Rigor & Integrity (audit-backed — see LEAKAGE_AUDIT.md)
- **Leakage fixed:** `grid_stress_index` was upstream-normalized on the full 2017–2025 panel (measured refit Δ up to 0.44). Cell 1.1b refits it (plus `price_norm`, `discom_stress`, and median imputation) on **train-only** statistics; `prep()` freezes every quantile/min-max stat on the train split.
- **Imputation disclosed:** 87.3% of state-day rows carry ≥1 imputed value (`market_price`/`coal`/`consumption`); the observed-only robustness slice is 12.7% (5,996 rows) — re-run headline metrics via `refused_fixes.units.observed_only`.
- **Units corrected:** `*_mwh` columns hold GWh-scale values and `carbon_proxy_tons` kilotons; label-only rename map in `refused_fixes.units.RENAME_MAP`.
- **Honest targets:** forecast skill is claimed **only** on observable targets (`total_generation_mwh`, `avg_market_price`); `palmp` / GSI / `fcfs` / `log_carbon` are derived policy overlays, not headline forecasting targets.
- **Baseline-anchored:** see `results/tables/baseline_metrics.csv`. Caution: `avg_market_price` is near-trivially persistent (naive MASE ≈ 0.075 — largely an artifact of the 87.3% imputation smoothing); price-skill claims must beat that anchor, not MAPE.
- **Before submission:** multi-seed re-runs (≥5 seeds, mean±std) + DM tests on the final head-to-head; observed-only robustness pass.
- **Dispatch RL, resolved (see `PUBLICATION_AUDIT_ALPHA.md`):** two audit-flagged weaknesses are now fixed. (1) `IndiaGridSim` dynamics are action-conditioned — an agent's own actions leave a decaying imprint on future `grid_stress_index`/`carbon_budget_pressure`/`coal_outage_stress`, not just historical replay. (2) `adaptive_lambda` is genuinely regime-adaptive — λ_t is sourced live from the trained `MCAG_Regime` router (not a static constant), evaluated as a 4th PPO arm (`DRO_regime_coupled_lambda`). Result: highest mean reward of all 4 arms (0.9562 vs 0.9337 standard PPO) with a 22.5% reward-std reduction vs standard PPO (`results/tables/results_summary.csv`, `results/tables/metrics_summary.json → dispatch_rl`).

---

### Target Journals

---

### Quick Start (Google Colab T4)
```python
# 1. Upload REFUSED_Alpha.zip and unzip
# 2. Runtime → Change runtime type → GPU → T4
# 3. Open REFUSED_Alpha_project/notebook/REFUSED_Alpha_Final.ipynb
# 4. Runtime → Run all
```
