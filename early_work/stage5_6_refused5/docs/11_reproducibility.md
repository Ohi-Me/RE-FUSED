# RE-FUSED-6 reproducibility specification

Everything in `results/refused6/ALL_STATS.md` is generated from the CSVs listed below by
`src/refused6/final_stats.py`. No number in that report is typed by hand.

## 1. Environment

| Item | Version |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.10.11 (`py -3.10`) |
| numpy / pandas / scipy | 2.2.6 / 2.3.3 / 1.15.3 |
| scikit-learn | 1.7.2 (HistGradientBoostingRegressor for all correctors and fine gates) |
| torch | 2.5.1+cu121 (GTX 1660 Ti 6 GB) — used by M4 (neuralforecast), M9 and M9b |
| neuralforecast | 3.2.2 (DLinear, NHITS, PatchTST in M4) |

Warning: `pip install neuralforecast` silently replaces the CUDA torch with a CPU build. Restore with
`pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121`.

Thread setting used by the pipeline: `OMP_NUM_THREADS=4`, `PYTHONIOENCODING=utf-8`.

## 2. Data (MD5)

| File | Bytes | MD5 | Used by |
|---|---|---|---|
| data/india_model_daily.parquet | 1,170,869 | AC232FCF603B81A9472EAD7C93DC8DED | M1, M1b–d, M5c, M7, M7b, M8, M9, M10 |
| data/india_panel_raw_rebuild.parquet | 1,712,179 | CF9C2C8AA4F225AEAB1CA8134BC4443F | data rebuild audit (RE-FUSED-5) |
| data/nyiso_5min.parquet | 173,995,501 | DB5C6C13D7C6E2B9AE07B183349AD80D | M6 |
| data/nyiso_15min.parquet | 91,562,557 | 8B1B884AB2925A39FB9BDC45EECFEBF3 | M6 |
| data/nyiso_1h.parquet | 24,945,807 | BAD703E50A593A5F7097797AFE5DC6C3 | M6, M7b, M8 |
| data/nyiso_1d.parquet | 1,755,505 | 2922D2BA6911591DD4B5288A889A75A3 | RE-FUSED-5 resolution study |
| data/public/ETTh1.csv | 2,589,657 | 8381763947C85F4BE6AC456C508460D6 | M4, M5c, M7b, M8, M9 |
| data/public/ETTh2.csv | 2,417,960 | 51A229A3FC13579DD939364FEFE9C7AB | M5c, M7b, M8, M9 |
| data/public/ETTm1.csv | 10,360,719 | 82D6BD89109C63D075D99C1077B33F38 | M8 |
| data/public/ETTm2.csv | 9,677,236 | 7687E47825335860BF58BCCB31BE0C56 | M8 |

Splits (fixed, chronological):
* India daily (20 series, 2017-09-08 .. 2025-04-22): train ≤ 2022-12-31, val = 2023, test = 2024-01-01 .. 2025-04-22.
  M10 adds origins train ≤ 2020 / val 2021 / test 2022 and train ≤ 2021 / val 2022 / test 2023.
* ETT (7 channels each): 70 / 10 / 20 % of timestamps.
* NYISO hourly (15 zones): the `split` column stored in the parquet.

## 3. Protocol constants (identical in every stage unless the stage says otherwise)

| Constant | Value | Why |
|---|---|---|
| Gate clip | [0, 1.5] for every rung | D6 — identical clipping |
| Gate estimator, coarse rungs | least squares ⟨R, r̂⟩ / ⟨r̂, r̂⟩ on validation | D5 — optimal under squared loss |
| Fine-rung estimator | WLS on z = R / r̂ (clipped ±10), weights r̂², bagged | D8 — ratio gate had variance ~1e18 |
| Corrector reparameterisation | r̂ ← g₁·r̂ with g₁ the global LS gate on validation | D9 — under-scaled corrector |
| Minimum cell size | 30 validation points; else fall back to the next coarser rung | |
| Regime cells | series × tercile of rs7 (India) / rs6 (hourly), thresholds from train | |
| Primary metric | squared-loss skill 1 − MSE / MSE_baseline, macro over series | |
| Seeds | M1: 10; M1b–d, M5c, M8, M9, M9b, M10: 5; M7b India/NYISO: 3 | |
| Parsimony rule (M5c) | one-sided paired t on time-block mean loss differences, α = 0.10 | D14 |
| Deployment LCB (M7, M7b) | one-sided t lower bound on validation block skill, α = 0.10 | |

## 4. Execution order and outputs

Run from `D:\\REFUSED5` with `py -3.10 -u <script>`.

| # | Script | Output CSV (results/refused6/) | Log |
|---|---|---|---|
| 1 | src/refused6/m1_canonical.py | frozen_v1/M1_runs_india.csv (+ bootstrap) | results/M1*.log |
| 2 | src/refused6/m1_integrity.py | integrity report (14/14 checks) | |
| 3 | src/refused6/m1_audit.py | audit tables A–F | |
| 4 | src/refused6/m1b_stable_gates.py | M1b_runs_india.csv | |
| 5 | src/refused6/m1b_estimator_validation.py | M1b_estimator_validation.csv | |
| 6 | src/refused6/m1c_rescaled_corrector.py | M1c_rescaled.csv | |
| 7 | src/refused6/m1d_clip_robustness.py | M1d_clip_robustness.csv | |
| 8 | src/refused6/m3_m5_ladder.py | M3M5_runs_full.csv | |
| 9 | src/refused6/m4_modern_baselines.py | M4_modern_baselines.csv | |
| 10 | src/refused6/m6_resolution_deconfounded.py | M6_resolution_deconfounded.csv | |
| 11 | src/refused6/m5b_selection_fixed.py | M5b_selection_full.csv | |
| 12 | src/refused6/m8_cross_domain.py | M8_cross_domain.csv | |
| 13 | src/refused6/m7_risk_control.py | M7_risk_control.csv | |
| 14 | src/refused6/m5c_parsimony_selection.py AB | M5c_parsimony.csv | results/M5c.log |
| 15 | src/refused6/m7b_hybrid_safety.py | M7b_hybrid.csv | results/M7b.log |
| 16 | src/refused6/m9_neural_gate.py | M9_neural_gate.csv | results/M9.log |
| 17 | src/refused6/m10_rolling_origin.py | M10_rolling_origin.csv, M10b_selection_rolling.csv | results/M10.log |
| 18 | src/refused6/m9b_neural_tuned.py | M9b_neural_tuned.csv | results/M9b.log |
| 19 | src/refused6/final_stats.py | ALL_STATS.md, ALL_STATS.json | results/final_stats.log |

Stages 11–19 were run unattended by `orchestrator.py`, `post_chain.py`, `post_chain2.py`, `post_chain3.py`, `post_chain4.py`
(each retries a failed stage once and logs to `results/refused6/PIPELINE_STATUS.md`).
The frozen M1 outputs are protected by `results/refused6/frozen_v1/MANIFEST.json` (MD5 per file).

## 5. What is and is not deterministic

* HistGradientBoosting with a fixed `random_state` is deterministic on one machine and thread count;
  results can differ in the last digits across thread counts.
* GPU training (M4, M9, M9b) is not bit-reproducible; seed-level numbers may move slightly, seed-averaged
  conclusions should not.
* Every pre-registered verdict is printed by the script itself from its CSV, so rerunning the script
  re-evaluates the verdict instead of copying it.

## 6. Lineage

Defects D1–D17 with discovery, fix and outcome: `docs/08_defect_lineage.md`.
Theory with proofs and novelty status: `docs/09_theory.md`.
Reviewer audits: `docs/10_reviewer_audit.md`.

## 7. Final report build

`docs/12_REFUSED6_final_report.md` is the written report. `docs/REFUSED6_final_report.html` and
`docs/REFUSED6_Final_Report.pdf` concatenate it with `results/refused6/ALL_STATS.md` (Appendix A) and
`docs/08_defect_lineage.md` (Appendix B). They are built with python-markdown and printed by headless Chrome
(`chrome --headless=new --print-to-pdf`).
