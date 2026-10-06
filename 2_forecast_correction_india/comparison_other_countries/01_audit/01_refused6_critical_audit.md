# RE-FUSED / Stage 0 — critical audit of RE-FUSED-6

Date: 13 Sep 2026. Input: frozen snapshot `90_prior_refused6/` (130 files, MD5 manifest, git commit fdaca1c).
Rule for this audit: every claim is re-derived from the CSVs, not from the prose; a claim is *kept*, *weakened*,
*needs re-test* or *withdrawn*; every weakness gets a concrete RE-FUSED action (A-numbers, used in 02_design).

Severity: **F** fatal · **Ma** major · **Mo** moderate · **Mi** minor.

---

## 1. Statistical inference (highest-risk area)

| # | Finding (measured) | Sev | Action |
|---|---|---|---|
| S1 | **Pseudo-replication.** Seeds share the same data and test period, yet were treated as independent units. M1 per-series vs per-instance: run level n=150, p=2.3e-77; config level n=15, p=3.4e-9 (sign test 6.1e-5). Between-config variance is 3.3× within-config (seed) variance. Seeds measure model randomness, not sampling variability. Every "p=1e-60"-style number in RE-FUSED-6 overstates evidence by tens of orders of magnitude. | **Ma** | A1: inference at the level of *independent evidence units* = (dataset × test period); within a unit, loss differentials are tested with a **moving-block bootstrap / HAC Diebold–Mariano** over time; seeds are averaged first. Headline claims are summarised by a random-effects meta-analysis across units. |
| S2 | Configs within a dataset (horizons 1,2,3,7 on the same series) are strongly dependent; counting "15/15 configs" as 15 independent wins is still optimistic. | Ma | A1 (same); report per-dataset effects and count datasets, not configs, as replicates. |
| S3 | Multiplicity not controlled across stages (≈ 40 tests). Holm used only in M1 audit and M8. | Ma | A2: pre-declared claim families (C1–C6), Holm within family, BY (FDR) across the paper; one table listing every inferential claim with its family. |
| S4 | Several ALL_STATS tables print p-values as `0.0` (float rounding). | Mi | A3: scientific notation everywhere; generator unit-tested. |
| S5 | ALL_STATS §M8 prints the raw descriptor correlations (resid_R2 ρ=−0.88) that D15 showed to be a scale artefact; the corrected analysis lives only in prose. | Mo | A3: generate corrected (relative-gap, partial, Holm) numbers from code. |
| S6 | Effect sizes are skill differences with no uncertainty interval at the unit level. | Mo | A1: block-bootstrap 95% CI for every headline difference. |

## 2. Experimental design and confounds

| # | Finding | Sev | Action |
|---|---|---|---|
| E1 | **Validation-only gate fitting is a design choice that drives the result.** The corrector is trained on train; all gates are fitted on the (small) validation window. A reviewer will say fine gates lose *only* because they were starved: with out-of-fold (cross-fitted) corrector residuals on the training period, a fine gate could use 5–10× more data. This is the single most important missing control — and the direct test of the claimed mechanism (estimation cost). | **F** (for the mechanism claim) | A4: **cross-fitted gates**: time-series K-fold out-of-fold corrector predictions on train; fit every rung on OOF residuals (large n) and/or train+validation; compare with validation-only. The principle predicts fine rungs gain most from more data; if they then win, the recommendation becomes data-size dependent — which is exactly the "when" the paper must state. |
| E2 | "Fine" conflates *instance dependence* with *estimator capacity* (a boosted model). | Ma | A5: continuous **complexity axis** on real data: gate models of increasing effective degrees of freedom (depth, leaves, width); measure generalised degrees of freedom (Ye 1998) and relate risk to df/n. |
| E3 | Adaptivity axes studied separately: units (series), states (regimes/instances), time (drift) is treated only as a nuisance. Drift *is* heterogeneity along time; time-local (rolling / forgetting) gates are the natural third axis and are what practitioners use. | Ma | A6: unified ladder over three axes — **unit** (global→series), **state** (series→cell→instance), **time** (static→rolling windows→exponential forgetting / online Fixed-Share). One trade-off: heterogeneity along the axis vs effective sample size along the axis. |
| E4 | **Test-period reuse.** India 2024-25 test period was scored by M1, M1b, M1c, M1d, M5c, M7, M7b, M8, M9, M9b and M10 (11 stages); ETT test splits by M4, M5c, M7b, M8, M9, M9b. Protocol decisions (D8, D9, clip, per-series default) were informed by those results → garden of forking paths. | **Ma** | A7: **confirmatory design**: freeze hypotheses, code and hyperparameters (git tag + SHA-256) on development data, then score once on data never touched: NYISO Jan–Aug 2026 (released after all RE-FUSED-6 work) and datasets never used (NYISO load + ISO forecasts, OPSD/ENTSO-E TSO forecasts, UCI electricity). |
| E5 | One corrector family (HistGradientBoosting, fixed hyperparameters). The price of adaptivity could depend on the corrector. | Ma | A8: corrector ∈ {ridge, LightGBM/HGB, MLP}; plus correcting deep and foundation baselines directly. |
| E6 | Deep baselines only on ETTh1 at h=24 (M4). No foundation model. | Ma | A9: NHITS, PatchTST, DLinear, TiDE/TSMixer (neuralforecast) and Chronos-Bolt (zero-shot) on ≥3 datasets; LightGBM global model; seasonal naive family. |
| E7 | Synthetic generator linear with multiplicative cell amplitudes; real-data phase behaviour (n, heterogeneity, SNR, drift) never mapped with ground truth. | Ma | A10: **semi-synthetic phase diagram**: real features and real residual noise, injected gate heterogeneity (unit/state/time) of known size; vary n_val, τ², SNR, drift rate; compare the observed best rung with the theoretical crossing condition. |
| E8 | Point forecasts only; no uncertainty. | Ma | A11: probabilistic track — quantiles/intervals via conformal residual quantiles with the *same adaptivity ladder* (global / per-series / per-cell / conditional quantile model / rolling); pinball loss, CRPS (quantile approximation), coverage, width, conditional coverage by regime. |
| E9 | No energy-system consequence quantified. | Ma | A12: energy track on public data with defensible valuation: (i) ISO day-ahead **load-forecast correction** (NYISO zonal) → two-settlement **imbalance cost** priced at realised RT−DA LBMP spread; (ii) **reserve dimensioning** from forecast-error quantiles (MW of reserve at a reliability target) with the adaptivity ladder; (iii) **battery arbitrage** scheduled on price forecasts, settled on realised prices. All assumptions stated with sensitivity ranges. |
| E10 | Operator schedule (India) is a dispatch programme, not a forecast; India target (balanced-unit generation) has limited operational meaning. | Mo | A13: keep India as a *mechanism* dataset (long history, drifting reference), not as the energy-impact dataset; energy impact from ISO/TSO forecasts. |
| E11 | Validation-size robustness only in synthetic data. | Mo | A10 + A4 (real subsampling of validation windows and cross-fitting sizes). |
| E12 | Block-LCB benefit under drift is in-sample India only. | Mo | A7/A12: fresh drift test — TSO day-ahead forecasts with documented bias changes (OPSD), NYISO 2026. |
| E13 | Expert-aggregation baselines (Fixed Share, EWA) flagged as mandatory in RE-FUSED-5's novelty map were never run. | Ma | A6 (time axis) includes online aggregation between baseline and corrected forecast. |
| E14 | Neural gate = small MLP on tabular features; no end-to-end joint training. | Mo | A9 variant: joint corrector+gate network on one dataset (limited, reported as such). |
| E15 | Clip [0, 1.5] chosen a priori; binding for fine rungs (40% of instance values clipped). | Mi | keep D11 sweep; report clip rate per rung; theory treats clipping as a constraint set. |

## 3. Claims

| RE-FUSED-6 claim | Re-derived status | Vulnerability | Action |
|---|---|---|---|
| Per-instance gates lose to coarse gates (India, M1) | **kept** (config level 15/15, sign p=6.1e-5) | pseudo-replication; validation-starvation confound (E1) | A1, A4 |
| … across 4 estimators (M1b), rescaling (M1c), caps (M1d) | **kept** | same | A1 |
| … across 6 datasets (M8) | **kept** at config level (11/12); datasets = 6 units | ETT channels are one physical system per file | A1: meta-analysis over units |
| … across 3 test years (M10) | **kept** (19/24 configs) | same data family | A7 fresh periods |
| … against neural gates (M9, M9b) | **kept**, bounded to small MLP | E14 | A9 |
| Best coarse rung is period/dataset dependent | **kept** (descriptive) | none | theory must explain it (plateau) |
| Rung selection from validation does not beat a fixed default (M5c, M10b) | **weakened**: fresh margins 0.001–0.003, not significant in 2023; "does not pay" ≠ "hurts" | small effect; selection rules limited (argmin, parsimony) | A5 GDF-Cp and heterogeneity-test criteria as new, pre-registered selectors |
| Block-LCB is the right deployment rule | **weakened**: gain under drift in-sample only; costs skill when stationary | E12 | A7, A12 |
| Instance-level clipping fails out of sample (M7b) | **kept** (fresh data) | — | keep |
| Correction value falls with baseline strength (M4) | **kept**, one dataset, one horizon | E6 | A9 |
| Finer settlement resolution does not raise correction value (M6) | **kept** (matched 5 zones) | narrow; not central to the principle | move to energy appendix |
| Heterogeneity predicts the fine-rung loss (M8, suggestive) | **not established** (n=12, Holm 0.046 on one of 7) | overfitting risk | A10 turns it into a controlled test |
| Theory Props 1–8 | **correct, elementary, known** | presenting as novel would be fatal | 09_theory: position as specialisations; add GDF-based estimation cost, time-axis tracking trade-off, and a testable crossing condition; prove only what is needed |

## 4. Data

| # | Finding | Sev | Action |
|---|---|---|---|
| D-a | India source = India Data Portal "daily power generation" (derived from CEA daily reports); licence/terms not recorded in RE-FUSED-6 | Ma (for release) | A14: verify licence on the portal; release only derived aggregates if redistribution is allowed, otherwise scripts + source link |
| D-b | NYISO MIS data public; terms of use not recorded | Mo | A14: record NYISO terms; release derived files if permitted |
| D-c | ETT: licence not recorded in RE-FUSED-6 | Mo | A14: record (repository licence) |
| D-d | Price series in the Indian panel unusable (identical across states) — documented | — | keep as data-audit appendix |
| D-e | No fresh, never-touched data | Ma | A7 |
| D-f | Absolute Windows paths hard-coded in every script | Mo | A15: one package, relative paths, config files |

## 5. Reproducibility engineering

| # | Finding | Action |
|---|---|---|
| R1 | Scripts import each other by file path (`importlib` hacks); no package, no tests | A15: `04_code/refused_gate` package with modules data / baselines / correctors / gates / selection / deployment / probabilistic / energy / stats / figures; smoke tests |
| R2 | No environment lock | A16: `13_env/requirements-lock.txt` from `pip freeze`, CUDA/driver versions, OS |
| R3 | No mapping from paper claims to code/results | A17: `CLAIMS_MAP.md` + machine-checked `claims.json` verified against result files before every PDF build |
| R4 | Pre-registration only in an internal markdown file | A18: git-tagged, SHA-256-hashed preregistration committed before confirmatory runs (local; external registration requires an account and is left to the authors) |
| R5 | GPU runs non-deterministic | record; average seeds; report seed spread |

## 6. Theory

| # | Finding | Action |
|---|---|---|
| T1 | Prop 3 variance term asymptotic, partitions only; boosted/neural gates are not partitions | T-A: estimation cost via **effective / generalised degrees of freedom** (Ye 1998; Efron 2004) which covers any gate estimator; measure GDF numerically |
| T2 | No statement unifying unit, state and time adaptivity | T-B: one decomposition — approximation gain = heterogeneity of the oracle gate along the axis (signal-energy weighted); estimation cost ≈ σ²·df/n_eff along the axis; time axis gives the classic tracking trade-off (window length vs drift rate) |
| T3 | No testable *when* condition beyond "Prop 8 says selection can fail" | T-C: crossing condition in measurable quantities (τ², σ², n per cell, df, drift q); pre-registered prediction of the semi-synthetic phase boundary and of real-data winners |
| T4 | Risk of over-claiming | every proposition labelled *known / specialisation / new combination*; no novelty claim for elementary algebra |

## 7. Remaining weaknesses of RE-FUSED-6

* Novelty of mechanism (F, unchanged by experiments) — RE-FUSED must make the contribution a **general, testable principle with a quantitative phase boundary validated on fresh data across axes and domains**.
* Absence of quantified energy-system consequences (F) → A12.
* Pseudo-replication, no uncertainty evaluation, limited baselines → A1, A9, A11.

## 8. What is preserved unchanged

All RE-FUSED-6 results, verdicts and the D1–D17 lineage remain on record in `90_prior_refused6/` and are cited as the
*development-phase evidence*. RE-FUSED re-analyses them with corrected inference (A1) rather than overwriting them.
