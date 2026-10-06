# O3 development protocol — LA-MCAG: latency-aware multi-context adaptive gating (fixed before any O3 training)

Version 1.0 — 15 Sep 2026. Development data only (FY2018-19 … FY2024-25), same targets, information set, splits and
metrics as O2 (`03_o2_dev_protocol.md` v1.2). Proposal objective O3: context-gated fusion (MCAG) and carbon layer
(GCAL), ablations A2–A4 and A6 at matched capacity with null and permuted context.

## 1. Motivation from the data (why a new model)

Indian public power data arrive with measured, source-specific delays (PSP ≈ 10.6 h; NPP d+1 evening; Grid-India price
files 8–11 days) and sources stop or pause. Examples: CEA State RE reports ended 18 Nov 2025; IMD gridded data end
31 Dec 2025; NPP is missing 13 Mar – 31 May 2020; Grid-India lost 90 PSP report dates. A context-fusion model for
operational use must therefore:

* use each source only as published;
* know how stale each source is;
* degrade gracefully when a source disappears.

MCAG in the proposal gates context. LA-MCAG adds two things:

1. **Staleness-aware gates.** Each gate sees the age of its source's latest observation.
2. **Adaptivity ladder with PART.** The gate is fixed, per-regime or per-instance. The level is predicted from
   validation data by the persistence-adjusted refinement test of RE-FUSED (price-of-adaptivity hypothesis).

## 2. Model

Inputs per issue day τ are the lag-aligned 14-day windows of `o2_03_nf.build_df`, split into blocks:

| Block | Columns (where present for the target) |
|---|---|
| target | normalised target, availability mask |
| market (M) | DAM ACP, DSM normal rate, RTM ACP (T5), price mask |
| supply – conventional (G) | NPP actual, programme and hydro generation; NPP mask |
| supply – renewables (R) | CEA RE total, wind, solar; RE mask (the source that stopped on 18 Nov 2025) |
| system state and reliability (S) | energy met, max demand, shortage, drawal schedule, OD/UD, actual drawal, regional and national energy met, regional wind/solar/hydro (PSP), frequency stress, outage share, coal-stock days; PSP mask |
| environment (W) | Tmax, Tmin, rain, CDD24; weather mask |
| carbon (C) | carbon intensity of State generation (`co2_ci_conv_op`) |

**Staleness s_k:** log(1 + days since the block's key column was last observed, as of the window end), computed from
the full history.

**Architecture:**

* Target encoder: 2-layer BiLSTM (hidden 64) over the target block, concatenated with the series embedding (8) and
  target-day calendar (12), then MLP to z (128).
* Base quantile head: Q_base = Linear(z).
* Context encoder k: GRU (hidden 32) over block k's window, concatenated with s_k to give c_k.
* Correction head k: r_k = MLP([z, c_k]) with dimension H × 7.
* Output: Q = sort(Q_base + Σ_k g_k · r_k).

**Gate levels (the adaptivity ladder):**

| Variant | Gate g_k ∈ (0, 1.5) |
|---|---|
| `mcag_fixed` | 1.5·σ(θ_k), one parameter per block |
| `mcag_regime` | 1.5·σ(θ_{k,ρ}), ρ = season (4) × ex-ante price-volatility state (2: trailing 28-day coefficient of variation of the published bid-area DAM price above or below the series' train median) |
| `mcag_instance` | 1.5·σ(MLP_k([z, c_k, s_k])), the proposal's MCAG made staleness-aware |
| `fusion_fixed` | no gates: Q = Linear-MLP([z, c_1 … c_K]) with hidden width set so the parameter count matches `mcag_instance` within 5 % (A2 comparator) |

**Training:**

* Pinball loss (7 quantiles × 3 horizons, masked).
* Adam 1e-3, batch 512, ≤ 60 epochs, early stopping (patience 6) on the last 180 train days, as for the BiLSTM.
* **Source dropout:** during training each context block is independently replaced by "missing" with probability
  0.15 (values 0, mask 1, staleness set to 30 days), identically for all variants.
* Seeds: 5 for the four main variants; 3 for ablation variants.

## 3. Ablations and tests (development test FY2024-25, rolling-180 conformal as in O2)

| ID | Comparison | Targets | Test |
|---|---|---|---|
| A2 | `mcag_instance` vs `fusion_fixed` (matched capacity) | T1–T5 | pooled scaled pinball and MASE differential, DM-HLN + 7-day block bootstrap, one-sided |
| A3 | real context vs null context (all blocks missing) vs permuted context (block windows permuted across issue days within series) | T1, T2, T5 | same |
| A4 | market-only vs all blocks | T1, T2, T5 | same |
| A6 | all blocks vs no carbon block | T1, T2, T3 (T5 has no carbon block) | same |
| L1 | robustness to source loss: at test time the R and W blocks are set missing for all dev-test days (staleness growing from the start of the development test); degradation of `mcag_instance` vs `fusion_fixed` | T1, T2 | difference in MASE increase |
| F5 | price of adaptivity: gains fixed → regime → instance on dev test vs PART predictions from validation | T1–T5 | sign agreement and rank correlation across target × H cells |

**PART operationalisation (validation only).** From `mcag_fixed` (seed-averaged, median level, normalised units):

* B = base median, r = Σ_k g_k r_k, R = z − B.
* `part2_partition` (coarse = constant, fine = season × volatility state, interleaved weekly blocks K = 6) predicts
  the regime-gate gain.
* `part2_instance` (features = staleness vector, block encodings summary [mean of c_k], calendar) predicts the
  instance-gate gain.
* Predicted refinement iff G > 0.

**Online time-adaptive gate (OTG, top rung of the ladder; v1.2).** For every gated or fusion variant, the context
correction is rescaled per series × H as Q = Q_base + ĝ(τ)·(Q − Q_base), with
ĝ(τ) = clip(Σ R·r / Σ r², 0, 1.5) over the 180 most recent days of median residuals published by τ
(target date ≤ τ − Ly), where R = y − base median and r = model median − base median. If fewer than 30 residuals
are available, ĝ = 1. This is the time axis of the adaptivity ladder (RE-FUSED `ewls`/prequential gates) and uses no
unpublished data. OTG is compared with its non-OTG counterpart (F5-time) and applied identically to `fusion_fixed`.

Holm correction within F2 (O3) and within F5. O2's selected model is reported alongside for context; LA-MCAG is not
tuned on the development test.

## Change log

| Date | Change | Reason |
|---|---|---|
| 15 Sep 2026 | v1.0 | before any O3 training |
| 15 Sep 2026 | v1.1: supply block split into G (NPP) and R (CEA RE); regime = season × ex-ante price-volatility state instead of season × DSM regime; A6 on T1–T3 | found while writing the data code, before any O3 training: DSM regimes R2/R3 do not occur in the training years (a per-DSM-regime gate could not be learned); the L1 source-loss test needs the RE source as its own block; the bid-area target T5 has no carbon block |
| 15 Sep 2026 | v1.2: online time-adaptive gate (OTG) added as the time rung of the adaptivity ladder; base-head quantiles saved for all variants; window-level normalisation (as O2 v1.3) | O3 smoke test on T5 (one seed): context corrections learned on FY2018-23 cut validation errors (e.g. Sep 2023 MAE 1,234 vs 2,645 without context) but became harmful from Feb 2024, when regional solar growth decoupled demand from price (development-test MAE 1,397 vs 832 for the seasonal naive; base head alone ≈ PatchTST). Development observation, logged before the main O3 runs; reserved period untouched |
