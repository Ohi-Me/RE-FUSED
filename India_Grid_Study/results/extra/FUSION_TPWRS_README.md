# Exploratory: settlement-aware fusion scheduler (protocol codes/design/09_fusion_tpwrs_protocol.md)

| Step | Job | Output |
|---|---|---|
| Development run (fit FY2023-24, score FY2024-25) | 34924 | `fusion_tpwrs_dev.json` |
| Protocol frozen, SHA-256 c8ab7b50...a23f3c | git commit 5c5d229 (10 Oct 2026 19:52 IST) | `codes/design/FUSION_TPWRS_SHA256.txt` |
| Frozen run (fit FY2024-25, score 1 Apr 2025 - 29 Aug 2026, already opened) | 34928 | `fusion_tpwrs_confirm.json`, `..._schedules.parquet` |
| Check against a grid-chosen fixed level (added after seeing the frozen result) | 34930 | `fusion_vs_fixed_level.json` (`x08_fusion_vs_fixed_level.py`) |

Variants: F0 one learned level; F1 level from the price ratio; F2 price ratio and band-relative risk; F3 logistic on all
decision-time features; F4 neural network on all features; F5 F4 plus a bias term. Three seeds each; the schedule is
the seed average, projected onto the schedule limits.

## Results on the evaluation period (mean system regret, Rs crore per day; paired 7-day block-bootstrap intervals)

| | Stylised settlement | DSM-closer settlement |
|---|---|---|
| A2 (fixed rule, frozen) | 15.80 | 66.89 |
| Fixed level chosen by grid on the calibration year | 15.49 (level 57 %) | 66.55 (level 54 %) |
| F0 (one level learned by the protocol's procedure) | 15.66 | 66.87 |
| F4 | 15.50 | 66.56 |
| F5 | 15.53 | 66.51 |
| F4 - F0 (protocol's primary test, Holm) | -0.159 [-0.238, -0.092], p_Holm < 0.001 | -0.314 [-0.562, -0.105], p_Holm 0.006 |
| F4 - grid fixed level | +0.008 [-0.036, +0.061] | +0.011 [-0.122, +0.143] |
| F5 - grid fixed level | +0.040 [-0.030, +0.118] | -0.038 [-0.162, +0.067] |

## Reading

* By the frozen protocol, all fusion variants beat F0. But F0 is a fixed level fitted with the same approximations as
  the fusion variants (80 % of the fitting days, limits as a clamp), and it lands at 53 %; a fixed level chosen by grid
  on the whole calibration year with the real projection does as well as the best fusion variant. The protocol's
  comparator was weaker than intended; this was found after the frozen run and is reported as such.
* Against the well-fitted fixed level, no variant (F1-F5, up to a neural network on all decision-time features, trained
  on the settlement cost) lowers mean regret. This extends the paper's negative result from per-group quantile rules to
  continuous, multi-signal rules learned on the settlement itself.
* In the development year no variant beat F0 under the stylised settlement either.
