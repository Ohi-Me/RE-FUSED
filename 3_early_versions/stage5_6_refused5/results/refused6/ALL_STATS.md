# RE-FUSED-6 — consolidated statistics
Generated 2026-09-13T08:11:14 from the result files in `results/refused6/`. Nothing here is typed by hand.

## Stage completion

| file | present | rows |
|---|---|---|
| frozen_v1/M1_runs_india.csv | True | 1200 |
| M1b_runs_india.csv | True | 1500 |
| M1b_estimator_validation.csv | True | 192 |
| M1c_rescaled.csv | True | 600 |
| M1d_clip_robustness.csv | True | 400 |
| M3M5_runs_full.csv | True | 370 |
| M5b_selection_full.csv | True | 370 |
| M4_modern_baselines.csv | True | 120 |
| M6_resolution_deconfounded.csv | True | 420 |
| M8_cross_domain.csv | True | 60 |
| M7_risk_control.csv | True | 270 |
| M5c_parsimony.csv | True | 280 |
| M7b_hybrid.csv | True | 350 |
| M9_neural_gate.csv | True | 300 |
| M10_rolling_origin.csv | True | 720 |
| M10b_selection_rolling.csv | True | 120 |
| M9b_neural_tuned.csv | True | 100 |

## M1 - canonical India run

Rows: 1200 (15 configs x 10 seeds x 8 rungs). Integrity suite: 14/14 passed (frozen_v1).

Mean squared-loss skill by rung:

| rung | skill |
|---|---|
| rung0_none | 0.0 |
| rung1_global | 0.0693 |
| rung2_series | 0.0991 |
| rung3_cell | 0.0944 |
| rung4_shrunk | 0.0862 |
| rung5_instance | 0.0413 |

Best rung per config: {'rung2_series': 10, 'rung0_none': 4, 'rung3_cell': 1}

Oracle capture by best feasible rung: 0.223

## M1b - stable gate estimators

| rung | skill_sq_macro |
|---|---|
| rung2_series | 0.0991 |
| rung3_cell | 0.0944 |
| rung4_wls_series | 0.0764 |
| rung4_wls_cell | 0.0758 |
| rung1_global | 0.0693 |
| rung5_ridge | 0.0615 |
| rung5_wls_bagged | 0.046 |
| rung5_raw_ratio | 0.0413 |
| rung5_wls | 0.0405 |
| rung0_none | 0.0 |

Fine estimators versus per-series (paired, unit = config x seed):

| fine_estimator | minus_per_series | wins | p |
|---|---|---|---|
| rung5_wls | -0.0585 | 0/150 | 1.0e-60 |
| rung5_wls_bagged | -0.0531 | 0/150 | 3.4e-64 |
| rung5_ridge | -0.0375 | 1/150 | 1.8e-50 |
| rung5_raw_ratio | -0.0578 | 0/150 | 2.3e-77 |

Estimator validation against the known closed-form gate:

| estimator | mse_vs_gstar | corr_vs_gstar | risk | absmax |
|---|---|---|---|---|
| raw_ratio | 0.2089 | 0.2814 | 0.8525 | 139996576.7028 |
| ridge_ratio | 0.1647 | 0.3256 | 0.8296 | 1028.8141 |
| wls | 0.1436 | 0.3238 | 0.8147 | 2.0692 |
| wls_bagged | 0.1328 | 0.3245 | 0.8044 | 1.9487 |

## M1c / M1d - reparameterisation and cap robustness

D9 (absorb global gate into corrector):

| rung | rescaled_minus_original | improves | p |
|---|---|---|---|
| rung2_series | 0.002 | 43/60 | 2.7e-09 |
| rung3_cell | 0.0039 | 51/60 | 1.2e-11 |
| rung4_shrunk | 0.0087 | 47/60 | 6.8e-10 |
| rung5_wls_bagged | 0.0207 | 49/60 | 9.0e-12 |

D11 (admissible cap sweep) - skill by cap:

| cap | rung1_global | rung2_series | rung3_cell | rung4_shrunk | rung5_wls_bagged |
|---|---|---|---|---|---|
| 1.0 | 0.0712 | 0.1008 | 0.0996 | 0.0953 | 0.0869 |
| 1.5 | 0.0712 | 0.0942 | 0.0915 | 0.0848 | 0.0676 |
| 3.0 | 0.0712 | 0.0914 | 0.0845 | 0.073 | 0.029 |
| inf | 0.0712 | 0.0914 | 0.0842 | 0.0728 | 0.0235 |

## M3 / M5 - synthetic mechanism and original selectors

Runs: 370 (37 configs x 10 seeds). Unimodality rate: 0.824. Mean validation-test rank correlation: 0.126

Mean regret versus oracle rung:

| index | regret |
|---|---|
| regret_fixed_rung3_cell | 0.0035 |
| regret_rho_rule | 0.0037 |
| regret_fixed_rung4_shrunk | 0.005 |
| regret_fixed_rung2_series | 0.0078 |
| regret_fixed_rung1_global | 0.0082 |
| regret_srm_c1.0 | 0.0306 |
| regret_srm_c0.5 | 0.0371 |
| regret_srm_c0.0 | 0.0381 |
| regret_val_argmin | 0.0383 |
| regret_fixed_rung5_instance | 0.0383 |
| regret_fixed_rung0_none | 0.1747 |

Oracle rung distribution: {'rung1_global': 124, 'rung3_cell': 117, 'rung4_shrunk': 76, 'rung2_series': 39, 'rung0_none': 12, 'rung5_raw_ratio': 2}

## M5b - corrected selectors (nested split, Mallows penalty)

Runs: 370

Mean regret versus oracle rung (nested split and Mallows penalty are the corrected selectors):

| index | regret |
|---|---|
| regret_fixed_rung3_cell | 0.0027 |
| regret_nested | 0.0031 |
| regret_fixed_rung4_shrunk | 0.0047 |
| regret_fixed_rung2_series | 0.0078 |
| regret_fixed_rung1_global | 0.0083 |
| regret_mallows2 | 0.0287 |
| regret_mallows1 | 0.0373 |
| regret_val_argmin | 0.0384 |
| regret_fixed_rung5_instance | 0.0384 |
| regret_srm_sqrt | 0.0384 |
| regret_fixed_rung0_none | 0.1747 |

Exact oracle-rung recovery:

| index | accuracy |
|---|---|
| correct_val_argmin | 0.0 |
| correct_nested | 0.5027 |
| correct_mallows1 | 0.0243 |
| correct_mallows2 | 0.0541 |
| correct_srm_sqrt | 0.0 |

Regret by validation size:

| n_val | regret_nested | regret_mallows1 | regret_mallows2 | regret_val_argmin | regret_fixed_rung3_cell | regret_fixed_rung2_series |
|---|---|---|---|---|---|---|
| 300.0 | 0.0097 | 0.0464 | 0.0464 | 0.0464 | 0.0008 | 0.0044 |
| 600.0 | 0.0043 | 0.0626 | 0.0578 | 0.0626 | 0.0 | 0.0088 |
| 1500.0 | 0.001 | 0.049 | 0.04 | 0.049 | 0.0065 | 0.0029 |
| 3000.0 | 0.0024 | 0.0364 | 0.0264 | 0.0376 | 0.003 | 0.0083 |
| 8000.0 | 0.001 | 0.0254 | 0.0121 | 0.0254 | 0.0008 | 0.0083 |
| 20000.0 | 0.0006 | 0.0 | 0.0001 | 0.0082 | 0.0017 | 0.011 |

## M5c - pre-registered parsimony selection (synthetic + fresh real data)


**A_synthetic** - 185 runs

| index | mean_regret |
|---|---|
| regret_fixed_cell | 0.0029 |
| regret_nested_argmin | 0.0031 |
| regret_fixed_global | 0.0081 |
| regret_fixed_series | 0.0082 |
| regret_nested_parsimony | 0.0117 |
| regret_fixed_shrunk | 0.0172 |
| regret_fixed_instance | 0.0392 |
| regret_fixed_none | 0.1763 |
| comparison | mean | first_lower | wilcoxon_p |
|---|---|---|---|
| nested_parsimony - nested_argmin | 0.0087 | 36/185 | 0.0002 |
| nested_parsimony - fixed_series | 0.0035 | 120/185 | 0.0249 |
| nested_parsimony - fixed_cell | 0.0088 | 63/185 | 0.0022 |

Tail: nested_argmin p95=0.0106 max=0.0651, nested_parsimony p95=0.0682 max=0.1893, fixed_series p95=0.0331 max=0.0797

**B_real** - 95 runs

| index | mean_regret |
|---|---|
| regret_fixed_series | 0.002 |
| regret_fixed_cell | 0.0026 |
| regret_nested_argmin | 0.0158 |
| regret_nested_parsimony | 0.0223 |
| regret_fixed_shrunk | 0.0227 |
| regret_fixed_global | 0.033 |
| regret_fixed_instance | 0.0411 |
| regret_fixed_none | 0.1254 |
| comparison | mean | first_lower | wilcoxon_p |
|---|---|---|---|
| nested_parsimony - nested_argmin | 0.0065 | 13/95 | 0.0068 |
| nested_parsimony - fixed_series | 0.0203 | 6/95 | 0.0 |
| nested_parsimony - fixed_cell | 0.0197 | 37/95 | 0.0 |

Tail: nested_argmin p95=0.0736 max=0.0916, nested_parsimony p95=0.0797 max=0.0916, fixed_series p95=0.0204 max=0.0236

## M4 - modern deep baselines

| baseline_model | rung0_none | rung1_global | rung2_series | rung3_cell | rung4_shrunk | rung5_instance |
|---|---|---|---|---|---|---|
| DLinear | 0.0 | -0.062 | -0.0324 | -0.0757 | -0.0597 | -0.0717 |
| NHITS | 0.0 | -0.0843 | 0.009 | -0.0233 | -0.0298 | -0.0533 |
| PatchTST | 0.0 | -0.1074 | -0.0678 | -0.083 | -0.068 | -0.0893 |
| Persistence | 0.0 | -0.0372 | 0.2442 | -0.0391 | 0.1145 | -0.0484 |

Baseline strength versus correction value (per-series rung):

| baseline_model | base_mae | skill_sq_macro | resid_R2 |
|---|---|---|---|
| DLinear | 1.6503 | -0.0324 | 0.0266 |
| NHITS | 1.492 | 0.009 | -0.0026 |
| PatchTST | 1.4624 | -0.0678 | 0.005 |
| Persistence | 2.766 | 0.2442 | 0.5788 |

## M6 - de-confounded resolution study

Matched zones, window and clock horizons across resolutions.

Baseline MAE:

| clock_h | baseline | 15min | 1d | 1h | 5min |
|---|---|---|---|---|---|
| 1 | b_da | 16.6163 | nan | 15.5327 | 16.9981 |
| 1 | b_pers | 13.9231 | nan | 11.9261 | 14.6672 |
| 24 | b_da | 16.6455 | 10.817 | 15.5597 | 17.0282 |
| 24 | b_pers | 22.7833 | 13.9714 | 21.7104 | 23.2541 |

Best coarse-rung skill:

| clock_h | baseline | 15min | 1d | 1h | 5min |
|---|---|---|---|---|---|
| 1 | b_da | 0.0973 | nan | 0.0772 | 0.0807 |
| 1 | b_pers | 0.0394 | nan | 0.0913 | 0.0371 |
| 24 | b_da | -0.0006 | 0.0036 | -0.0001 | -0.0031 |
| 24 | b_pers | 0.1068 | 0.0229 | 0.0947 | 0.1458 |

Out-of-sample residual R2:

| clock_h | baseline | 15min | 1d | 1h | 5min |
|---|---|---|---|---|---|
| 1 | b_da | 0.0929 | nan | 0.068 | 0.0771 |
| 1 | b_pers | 0.0465 | nan | 0.0754 | 0.0549 |
| 24 | b_da | -0.0018 | 0.0028 | -0.0006 | -0.0056 |
| 24 | b_pers | 0.114 | 0.0213 | 0.0793 | 0.1525 |

## M8 - cross-domain ladder and descriptors

| dataset | h | resid_R2_val | resid_acf1 | gate_var_between | gate_var_est | n_val_per_cell | best_coarse_skill | fine_minus_coarse |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 1 | 0.4091 | 0.2947 | 0.223 | 0.2145 | 580.6667 | 0.2087 | -0.0568 |
| ETTh1 | 24 | 0.104 | 0.8177 | 0.1285 | 0.53 | 580.6667 | 0.0553 | -0.0123 |
| ETTh2 | 1 | 0.1452 | -0.172 | 0.0887 | 0.472 | 580.6667 | 0.1463 | -0.0073 |
| ETTh2 | 24 | 0.0269 | 0.8258 | 0.5706 | 1.551 | 580.6667 | 0.0185 | -0.0062 |
| ETTm1 | 1 | 0.1019 | -0.0648 | 0.1609 | 0.223 | 2322.6667 | 0.0262 | -0.0158 |
| ETTm1 | 24 | 0.6436 | 0.9728 | 0.1778 | 0.041 | 2322.6667 | 0.3527 | -0.1466 |
| ETTm2 | 1 | 0.0441 | -0.1316 | 0.1609 | 0.3019 | 2322.6667 | 0.0786 | -0.0041 |
| ETTm2 | 24 | 0.4813 | 0.9236 | 0.1042 | 0.0513 | 2322.6667 | 0.2595 | -0.0803 |
| india_daily | 1 | 0.2109 | 0.0938 | 0.1191 | 0.3254 | 121.6667 | 0.1439 | -0.0104 |
| india_daily | 3 | 0.1849 | 0.6438 | 0.1284 | 0.2736 | 121.6667 | 0.1193 | -0.0333 |
| nyiso_1h | 1 | 0.0266 | -0.3007 | 0.0448 | 0.3663 | 2927.6667 | 0.0834 | -0.0033 |
| nyiso_1h | 24 | 0.0343 | 0.3954 | 0.0282 | 0.2913 | 2927.6667 | 0.1029 | 0.0148 |

Best rung by dataset: {('ETTh1', 1): 'rung3_cell', ('ETTh1', 24): 'rung3_cell', ('ETTh2', 1): 'rung4_shrunk', ('ETTh2', 24): 'rung3_cell', ('ETTm1', 1): 'rung3_cell', ('ETTm1', 24): 'rung3_cell', ('ETTm2', 1): 'rung3_cell', ('ETTm2', 24): 'rung2_series', ('india_daily', 1): 'rung4_shrunk', ('india_daily', 3): 'rung2_series', ('nyiso_1h', 1): 'rung1_global', ('nyiso_1h', 24): 'rung4_shrunk'}

Do validation-only descriptors predict the fine-rung gap? (unit = dataset x horizon)

| descriptor | spearman | p |
|---|---|---|
| resid_R2_val | -0.8811 | 0.0002 |
| gate_var_between | -0.4476 | 0.1446 |
| gate_var_est | 0.6853 | 0.0139 |
| n_val_per_cell | 0.3203 | 0.3102 |
| resid_acf1 | -0.5804 | 0.0479 |
| base_mae_over_sd | -0.1678 | 0.6021 |

## M7 - risk-controlled deployment versus alternatives

| baseline | rule | coverage | degraded_series | mean_skill | withheld | withheld_helpful | worst_series_skill |
|---|---|---|---|---|---|---|---|
| operator_schedule | always | 0.8367 | 6.8 | 0.0032 | 0.0 | 0.0 | -0.4459 |
| operator_schedule | block_lcb | 0.18 | 0.4 | 0.0245 | 16.4 | 6.7333 | -0.0472 |
| operator_schedule | clip90 | 0.8367 | 6.6667 | -0.0015 | 0.0 | 0.0 | -0.4371 |
| operator_schedule | crc_quantile | 0.0033 | 0.0 | 0.0007 | 16.6667 | 9.8667 | 0.0 |
| operator_schedule | never | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| operator_schedule | unc_selective | 0.7959 | 6.7333 | 0.0045 | 0.0 | 0.0 | -0.4283 |
| persistence | always | 0.9933 | 1.6667 | 0.1097 | 0.0 | 0.0 | -0.0202 |
| persistence | block_lcb | 0.7233 | 0.8667 | 0.0929 | 5.5333 | 4.6 | -0.0186 |
| persistence | clip90 | 0.9933 | 0.8 | 0.1139 | 0.0 | 0.0 | -0.0053 |
| persistence | crc_quantile | 0.0467 | 0.0 | 0.0099 | 18.9333 | 17.2667 | 0.0 |
| persistence | never | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| persistence | unc_selective | 0.9776 | 1.6667 | 0.1083 | 0.0 | 0.0 | -0.0211 |
| roll7 | always | 1.0 | 2.6667 | 0.1234 | 0.0 | 0.0 | -0.064 |
| roll7 | block_lcb | 0.6733 | 1.0667 | 0.0941 | 6.5333 | 4.9333 | -0.0345 |
| roll7 | clip90 | 1.0 | 1.6667 | 0.1263 | 0.0 | 0.0 | -0.0295 |
| roll7 | crc_quantile | 0.0767 | 0.0 | 0.0189 | 18.4667 | 15.8 | 0.0 |
| roll7 | never | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| roll7 | unc_selective | 0.9811 | 3.0 | 0.1206 | 0.0 | 0.0 | -0.0628 |

Block-LCB versus each alternative (paired, unit = config x seed):

| rule | block_lcb_minus_rule | lcb_better | p |
|---|---|---|---|
| always | -0.0083 | 15/45 | 3.7e-02 |
| never | 0.0705 | 45/45 | 1.1e-09 |
| crc_quantile | 0.0606 | 45/45 | 2.0e-11 |
| clip90 | -0.0091 | 19/45 | 6.2e-02 |
| unc_selective | -0.0073 | 19/45 | 4.7e-02 |

## M7b - pre-registered hybrid deployment rule


**in_sample_india** - 126 rule-runs

| rule | degraded_series | mean_skill | withheld | worst_series_skill |
|---|---|---|---|---|
| always | 3.0 | 0.0932 | 0.0 | -0.198 |
| block_lcb | 0.4444 | 0.0935 | 8.1111 | -0.028 |
| clip90 | 2.7222 | 0.0918 | 0.0 | -0.1808 |
| clip90_lcb | 0.1667 | 0.0932 | 8.1111 | -0.0215 |
| crc_quantile | 0.0 | 0.0162 | 18.6667 | 0.0 |
| never | 0.0 | 0.0 | 0.0 | 0.0 |
| unc_selective | 3.1111 | 0.0917 | 0.0 | -0.1925 |

**fresh_stationary** - 224 rule-runs

| rule | degraded_series | mean_skill | withheld | worst_series_skill |
|---|---|---|---|---|
| always | 0.3125 | 0.1009 | 0.0 | 0.0153 |
| block_lcb | 0.125 | 0.0968 | 1.875 | 0.0139 |
| clip90 | 0.1562 | 0.0679 | 0.0 | 0.0058 |
| clip90_lcb | 0.0 | 0.0647 | 1.875 | 0.0027 |
| crc_quantile | 0.0 | 0.0389 | 8.4688 | 0.0 |
| never | 0.0 | 0.0 | 0.0 | 0.0 |
| unc_selective | 0.25 | 0.0913 | 0.0 | 0.0121 |

## M9 - neural sigmoid gate (reviewer objection 1.3)

| dataset | baseline | h | global | neural_gate | neural_gate_refit | none | per_series | per_series_V1 |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | persistence | 1 | 0.1597 | 0.1604 | 0.1728 | 0.0 | 0.2067 | 0.2064 |
| ETTh1 | persistence | 24 | 0.0518 | 0.0507 | 0.0433 | 0.0 | 0.0498 | 0.0488 |
| ETTh2 | persistence | 1 | 0.1462 | 0.1455 | 0.1473 | 0.0 | 0.1456 | 0.1437 |
| ETTh2 | persistence | 24 | 0.0139 | 0.0158 | 0.0165 | 0.0 | 0.0181 | 0.0146 |
| india_daily | persistence | 1 | 0.1396 | 0.1415 | 0.1443 | 0.0 | 0.1439 | 0.1396 |
| india_daily | persistence | 3 | 0.0953 | 0.1148 | 0.1188 | 0.0 | 0.1193 | 0.1107 |
| india_daily | roll7 | 1 | 0.1829 | 0.194 | 0.2045 | 0.0 | 0.2078 | 0.1831 |
| india_daily | roll7 | 3 | 0.0408 | 0.0681 | 0.0826 | 0.0 | 0.086 | 0.0586 |
| india_daily | snaive7 | 1 | 0.3168 | 0.3247 | 0.3312 | 0.0 | 0.3333 | 0.3204 |
| india_daily | snaive7 | 3 | 0.0788 | 0.1143 | 0.1281 | 0.0 | 0.1334 | 0.1003 |

| comparison | mean | neural_better | wilcoxon_p |
|---|---|---|---|
| neural_gate - per_series | -0.0114 | 6/50 | 0.0 |
| neural_gate - per_series_V1 | 0.0004 | 41/50 | 0.0005 |
| neural_gate - global | 0.0104 | 41/50 | 0.0 |
| neural_gate - none | 0.133 | 50/50 | 0.0 |
| neural_gate_refit - per_series | -0.0054 | 16/50 | 0.0 |
| neural_gate_refit - per_series_V1 | 0.0063 | 40/50 | 0.0008 |
| neural_gate_refit - global | 0.0164 | 42/50 | 0.0 |
| neural_gate_refit - none | 0.1389 | 50/50 | 0.0 |

| gate | skill_sq_macro | degraded_series | gate_sd |
|---|---|---|---|
| global | 0.1226 | 1.54 | 0.0 |
| neural_gate | 0.133 | 1.62 | 0.2229 |
| neural_gate_refit | 0.1389 | 1.36 | 0.2574 |
| none | 0.0 | 0.0 | 0.0 |
| per_series | 0.1444 | 0.84 | 0.3118 |
| per_series_V1 | 0.1326 | 1.54 | 0.3487 |

**M9b - neural gate with hyperparameters selected on V2, refit on V**

| dataset | baseline | h | neural_gate_tuned | per_series |
|---|---|---|---|---|
| ETTh1 | persistence | 1 | 0.1779 | 0.2067 |
| ETTh1 | persistence | 24 | 0.0436 | 0.0498 |
| ETTh2 | persistence | 1 | 0.1524 | 0.1456 |
| ETTh2 | persistence | 24 | 0.017 | 0.0181 |
| india_daily | persistence | 1 | 0.1441 | 0.1439 |
| india_daily | persistence | 3 | 0.1159 | 0.1193 |
| india_daily | roll7 | 1 | 0.2019 | 0.2078 |
| india_daily | roll7 | 3 | 0.0863 | 0.086 |
| india_daily | snaive7 | 1 | 0.3311 | 0.3333 |
| india_daily | snaive7 | 3 | 0.1273 | 0.1334 |

neural_gate_tuned - per_series: mean -0.0046, tuned better in 16/50, Wilcoxon p=1.19e-03

Chosen configurations (hidden/wd/lr): 128/0.001/0.003 x11, 128/1e-05/0.001 x10, 128/1e-05/0.003 x10, 32/0.001/0.003 x8, 32/1e-05/0.001 x5, 32/0.001/0.001 x2, 128/0.001/0.001 x2, 32/1e-05/0.003 x2

## M10 - rolling-origin India re-evaluation (reviewer objection 2.1)

| origin | baseline | h | rung0_none | rung1_global | rung2_series | rung3_cell | rung4_shrunk | rung5_instance | best |
|---|---|---|---|---|---|---|---|---|---|
| origin_2022 | operator_schedule | 1 | 0.0 | 0.2323 | 0.2685 | 0.2629 | 0.3025 | 0.2998 | rung4_shrunk |
| origin_2022 | operator_schedule | 3 | 0.0 | 0.1597 | 0.1784 | 0.1667 | 0.2027 | 0.2009 | rung4_shrunk |
| origin_2022 | persistence | 1 | 0.0 | 0.1344 | 0.1456 | 0.1375 | 0.1384 | 0.1251 | rung2_series |
| origin_2022 | persistence | 3 | 0.0 | 0.0907 | 0.0909 | 0.0879 | 0.0962 | 0.0906 | rung4_shrunk |
| origin_2022 | roll7 | 1 | 0.0 | 0.2129 | 0.2252 | 0.2258 | 0.2271 | 0.2096 | rung4_shrunk |
| origin_2022 | roll7 | 3 | 0.0 | 0.0816 | 0.0736 | 0.074 | 0.0843 | 0.0785 | rung4_shrunk |
| origin_2022 | snaive7 | 1 | 0.0 | 0.3602 | 0.3695 | 0.3721 | 0.3716 | 0.3554 | rung3_cell |
| origin_2022 | snaive7 | 3 | 0.0 | 0.1227 | 0.1339 | 0.1309 | 0.1296 | 0.1149 | rung2_series |
| origin_2023 | operator_schedule | 1 | 0.0 | -0.0219 | -0.0169 | -0.014 | 0.0038 | 0.0039 | rung5_instance |
| origin_2023 | operator_schedule | 3 | 0.0 | -0.0574 | -0.0694 | -0.0658 | -0.0597 | -0.0599 | rung0_none |
| origin_2023 | persistence | 1 | 0.0 | 0.1408 | 0.1445 | 0.1425 | 0.1456 | 0.1352 | rung4_shrunk |
| origin_2023 | persistence | 3 | 0.0 | 0.1293 | 0.1322 | 0.1287 | 0.1307 | 0.1214 | rung2_series |
| origin_2023 | roll7 | 1 | 0.0 | 0.2525 | 0.2545 | 0.2511 | 0.2548 | 0.2434 | rung4_shrunk |
| origin_2023 | roll7 | 3 | 0.0 | 0.1143 | 0.1223 | 0.1163 | 0.1107 | 0.1011 | rung2_series |
| origin_2023 | snaive7 | 1 | 0.0 | 0.4099 | 0.4068 | 0.4077 | 0.414 | 0.4056 | rung4_shrunk |
| origin_2023 | snaive7 | 3 | 0.0 | 0.1617 | 0.1724 | 0.1685 | 0.1659 | 0.1538 | rung2_series |
| origin_2024 | operator_schedule | 1 | 0.0 | -0.0895 | 0.0042 | 0.0004 | 0.0223 | 0.0131 | rung4_shrunk |
| origin_2024 | operator_schedule | 3 | 0.0 | -0.0521 | -0.0041 | -0.0055 | 0.0031 | 0.0001 | rung4_shrunk |
| origin_2024 | persistence | 1 | 0.0 | 0.1396 | 0.1439 | 0.141 | 0.1408 | 0.1299 | rung2_series |
| origin_2024 | persistence | 3 | 0.0 | 0.0953 | 0.1193 | 0.1172 | 0.1036 | 0.0856 | rung2_series |
| origin_2024 | roll7 | 1 | 0.0 | 0.1829 | 0.2078 | 0.2067 | 0.1906 | 0.1638 | rung2_series |
| origin_2024 | roll7 | 3 | 0.0 | 0.0408 | 0.086 | 0.0837 | 0.067 | 0.0448 | rung2_series |
| origin_2024 | snaive7 | 1 | 0.0 | 0.3168 | 0.3333 | 0.334 | 0.3224 | 0.2981 | rung3_cell |
| origin_2024 | snaive7 | 3 | 0.0 | 0.0788 | 0.1334 | 0.1307 | 0.0975 | 0.0661 | rung2_series |

| origin | configs | instance_best | instance_below_best_coarse | mean_gap | best_counts |
|---|---|---|---|---|---|
| origin_2022 | 8 | 0 | 6 | -0.0027 | rung4_shrunk=5; rung2_series=2; rung3_cell=1 |
| origin_2023 | 8 | 1 | 7 | -0.0075 | rung2_series=3; rung4_shrunk=3; rung0_none=1; rung5_instance=1 |
| origin_2024 | 8 | 0 | 6 | -0.0279 | rung2_series=5; rung4_shrunk=2; rung3_cell=1 |

**M10b - rung selection vs fixed defaults by origin** (origin_2024 overlaps M5c Part B: in-sample)

| origin | nested_argmin | nested_parsimony | fixed_series | fixed_cell | fixed_shrunk | fixed_global | fixed_instance | fixed_none |
|---|---|---|---|---|---|---|---|---|
| origin_2022 | 0.0148 | 0.014 | 0.0117 | 0.0127 | 0.0035 | 0.0244 | 0.0152 | 0.2045 |
| origin_2023 | 0.0084 | 0.0066 | 0.0072 | 0.0078 | 0.0107 | 0.0103 | 0.0209 | 0.1706 |
| origin_2024 | 0.0142 | 0.0237 | 0.002 | 0.0033 | 0.0259 | 0.0421 | 0.0509 | 0.1341 |

| origin | comparison | mean | selector_lower | wilcoxon_p |
|---|---|---|---|---|
| origin_2022 | nested_argmin - fixed_series | 0.0031 | 10/40 | 0.0332 |
| origin_2022 | nested_parsimony - fixed_series | 0.0023 | 4/40 | 0.0015 |
| origin_2023 | nested_argmin - fixed_series | 0.0011 | 11/40 | 0.6143 |
| origin_2023 | nested_parsimony - fixed_series | -0.0007 | 12/40 | 0.9808 |
| origin_2024 | nested_argmin - fixed_series | 0.0122 | 1/40 | 0.0043 |
| origin_2024 | nested_parsimony - fixed_series | 0.0217 | 0/40 | 0.0 |
