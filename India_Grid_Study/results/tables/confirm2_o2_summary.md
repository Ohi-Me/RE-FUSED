# O2 round 2 — base models, combinations and the final forecaster

Development test FY2024-25 is reported only; every choice was made on the validation year (protocol `codes/design/06_dev2_protocol.md`). MASE < 1 beats the seasonal naive.

## T1 — energy met (final: **ens_stack**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.8695, ens_top 0.8644, ens_stack 0.8529, ens_online 0.8591

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.943 | 1.028 | 1.022 | 0.768 | 0.307 | 0.774 / 0.878 |
| bitcn | 0.967 | 1.002 | 1.000 | 0.749 | 0.310 | 0.786 / 0.891 |
| chronos | 0.941 | 0.969 | 0.961 | 0.725 | 0.293 | 0.785 / 0.887 |
| chronos2 | 0.870 | 0.918 | 0.928 | 0.686 | 0.281 | 0.781 / 0.886 |
| chronos_base | 0.928 | 0.958 | 0.953 | 0.716 | 0.290 | 0.784 / 0.885 |
| ens_online | 0.859 | 0.901 | 0.914 | 0.673 | 0.273 | 0.790 / 0.893 |
| ens_stack **(final)** | 0.849 | 0.892 | 0.909 | 0.667 | nan | – |
| ens_top | 0.863 | 0.901 | 0.912 | 0.674 | 0.273 | 0.788 / 0.890 |
| lgbm | 0.889 | 0.927 | 0.934 | 0.693 | 0.281 | 0.783 / 0.887 |
| mcag_instance_sd | 0.894 | 0.936 | 0.937 | 0.700 | 0.278 | 0.788 / 0.890 |
| nbeatsx | 0.989 | 1.005 | 1.030 | 0.751 | 0.317 | 0.782 / 0.886 |
| nhits | 0.999 | 1.014 | 1.052 | 0.758 | 0.315 | 0.782 / 0.887 |
| patchtst | 0.982 | 1.021 | 1.018 | 0.763 | 0.317 | 0.790 / 0.896 |
| seasonal_naive | 1.278 | 1.338 | 1.329 | 1.000 | nan | – |
| tft | 0.907 | 0.949 | 0.967 | 0.709 | 0.294 | 0.788 / 0.892 |
| tide | 0.922 | 0.954 | 0.965 | 0.713 | 0.309 | 0.784 / 0.902 |
| xgb | 0.889 | 0.932 | 0.937 | 0.697 | 0.282 | 0.794 / 0.895 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.5415 | [-0.6025, -0.4813] | 1.0e-94 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.4339 | [-0.4875, -0.3844] | 1.9e-83 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.3610 | [-0.4086, -0.3202] | 1.7e-64 |
| vs seasonal_naive (seasonal_naive) | all | -0.4448 | [-0.5000, -0.3955] | 1.8e-87 |
| vs round1_selected (lgbm) | 1 | -0.0242 | [-0.0311, -0.0181] | 2.4e-17 |
| vs round1_selected (lgbm) | 2 | -0.0296 | [-0.0384, -0.0211] | 3.4e-15 |
| vs round1_selected (lgbm) | 3 | -0.0497 | [-0.0631, -0.0342] | 5.6e-13 |
| vs round1_selected (lgbm) | all | -0.0346 | [-0.0429, -0.0258] | 3.8e-21 |
| vs best_single (chronos2) | 1 | -0.0242 | [-0.0298, -0.0187] | 1.6e-19 |
| vs best_single (chronos2) | 2 | -0.0259 | [-0.0328, -0.0189] | 1.0e-16 |
| vs best_single (chronos2) | 3 | -0.0254 | [-0.0339, -0.0173] | 6.6e-11 |
| vs best_single (chronos2) | all | -0.0252 | [-0.0317, -0.0187] | 4.5e-19 |

Seeds of the final forecaster: MASE 0.895–0.899; seeds with p < 0.05 vs seasonal naive: 5/5

## T2 — actual drawal (final: **ens_stack**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.8674, ens_top 0.8564, ens_stack 0.8523, ens_online 0.8556

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.936 | 1.020 | 1.012 | 0.762 | 0.305 | 0.784 / 0.887 |
| bitcn | 0.967 | 1.023 | 1.021 | 0.765 | 0.315 | 0.787 / 0.893 |
| chronos | 0.926 | 0.983 | 0.979 | 0.734 | 0.296 | 0.789 / 0.890 |
| chronos2 | 0.867 | 0.941 | 0.952 | 0.703 | 0.286 | 0.783 / 0.885 |
| chronos_base | 0.920 | 0.977 | 0.973 | 0.730 | 0.293 | 0.791 / 0.891 |
| ens_online | 0.856 | 0.927 | 0.938 | 0.693 | 0.280 | 0.796 / 0.893 |
| ens_stack **(final)** | 0.851 | 0.922 | 0.932 | 0.689 | nan | – |
| ens_top | 0.856 | 0.925 | 0.933 | 0.691 | 0.280 | 0.789 / 0.888 |
| lgbm | 0.882 | 0.953 | 0.951 | 0.712 | 0.288 | 0.790 / 0.889 |
| mcag_instance | 0.884 | 0.967 | 0.963 | 0.722 | 0.286 | 0.790 / 0.892 |
| nbeatsx | 0.953 | 1.015 | 1.043 | 0.758 | 0.319 | 0.788 / 0.890 |
| nhits | 0.955 | 1.016 | 1.028 | 0.759 | 0.314 | 0.790 / 0.892 |
| patchtst | 0.963 | 1.027 | 1.022 | 0.767 | 0.319 | 0.793 / 0.896 |
| seasonal_naive | 1.254 | 1.339 | 1.338 | 1.000 | nan | – |
| tft | 0.913 | 0.979 | 0.989 | 0.731 | 0.302 | 0.793 / 0.895 |
| tide | 0.926 | 0.990 | 0.995 | 0.740 | 0.314 | 0.788 / 0.897 |
| xgb | 0.883 | 0.954 | 0.954 | 0.713 | 0.287 | 0.795 / 0.895 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.5134 | [-0.5716, -0.4596] | 5.1e-103 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.4036 | [-0.4545, -0.3586] | 1.5e-87 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.3343 | [-0.3776, -0.2956] | 8.4e-66 |
| vs seasonal_naive (seasonal_naive) | all | -0.4168 | [-0.4680, -0.3721] | 6.6e-93 |
| vs round1_selected (lgbm) | 1 | -0.0216 | [-0.0277, -0.0159] | 2.2e-16 |
| vs round1_selected (lgbm) | 2 | -0.0312 | [-0.0404, -0.0219] | 2.1e-15 |
| vs round1_selected (lgbm) | 3 | -0.0420 | [-0.0529, -0.0298] | 2.2e-14 |
| vs round1_selected (lgbm) | all | -0.0318 | [-0.0399, -0.0230] | 9.8e-20 |
| vs best_single (chronos2) | 1 | -0.0211 | [-0.0261, -0.0163] | 5.1e-19 |
| vs best_single (chronos2) | 2 | -0.0185 | [-0.0252, -0.0120] | 6.0e-12 |
| vs best_single (chronos2) | 3 | -0.0187 | [-0.0252, -0.0120] | 9.3e-11 |
| vs best_single (chronos2) | all | -0.0195 | [-0.0250, -0.0137] | 5.8e-17 |

Seeds of the final forecaster: MASE 0.923–0.931; seeds with p < 0.05 vs seasonal naive: 5/5

## T3 — conventional generation (final: **ens_top**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.7436, ens_top 0.7369, ens_stack 0.7258, ens_online 0.7276

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.810 | 1.008 | 0.944 | 0.892 | 0.289 | 0.786 / 0.890 |
| bitcn | 1.009 | 1.149 | 1.098 | 1.016 | 0.378 | 0.780 / 0.880 |
| chronos | 0.791 | 0.908 | 0.898 | 0.803 | 0.274 | 0.784 / 0.886 |
| chronos2 | 0.749 | 0.853 | 0.848 | 0.755 | 0.258 | 0.781 / 0.884 |
| chronos_base | 0.788 | 0.906 | 0.896 | 0.802 | 0.274 | 0.781 / 0.885 |
| ens_online | 0.728 | 0.836 | 0.830 | 0.741 | 0.252 | 0.785 / 0.886 |
| ens_stack | 0.723 | 0.831 | 0.827 | 0.736 | 0.251 | 0.791 / 0.892 |
| ens_top **(final)** | 0.743 | 0.859 | 0.850 | 0.761 | nan | – |
| lgbm | 0.744 | 0.862 | 0.852 | 0.763 | 0.259 | 0.793 / 0.892 |
| mcag_instance | 0.755 | 0.865 | 0.848 | 0.766 | 0.259 | 0.783 / 0.884 |
| nbeatsx | 0.799 | 1.161 | 4.178 | 1.025 | 0.463 | 0.788 / 0.891 |
| nhits | 0.816 | 2.040 | 16.233 | 1.794 | 0.820 | 0.792 / 0.892 |
| patchtst | 0.818 | 0.937 | 0.934 | 0.829 | 0.293 | 0.789 / 0.891 |
| seasonal_naive | 0.965 | 1.130 | 1.116 | 1.000 | nan | – |
| tft | 0.769 | 0.888 | 0.882 | 0.786 | 0.280 | 0.787 / 0.889 |
| tide | 0.776 | 0.891 | 0.887 | 0.789 | 0.291 | 0.787 / 0.895 |
| xgb | 0.743 | 0.860 | 0.850 | 0.761 | 0.260 | 0.792 / 0.894 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.3513 | [-0.3966, -0.3074] | 1.4e-72 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.2609 | [-0.2987, -0.2246] | 5.6e-57 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.1987 | [-0.2297, -0.1675] | 6.4e-41 |
| vs seasonal_naive (seasonal_naive) | all | -0.2701 | [-0.3076, -0.2348] | 1.4e-62 |
| vs round1_selected (lgbm) | 1 | -0.0042 | [-0.0072, -0.0017] | 6.4e-04 |
| vs round1_selected (lgbm) | 2 | -0.0046 | [-0.0076, -0.0019] | 7.0e-05 |
| vs round1_selected (lgbm) | 3 | -0.0000 | [-0.0022, +0.0022] | 0.49 |
| vs round1_selected (lgbm) | all | -0.0029 | [-0.0052, -0.0009] | 9.1e-04 |
| vs best_single (xgb) | 1 | -0.0002 | [-0.0028, +0.0028] | 0.43 |
| vs best_single (xgb) | 2 | +0.0012 | [-0.0016, +0.0043] | 0.84 |
| vs best_single (xgb) | 3 | -0.0033 | [-0.0055, -0.0011] | 7.9e-04 |
| vs best_single (xgb) | all | -0.0008 | [-0.0027, +0.0015] | 0.21 |

Seeds of the final forecaster: MASE 0.860–0.863; seeds with p < 0.05 vs seasonal naive: 5/5

## T4 — RE generation (final: **chronos2**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.7725, ens_top 0.7682, ens_stack 0.7692, ens_online 0.7717

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.877 | 1.361 | 1.317 | 0.843 | 0.424 | 0.732 / 0.853 |
| bitcn | 0.931 | 1.304 | 1.338 | 0.807 | 0.425 | 0.809 / 0.908 |
| chronos | 0.815 | 1.164 | 1.255 | 0.721 | 0.375 | 0.783 / 0.885 |
| chronos2 **(final)** | 0.772 | 1.107 | 1.207 | 0.685 | nan | – |
| chronos_base | 0.814 | 1.165 | 1.257 | 0.721 | 0.375 | 0.778 / 0.884 |
| ens_online | 0.772 | 1.115 | 1.210 | 0.690 | 0.358 | 0.780 / 0.879 |
| ens_stack | 0.766 | 1.131 | 1.209 | 0.701 | 0.361 | 0.772 / 0.883 |
| ens_top | 0.768 | 1.141 | 1.210 | 0.706 | 0.363 | 0.765 / 0.880 |
| lgbm | 0.805 | 1.291 | 1.274 | 0.800 | 0.411 | 0.764 / 0.870 |
| mcag_instance | 0.797 | 1.222 | 1.245 | 0.757 | 0.382 | 0.747 / 0.871 |
| nbeatsx | 0.852 | 1.266 | 1.524 | 0.784 | 0.424 | 0.794 / 0.894 |
| nhits | 0.865 | 1.277 | 1.325 | 0.791 | 0.413 | 0.795 / 0.896 |
| patchtst | 0.901 | 1.438 | 1.472 | 0.890 | 0.477 | 0.734 / 0.848 |
| seasonal_naive | 1.025 | 1.615 | 1.685 | 1.000 | nan | – |
| tft | 0.825 | 1.216 | 1.278 | 0.753 | 0.395 | 0.788 / 0.891 |
| tide | 0.838 | 1.342 | 1.296 | 0.831 | 0.445 | 0.783 / 0.908 |
| xgb | 0.813 | 1.309 | 1.284 | 0.810 | 0.409 | 0.765 / 0.875 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.5671 | [-0.7635, -0.4175] | 1.4e-08 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.4974 | [-0.6924, -0.3471] | 5.2e-07 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.4593 | [-0.6592, -0.3087] | 2.9e-06 |
| vs seasonal_naive (seasonal_naive) | all | -0.5080 | [-0.7026, -0.3573] | 3.0e-07 |
| vs round1_selected (chronos) | 1 | -0.0620 | [-0.0876, -0.0426] | 2.3e-09 |
| vs round1_selected (chronos) | 2 | -0.0544 | [-0.0816, -0.0320] | 1.4e-07 |
| vs round1_selected (chronos) | 3 | -0.0557 | [-0.0861, -0.0295] | 1.9e-06 |
| vs round1_selected (chronos) | all | -0.0574 | [-0.0838, -0.0358] | 4.9e-10 |

Seeds of the final forecaster: MASE 1.107–1.107; seeds with p < 0.05 vs seasonal naive: 5/5

## T5 — DAM price (final: **ens_top**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.5826, ens_top 0.6401, ens_stack 0.6118, ens_online 0.6013

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 2.020 | 2.598 | 1.845 | 2.530 | 0.302 | 0.737 / 0.848 |
| bitcn | 0.701 | 0.818 | 0.673 | 0.797 | 0.258 | 0.775 / 0.877 |
| chronos | 0.778 | 0.877 | 0.733 | 0.854 | 0.262 | 0.718 / 0.842 |
| chronos2 | 0.583 | 0.711 | 0.597 | 0.692 | 0.217 | 0.747 / 0.848 |
| chronos_base | 0.784 | 0.884 | 0.743 | 0.860 | 0.265 | 0.720 / 0.837 |
| ens_online | 0.601 | 0.711 | 0.597 | 0.692 | 0.217 | 0.747 / 0.848 |
| ens_stack | 0.607 | 0.717 | 0.608 | 0.698 | 0.217 | 0.768 / 0.864 |
| ens_top **(final)** | 0.621 | 0.741 | 0.616 | 0.722 | nan | – |
| lgbm | 0.937 | 1.123 | 0.888 | 1.094 | 0.268 | 0.740 / 0.849 |
| mcag_instance_sd | 1.105 | 1.406 | 1.088 | 1.369 | 0.249 | 0.774 / 0.879 |
| nbeatsx | 0.813 | 1.148 | 1.437 | 1.118 | 0.353 | 0.741 / 0.838 |
| nhits | 0.773 | 1.041 | 1.020 | 1.013 | 0.320 | 0.753 / 0.848 |
| patchtst | 0.767 | 0.970 | 0.783 | 0.945 | 0.291 | 0.767 / 0.860 |
| seasonal_naive | 0.831 | 1.027 | 0.864 | 1.000 | nan | – |
| tft | 0.862 | 1.012 | 0.870 | 0.986 | 0.304 | 0.788 / 0.878 |
| tide | 0.697 | 0.836 | 0.697 | 0.814 | 0.269 | 0.760 / 0.894 |
| xgb | 1.120 | 1.293 | 0.998 | 1.259 | 0.264 | 0.742 / 0.855 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.3161 | [-0.4267, -0.2010] | 6.9e-14 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.2852 | [-0.3928, -0.1819] | 9.9e-13 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.2548 | [-0.3574, -0.1496] | 1.4e-08 |
| vs seasonal_naive (seasonal_naive) | all | -0.2860 | [-0.3925, -0.1811] | 1.3e-12 |
| vs round1_selected (bilstm) | 1 | -1.9499 | [-2.1758, -1.7328] | 8.2e-104 |
| vs round1_selected (bilstm) | 2 | -1.8553 | [-2.0797, -1.6375] | 3.0e-95 |
| vs round1_selected (bilstm) | 3 | -1.7644 | [-1.9881, -1.5411] | 1.1e-68 |
| vs round1_selected (bilstm) | all | -1.8576 | [-2.0822, -1.6347] | 9.1e-97 |
| vs best_single (chronos2) | 1 | +0.0550 | [+0.0205, +0.1014] | 1 |
| vs best_single (chronos2) | 2 | +0.0244 | [-0.0129, +0.0670] | 0.93 |
| vs best_single (chronos2) | 3 | +0.0131 | [-0.0276, +0.0561] | 0.76 |
| vs best_single (chronos2) | all | +0.0312 | [-0.0055, +0.0717] | 0.98 |

Seeds of the final forecaster: MASE 0.738–0.753; seeds with p < 0.05 vs seasonal naive: 5/5
