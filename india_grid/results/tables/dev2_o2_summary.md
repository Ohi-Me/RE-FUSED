# O2 round 2 — base models, combinations and the final forecaster

Development test FY2024-25 is reported only; every choice was made on the validation year (protocol `codes/design/06_dev2_protocol.md`). MASE < 1 beats the seasonal naive.

## T1 — energy met (final: **ens_stack**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.8233, ens_top 0.8019, ens_stack 0.7999, ens_online 0.8044

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.900 | 1.017 | 1.041 | 0.766 | 0.306 | 0.792 / 0.890 |
| bitcn | 0.912 | 0.994 | 1.035 | 0.748 | 0.312 | 0.787 / 0.887 |
| chronos | 0.881 | 0.977 | 1.013 | 0.735 | 0.301 | 0.795 / 0.893 |
| chronos2 | 0.814 | 0.903 | 0.965 | 0.680 | 0.281 | 0.799 / 0.896 |
| chronos_base | 0.868 | 0.964 | 1.009 | 0.725 | 0.298 | 0.796 / 0.895 |
| ens_online | 0.804 | 0.890 | 0.954 | 0.670 | 0.277 | 0.798 / 0.895 |
| ens_stack **(final)** | 0.797 | 0.884 | 0.948 | 0.665 | 0.275 | 0.800 / 0.896 |
| ens_top | 0.802 | 0.899 | 0.954 | 0.676 | 0.279 | 0.795 / 0.894 |
| lgbm | 0.823 | 0.924 | 0.976 | 0.695 | 0.289 | 0.792 / 0.892 |
| mcag_instance_sd | 0.833 | 0.950 | 0.982 | 0.715 | 0.286 | 0.790 / 0.890 |
| nbeatsx | 2.448 | 1.033 | 1.072 | 0.778 | 0.331 | 0.790 / 0.892 |
| nhits | 1.278 | 1.038 | 1.093 | 0.781 | 0.328 | 0.793 / 0.894 |
| patchtst | 0.917 | 1.019 | 1.066 | 0.767 | 0.324 | 0.795 / 0.894 |
| seasonal_naive | 1.201 | 1.329 | 1.360 | 1.000 | nan | – |
| tft | 0.858 | 0.943 | 1.010 | 0.710 | 0.299 | 0.799 / 0.897 |
| tide | 0.863 | 0.961 | 1.014 | 0.723 | 0.318 | 0.775 / 0.894 |
| xgb | 0.822 | 0.923 | 0.975 | 0.695 | 0.289 | 0.793 / 0.891 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.5307 | [-0.5912, -0.4693] | 3.0e-75 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.4361 | [-0.4829, -0.3769] | 2.5e-63 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.3685 | [-0.4100, -0.3166] | 2.3e-49 |
| vs seasonal_naive (seasonal_naive) | all | -0.4461 | [-0.4952, -0.3887] | 4.6e-67 |
| vs round1_selected (lgbm) | 1 | -0.0303 | [-0.0373, -0.0207] | 1.3e-15 |
| vs round1_selected (lgbm) | 2 | -0.0402 | [-0.0497, -0.0269] | 3.3e-15 |
| vs round1_selected (lgbm) | 3 | -0.0496 | [-0.0617, -0.0339] | 3.0e-14 |
| vs round1_selected (lgbm) | all | -0.0406 | [-0.0494, -0.0279] | 7.2e-19 |
| vs best_single (chronos2) | 1 | -0.0204 | [-0.0273, -0.0149] | 4.3e-11 |
| vs best_single (chronos2) | 2 | -0.0196 | [-0.0281, -0.0129] | 1.4e-08 |
| vs best_single (chronos2) | 3 | -0.0173 | [-0.0265, -0.0100] | 1.2e-05 |
| vs best_single (chronos2) | all | -0.0189 | [-0.0265, -0.0129] | 2.4e-10 |

Seeds of the final forecaster: MASE 0.886–0.889; seeds with p < 0.05 vs seasonal naive: 5/5

## T2 — actual drawal (final: **ens_stack**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.8469, ens_top 0.8365, ens_stack 0.8335, ens_online 0.8374

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.918 | 1.011 | 1.000 | 0.771 | 0.302 | 0.790 / 0.894 |
| bitcn | 0.974 | 1.008 | 1.016 | 0.769 | 0.312 | 0.790 / 0.892 |
| chronos | 0.901 | 0.967 | 0.976 | 0.738 | 0.294 | 0.798 / 0.896 |
| chronos2 | 0.847 | 0.906 | 0.931 | 0.691 | 0.278 | 0.801 / 0.899 |
| chronos_base | 0.892 | 0.962 | 0.975 | 0.734 | 0.293 | 0.796 / 0.896 |
| ens_online | 0.837 | 0.903 | 0.927 | 0.688 | 0.277 | 0.799 / 0.898 |
| ens_stack **(final)** | 0.832 | 0.896 | 0.917 | 0.683 | 0.273 | 0.802 / 0.901 |
| ens_top | 0.834 | 0.900 | 0.919 | 0.686 | 0.275 | 0.799 / 0.898 |
| lgbm | 0.861 | 0.936 | 0.943 | 0.714 | 0.287 | 0.792 / 0.893 |
| mcag_instance | 0.872 | 0.966 | 0.960 | 0.737 | 0.288 | 0.791 / 0.896 |
| nbeatsx | 1.106 | 0.992 | 1.021 | 0.757 | 0.318 | 0.798 / 0.896 |
| nhits | 1.450 | 1.001 | 1.024 | 0.763 | 0.319 | 0.795 / 0.895 |
| patchtst | 0.931 | 1.006 | 1.011 | 0.767 | 0.316 | 0.799 / 0.900 |
| seasonal_naive | 1.228 | 1.311 | 1.305 | 1.000 | nan | – |
| tft | 0.896 | 0.963 | 0.979 | 0.734 | 0.300 | 0.801 / 0.901 |
| tide | 0.902 | 0.969 | 0.984 | 0.739 | 0.311 | 0.788 / 0.898 |
| xgb | 0.863 | 0.936 | 0.943 | 0.714 | 0.286 | 0.793 / 0.893 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.5038 | [-0.5516, -0.4537] | 3.0e-92 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.4032 | [-0.4451, -0.3598] | 5.4e-79 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.3391 | [-0.3779, -0.3000] | 3.3e-60 |
| vs seasonal_naive (seasonal_naive) | all | -0.4152 | [-0.4581, -0.3713] | 1.4e-83 |
| vs round1_selected (lgbm) | 1 | -0.0330 | [-0.0397, -0.0254] | 1.1e-21 |
| vs round1_selected (lgbm) | 2 | -0.0415 | [-0.0497, -0.0319] | 9.2e-23 |
| vs round1_selected (lgbm) | 3 | -0.0450 | [-0.0541, -0.0345] | 5.4e-22 |
| vs round1_selected (lgbm) | all | -0.0400 | [-0.0476, -0.0310] | 2.0e-28 |
| vs best_single (chronos2) | 1 | -0.0097 | [-0.0148, -0.0056] | 2.0e-05 |
| vs best_single (chronos2) | 2 | -0.0127 | [-0.0188, -0.0077] | 9.5e-07 |
| vs best_single (chronos2) | 3 | -0.0089 | [-0.0167, -0.0029] | 0.0029 |
| vs best_single (chronos2) | all | -0.0102 | [-0.0163, -0.0057] | 8.0e-06 |

Seeds of the final forecaster: MASE 0.897–0.899; seeds with p < 0.05 vs seasonal naive: 5/5

## T3 — conventional generation (final: **ens_top**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.8137, ens_top 0.8037, ens_stack 0.9784, ens_online 0.8068

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 0.919 | 0.851 | 0.823 | 0.863 | 0.254 | 0.801 / 0.894 |
| bitcn | 1.178 | 1.034 | 1.018 | 1.049 | 0.340 | 0.773 / 0.874 |
| chronos | 0.880 | 0.807 | 0.805 | 0.819 | 0.247 | 0.786 / 0.888 |
| chronos2 | 0.814 | 0.764 | 0.771 | 0.775 | 0.233 | 0.794 / 0.891 |
| chronos_base | 0.875 | 0.804 | 0.804 | 0.815 | 0.246 | 0.782 / 0.885 |
| ens_online | 0.807 | 0.744 | 0.746 | 0.755 | 0.226 | 0.800 / 0.898 |
| ens_stack | 0.800 | 0.741 | 0.745 | 0.752 | 0.226 | 0.797 / 0.894 |
| ens_top **(final)** | 0.803 | 0.742 | 0.743 | 0.753 | 0.226 | 0.799 / 0.895 |
| lgbm | 0.830 | 0.764 | 0.755 | 0.775 | 0.230 | 0.801 / 0.897 |
| mcag_instance | 0.849 | 0.782 | 0.765 | 0.794 | 0.233 | 0.790 / 0.891 |
| nbeatsx | 5.958 | 0.841 | 0.867 | 0.853 | 0.269 | 0.794 / 0.892 |
| nhits | 3.636 | 0.841 | 0.931 | 0.853 | 0.266 | 0.795 / 0.895 |
| patchtst | 0.905 | 0.833 | 0.837 | 0.845 | 0.263 | 0.792 / 0.893 |
| seasonal_naive | 1.083 | 0.986 | 0.979 | 1.000 | nan | – |
| tft | 0.857 | 0.790 | 0.791 | 0.802 | 0.248 | 0.797 / 0.892 |
| tide | 0.866 | 0.798 | 0.802 | 0.809 | 0.262 | 0.787 / 0.892 |
| xgb | 0.829 | 0.762 | 0.754 | 0.773 | 0.230 | 0.800 / 0.897 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.3048 | [-0.3447, -0.2678] | 2.0e-56 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.2355 | [-0.2696, -0.2014] | 8.1e-44 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.1905 | [-0.2194, -0.1607] | 2.4e-33 |
| vs seasonal_naive (seasonal_naive) | all | -0.2432 | [-0.2766, -0.2107] | 1.9e-48 |
| vs round1_selected (lgbm) | 1 | -0.0164 | [-0.0238, -0.0090] | 4.1e-07 |
| vs round1_selected (lgbm) | 2 | -0.0215 | [-0.0293, -0.0136] | 6.8e-11 |
| vs round1_selected (lgbm) | 3 | -0.0271 | [-0.0371, -0.0170] | 6.2e-10 |
| vs round1_selected (lgbm) | all | -0.0217 | [-0.0293, -0.0133] | 8.4e-12 |
| vs best_single (chronos2) | 1 | -0.0258 | [-0.0324, -0.0191] | 3.0e-17 |
| vs best_single (chronos2) | 2 | -0.0198 | [-0.0271, -0.0126] | 4.4e-10 |
| vs best_single (chronos2) | 3 | -0.0197 | [-0.0299, -0.0103] | 1.4e-06 |
| vs best_single (chronos2) | all | -0.0217 | [-0.0297, -0.0142] | 1.2e-12 |

Seeds of the final forecaster: MASE 0.743–0.744; seeds with p < 0.05 vs seasonal naive: 5/5

## T4 — RE generation (final: **chronos2**, calibration window 365 days)

Cross-fitted validation MASE: best_single 0.9157, ens_top 0.9676, ens_stack 0.9167, ens_online 0.9214

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 1.313 | 0.962 | 0.780 | 0.880 | 0.304 | 0.779 / 0.875 |
| bitcn | 1.275 | 0.998 | 0.848 | 0.913 | 0.322 | 0.780 / 0.876 |
| chronos | 1.055 | 0.875 | 0.751 | 0.800 | 0.268 | 0.795 / 0.892 |
| chronos2 **(final)** | 0.918 | 0.823 | 0.708 | 0.753 | 0.253 | 0.794 / 0.897 |
| chronos_base | 1.048 | 0.872 | 0.753 | 0.798 | 0.268 | 0.794 / 0.893 |
| ens_online | 0.923 | 0.824 | 0.710 | 0.753 | 0.253 | 0.794 / 0.895 |
| ens_stack | 0.918 | 0.823 | 0.708 | 0.753 | 0.253 | 0.794 / 0.897 |
| ens_top | 0.970 | 0.835 | 0.720 | 0.764 | 0.257 | 0.796 / 0.894 |
| lgbm | 1.274 | 0.864 | 0.736 | 0.790 | 0.268 | 0.794 / 0.894 |
| mcag_instance | 1.283 | 0.855 | 0.721 | 0.782 | 0.271 | 0.797 / 0.889 |
| nbeatsx | 1.561 | 0.947 | 0.805 | 0.866 | 0.303 | 0.794 / 0.893 |
| nhits | 1.620 | 0.943 | 0.797 | 0.862 | 0.298 | 0.791 / 0.891 |
| patchtst | 1.450 | 0.956 | 0.828 | 0.874 | 0.314 | 0.789 / 0.889 |
| seasonal_naive | 1.386 | 1.093 | 0.952 | 1.000 | nan | – |
| tft | 1.267 | 0.896 | 0.770 | 0.819 | 0.283 | 0.789 / 0.891 |
| tide | 1.210 | 0.907 | 0.772 | 0.829 | 0.292 | 0.783 / 0.883 |
| xgb | 1.269 | 0.864 | 0.736 | 0.790 | 0.269 | 0.793 / 0.894 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.3083 | [-0.3549, -0.2598] | 7.0e-47 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.2647 | [-0.3082, -0.2221] | 1.8e-41 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.2399 | [-0.2782, -0.2008] | 9.4e-35 |
| vs seasonal_naive (seasonal_naive) | all | -0.2701 | [-0.3119, -0.2301] | 1.0e-43 |
| vs round1_selected (chronos) | 1 | -0.0548 | [-0.0648, -0.0393] | 8.1e-15 |
| vs round1_selected (chronos) | 2 | -0.0484 | [-0.0596, -0.0316] | 1.5e-10 |
| vs round1_selected (chronos) | 3 | -0.0512 | [-0.0638, -0.0326] | 4.1e-10 |
| vs round1_selected (chronos) | all | -0.0513 | [-0.0620, -0.0354] | 6.4e-14 |

Seeds of the final forecaster: MASE 0.823–0.823; seeds with p < 0.05 vs seasonal naive: 5/5

## T5 — DAM price (final: **ens_top**, calibration window 365 days)

Cross-fitted validation MASE: best_single 1.2024, ens_top 1.1178, ens_stack 1.1304, ens_online 1.1917

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 1.397 | 1.264 | 0.956 | 1.291 | 0.276 | 0.786 / 0.890 |
| bitcn | 1.303 | 0.873 | 0.697 | 0.892 | 0.289 | 0.772 / 0.844 |
| chronos | 1.641 | 0.914 | 0.710 | 0.933 | 0.274 | 0.760 / 0.875 |
| chronos2 | 1.202 | 0.684 | 0.553 | 0.699 | 0.213 | 0.773 / 0.878 |
| chronos_base | 1.754 | 0.920 | 0.737 | 0.940 | 0.279 | 0.768 / 0.870 |
| ens_online | 1.192 | 0.684 | 0.552 | 0.699 | 0.211 | 0.768 / 0.873 |
| ens_stack | 1.116 | 0.777 | 0.625 | 0.794 | 0.225 | 0.788 / 0.887 |
| ens_top **(final)** | 1.118 | 0.818 | 0.652 | 0.836 | 0.233 | 0.787 / 0.883 |
| lgbm | 1.544 | 1.499 | 1.113 | 1.531 | 0.254 | 0.847 / 0.923 |
| mcag_instance_sd | 1.309 | 1.329 | 1.019 | 1.358 | 0.259 | 0.807 / 0.894 |
| nbeatsx | 1.463 | 0.927 | 0.760 | 0.947 | 0.285 | 0.831 / 0.920 |
| nhits | 1.466 | 0.931 | 0.767 | 0.951 | 0.291 | 0.818 / 0.910 |
| patchtst | 1.824 | 0.902 | 0.721 | 0.921 | 0.286 | 0.806 / 0.893 |
| seasonal_naive | 1.909 | 0.979 | 0.775 | 1.000 | nan | – |
| tft | 1.630 | 0.934 | 0.770 | 0.954 | 0.278 | 0.821 / 0.917 |
| tide | 1.482 | 0.842 | 0.674 | 0.860 | 0.270 | 0.810 / 0.891 |
| xgb | 1.532 | 1.301 | 0.987 | 1.328 | 0.252 | 0.849 / 0.932 |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | -0.1902 | [-0.3301, -0.0500] | 4.1e-05 |
| vs seasonal_naive (seasonal_naive) | 2 | -0.1611 | [-0.2923, -0.0256] | 4.2e-04 |
| vs seasonal_naive (seasonal_naive) | 3 | -0.1316 | [-0.2597, -0.0022] | 0.0076 |
| vs seasonal_naive (seasonal_naive) | all | -0.1603 | [-0.2892, -0.0271] | 3.4e-04 |
| vs round1_selected (bilstm) | 1 | -0.4970 | [-0.6095, -0.3805] | 2.5e-32 |
| vs round1_selected (bilstm) | 2 | -0.4282 | [-0.5375, -0.3099] | 6.2e-26 |
| vs round1_selected (bilstm) | 3 | -0.4111 | [-0.5261, -0.2900] | 3.6e-18 |
| vs round1_selected (bilstm) | all | -0.4471 | [-0.5605, -0.3304] | 1.0e-28 |
| vs best_single (chronos2) | 1 | +0.1540 | [+0.0919, +0.2416] | 1 |
| vs best_single (chronos2) | 2 | +0.1348 | [+0.0709, +0.2220] | 1 |
| vs best_single (chronos2) | 3 | +0.1123 | [+0.0486, +0.1881] | 1 |
| vs best_single (chronos2) | all | +0.1328 | [+0.0700, +0.2110] | 1 |

Seeds of the final forecaster: MASE 0.795–0.859; seeds with p < 0.05 vs seasonal naive: 5/5
