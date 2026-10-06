# O3 development results — LA-MCAG (development test FY2024-25)

MASE uses train-period seasonal-naive scales; rel. SN is the scaled MAE ratio to the seasonal naive on the same rows. Intervals: rolling-180 conformal at C1 for every variant. `+otg` = online time-adaptive gate; `:base` = base head without context.

## T1 — energy met

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| null_context | 0.925 | 0.691 | 0.281 | 0.796 | 0.894 |
| market_only | 0.926 | 0.692 | 0.282 | 0.799 | 0.896 |
| null_context+otg | 0.930 | 0.695 | 0.285 | 0.797 | 0.895 |
| permuted_context | 0.931 | 0.696 | 0.283 | 0.794 | 0.895 |
| mcag_regime | 0.931 | 0.696 | 0.277 | 0.788 | 0.890 |
| market_only+otg | 0.932 | 0.697 | 0.286 | 0.796 | 0.893 |
| mcag_regime+otg | 0.933 | 0.697 | 0.284 | 0.785 | 0.885 |
| mcag_fixed | 0.933 | 0.697 | 0.277 | 0.786 | 0.890 |
| fusion_fixed+otg | 0.934 | 0.698 | 0.285 | 0.785 | 0.886 |
| mcag_fixed+otg | 0.934 | 0.699 | 0.284 | 0.782 | 0.883 |
| no_carbon | 0.935 | 0.699 | 0.279 | 0.789 | 0.892 |
| fusion_fixed | 0.935 | 0.699 | 0.279 | 0.791 | 0.891 |
| permuted_context+otg | 0.936 | 0.700 | 0.287 | 0.793 | 0.892 |
| mcag_instance+otg | 0.936 | 0.700 | 0.285 | 0.783 | 0.884 |
| no_carbon+otg | 0.938 | 0.701 | 0.285 | 0.784 | 0.887 |
| mcag_instance | 0.939 | 0.702 | 0.278 | 0.788 | 0.891 |
| market_only:base | 0.997 | 0.746 | 0.312 | 0.772 | 0.875 |
| fusion_fixed:base | 1.015 | 0.759 | 0.317 | 0.769 | 0.874 |
| null_context:base | 1.035 | 0.774 | 0.322 | 0.768 | 0.875 |
| permuted_context:base | 1.040 | 0.778 | 0.325 | 0.766 | 0.872 |
| mcag_regime:base | 1.077 | 0.805 | 0.338 | 0.760 | 0.870 |
| mcag_fixed:base | 1.079 | 0.807 | 0.338 | 0.760 | 0.870 |
| no_carbon:base | 1.081 | 0.808 | 0.338 | 0.761 | 0.870 |
| mcag_instance:base | 1.084 | 0.810 | 0.340 | 0.760 | 0.870 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | +0.0045 | [-0.0004, +0.0088] | 0.99 |
| A2 instance vs fusion | pinball | -0.0002 | [-0.0014, +0.0010] | 0.37 |
| A2+OTG instance vs fusion | mae | +0.0028 | [-0.0017, +0.0069] | 0.94 |
| A2+OTG instance vs fusion | pinball | +0.0006 | [-0.0005, +0.0018] | 0.9 |
| ladder regime vs fixed | mae | -0.0015 | [-0.0030, +0.0001] | 0.0067 |
| ladder regime vs fixed | pinball | -0.0005 | [-0.0008, -0.0002] | 1.9e-04 |
| ladder instance vs regime | mae | +0.0079 | [+0.0041, +0.0110] | 1 |
| ladder instance vs regime | pinball | +0.0015 | [+0.0006, +0.0024] | 1 |
| ladder OTG vs instance | mae | -0.0027 | [-0.0068, +0.0020] | 0.057 |
| ladder OTG vs instance | pinball | +0.0070 | [+0.0055, +0.0088] | 1 |
| context vs base head | mae | -0.1472 | [-0.1757, -0.1223] | 5.9e-44 |
| context vs base head | pinball | -0.0540 | [-0.0631, -0.0465] | 2.7e-56 |
| A3 real vs null | mae | +0.0138 | [-0.0053, +0.0329] | 0.97 |
| A3 real vs null | pinball | -0.0021 | [-0.0061, +0.0021] | 0.095 |
| A3 real vs null+otg | mae | +0.0058 | [-0.0112, +0.0239] | 0.81 |
| A3 real vs null+otg | pinball | +0.0004 | [-0.0033, +0.0042] | 0.61 |
| A3 real vs permuted | mae | +0.0077 | [-0.0104, +0.0266] | 0.86 |
| A3 real vs permuted | pinball | -0.0036 | [-0.0079, +0.0009] | 0.015 |
| A3 real vs permuted+otg | mae | -0.0005 | [-0.0171, +0.0171] | 0.47 |
| A3 real vs permuted+otg | pinball | -0.0019 | [-0.0058, +0.0020] | 0.11 |
| A4 all vs market-only | mae | +0.0125 | [-0.0057, +0.0310] | 0.97 |
| A4 all vs market-only | pinball | -0.0023 | [-0.0062, +0.0017] | 0.066 |
| A4 all vs market-only+otg | mae | +0.0042 | [-0.0120, +0.0210] | 0.75 |
| A4 all vs market-only+otg | pinball | -0.0011 | [-0.0047, +0.0026] | 0.21 |
| A6 all vs no carbon | mae | +0.0037 | [-0.0037, +0.0111] | 0.89 |
| A6 all vs no carbon | pinball | +0.0009 | [-0.0004, +0.0022] | 0.96 |
| A6 all vs no carbon+otg | mae | -0.0017 | [-0.0079, +0.0045] | 0.24 |
| A6 all vs no carbon+otg | pinball | +0.0007 | [-0.0007, +0.0020] | 0.89 |

## T2 — actual drawal

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| mcag_regime | 0.962 | 0.719 | 0.286 | 0.788 | 0.891 |
| no_carbon | 0.963 | 0.720 | 0.286 | 0.788 | 0.892 |
| mcag_fixed | 0.963 | 0.720 | 0.286 | 0.789 | 0.890 |
| null_context | 0.964 | 0.720 | 0.292 | 0.794 | 0.894 |
| fusion_fixed | 0.966 | 0.721 | 0.287 | 0.792 | 0.894 |
| mcag_instance | 0.967 | 0.722 | 0.286 | 0.790 | 0.892 |
| market_only | 0.967 | 0.722 | 0.293 | 0.796 | 0.895 |
| permuted_context | 0.967 | 0.722 | 0.293 | 0.794 | 0.895 |
| no_carbon+otg | 0.968 | 0.723 | 0.293 | 0.789 | 0.891 |
| mcag_regime+otg | 0.969 | 0.724 | 0.293 | 0.787 | 0.888 |
| mcag_fixed+otg | 0.969 | 0.724 | 0.293 | 0.786 | 0.888 |
| fusion_fixed+otg | 0.970 | 0.725 | 0.295 | 0.789 | 0.889 |
| null_context+otg | 0.970 | 0.725 | 0.298 | 0.794 | 0.893 |
| mcag_instance+otg | 0.971 | 0.726 | 0.294 | 0.786 | 0.888 |
| permuted_context+otg | 0.972 | 0.726 | 0.299 | 0.793 | 0.891 |
| market_only+otg | 0.973 | 0.727 | 0.299 | 0.791 | 0.890 |
| market_only:base | 1.015 | 0.758 | 0.314 | 0.774 | 0.879 |
| fusion_fixed:base | 1.019 | 0.762 | 0.316 | 0.773 | 0.877 |
| null_context:base | 1.038 | 0.775 | 0.321 | 0.770 | 0.876 |
| permuted_context:base | 1.041 | 0.777 | 0.322 | 0.769 | 0.875 |
| mcag_regime:base | 1.072 | 0.801 | 0.334 | 0.764 | 0.870 |
| no_carbon:base | 1.073 | 0.801 | 0.334 | 0.764 | 0.870 |
| mcag_fixed:base | 1.073 | 0.801 | 0.334 | 0.763 | 0.870 |
| mcag_instance:base | 1.074 | 0.802 | 0.335 | 0.763 | 0.870 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | +0.0011 | [-0.0033, +0.0048] | 0.74 |
| A2 instance vs fusion | pinball | -0.0009 | [-0.0021, +0.0001] | 0.027 |
| A2+OTG instance vs fusion | mae | +0.0013 | [-0.0028, +0.0048] | 0.79 |
| A2+OTG instance vs fusion | pinball | -0.0011 | [-0.0023, +0.0000] | 0.014 |
| ladder regime vs fixed | mae | -0.0017 | [-0.0028, -0.0006] | 5.6e-05 |
| ladder regime vs fixed | pinball | -0.0003 | [-0.0006, -0.0000] | 0.0045 |
| ladder instance vs regime | mae | +0.0050 | [+0.0026, +0.0071] | 1 |
| ladder instance vs regime | pinball | +0.0002 | [-0.0005, +0.0009] | 0.74 |
| ladder OTG vs instance | mae | +0.0045 | [-0.0003, +0.0100] | 0.99 |
| ladder OTG vs instance | pinball | +0.0078 | [+0.0063, +0.0098] | 1 |
| context vs base head | mae | -0.1026 | [-0.1284, -0.0818] | 3.2e-33 |
| context vs base head | pinball | -0.0415 | [-0.0492, -0.0353] | 3.8e-53 |
| A3 real vs null | mae | +0.0047 | [-0.0096, +0.0175] | 0.82 |
| A3 real vs null | pinball | -0.0051 | [-0.0086, -0.0017] | 3.9e-05 |
| A3 real vs null+otg | mae | +0.0025 | [-0.0095, +0.0140] | 0.7 |
| A3 real vs null+otg | pinball | -0.0038 | [-0.0070, -0.0010] | 4.2e-04 |
| A3 real vs permuted | mae | +0.0016 | [-0.0127, +0.0145] | 0.62 |
| A3 real vs permuted | pinball | -0.0062 | [-0.0098, -0.0029] | 5.3e-07 |
| A3 real vs permuted+otg | mae | +0.0005 | [-0.0118, +0.0117] | 0.54 |
| A3 real vs permuted+otg | pinball | -0.0052 | [-0.0083, -0.0024] | 3.9e-06 |
| A4 all vs market-only | mae | +0.0016 | [-0.0141, +0.0150] | 0.61 |
| A4 all vs market-only | pinball | -0.0057 | [-0.0094, -0.0025] | 4.6e-06 |
| A4 all vs market-only+otg | mae | -0.0005 | [-0.0140, +0.0114] | 0.46 |
| A4 all vs market-only+otg | pinball | -0.0045 | [-0.0077, -0.0016] | 6.2e-05 |
| A6 all vs no carbon | mae | +0.0053 | [+0.0013, +0.0088] | 1 |
| A6 all vs no carbon | pinball | +0.0006 | [-0.0002, +0.0014] | 0.96 |
| A6 all vs no carbon+otg | mae | +0.0050 | [+0.0014, +0.0084] | 1 |
| A6 all vs no carbon+otg | pinball | +0.0012 | [+0.0004, +0.0020] | 1 |

## T3 — conventional generation

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| mcag_regime+otg | 0.861 | 0.763 | 0.264 | 0.780 | 0.880 |
| mcag_regime | 0.862 | 0.763 | 0.258 | 0.785 | 0.885 |
| mcag_fixed+otg | 0.863 | 0.764 | 0.264 | 0.779 | 0.881 |
| mcag_instance+otg | 0.863 | 0.764 | 0.264 | 0.777 | 0.879 |
| mcag_fixed | 0.865 | 0.765 | 0.259 | 0.783 | 0.885 |
| mcag_instance | 0.865 | 0.766 | 0.259 | 0.783 | 0.884 |
| fusion_fixed+otg | 0.866 | 0.767 | 0.264 | 0.778 | 0.882 |
| no_carbon+otg | 0.866 | 0.767 | 0.265 | 0.773 | 0.876 |
| no_carbon | 0.873 | 0.773 | 0.259 | 0.781 | 0.882 |
| fusion_fixed | 0.877 | 0.776 | 0.260 | 0.780 | 0.885 |
| fusion_fixed:base | 0.919 | 0.814 | 0.288 | 0.774 | 0.875 |
| no_carbon:base | 0.954 | 0.844 | 0.303 | 0.760 | 0.866 |
| mcag_regime:base | 0.954 | 0.845 | 0.303 | 0.760 | 0.867 |
| mcag_fixed:base | 0.955 | 0.845 | 0.303 | 0.761 | 0.867 |
| mcag_instance:base | 0.955 | 0.846 | 0.303 | 0.760 | 0.866 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0119 | [-0.0192, -0.0048] | 3.1e-06 |
| A2 instance vs fusion | pinball | -0.0014 | [-0.0031, +0.0004] | 0.018 |
| A2+OTG instance vs fusion | mae | -0.0029 | [-0.0083, +0.0026] | 0.097 |
| A2+OTG instance vs fusion | pinball | -0.0002 | [-0.0018, +0.0014] | 0.38 |
| ladder regime vs fixed | mae | -0.0025 | [-0.0037, -0.0015] | 7.4e-09 |
| ladder regime vs fixed | pinball | -0.0003 | [-0.0007, -0.0001] | 0.0012 |
| ladder instance vs regime | mae | +0.0031 | [+0.0005, +0.0059] | 1 |
| ladder instance vs regime | pinball | +0.0005 | [-0.0001, +0.0013] | 0.96 |
| ladder OTG vs instance | mae | -0.0021 | [-0.0062, +0.0023] | 0.11 |
| ladder OTG vs instance | pinball | +0.0054 | [+0.0035, +0.0070] | 1 |
| context vs base head | mae | -0.0924 | [-0.1210, -0.0660] | 1.5e-18 |
| context vs base head | pinball | -0.0392 | [-0.0497, -0.0304] | 2.2e-27 |
| A6 all vs no carbon | mae | -0.0009 | [-0.0063, +0.0043] | 0.33 |
| A6 all vs no carbon | pinball | +0.0009 | [-0.0001, +0.0019] | 0.98 |
| A6 all vs no carbon+otg | mae | +0.0016 | [-0.0030, +0.0062] | 0.8 |
| A6 all vs no carbon+otg | pinball | +0.0000 | [-0.0012, +0.0013] | 0.54 |

## T4 — RE generation

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| mcag_fixed | 1.213 | 0.751 | 0.378 | 0.753 | 0.871 |
| mcag_regime | 1.218 | 0.754 | 0.381 | 0.749 | 0.869 |
| mcag_instance | 1.222 | 0.757 | 0.382 | 0.747 | 0.871 |
| fusion_fixed | 1.235 | 0.765 | 0.384 | 0.753 | 0.871 |
| mcag_fixed+otg | 1.253 | 0.776 | 0.412 | 0.759 | 0.864 |
| fusion_fixed+otg | 1.255 | 0.777 | 0.411 | 0.749 | 0.860 |
| mcag_regime+otg | 1.259 | 0.779 | 0.414 | 0.754 | 0.863 |
| mcag_instance+otg | 1.264 | 0.783 | 0.415 | 0.754 | 0.862 |
| fusion_fixed:base | 1.340 | 0.830 | 0.446 | 0.740 | 0.846 |
| mcag_fixed:base | 1.379 | 0.854 | 0.469 | 0.730 | 0.841 |
| mcag_regime:base | 1.380 | 0.854 | 0.469 | 0.730 | 0.841 |
| mcag_instance:base | 1.382 | 0.856 | 0.470 | 0.731 | 0.840 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0127 | [-0.0335, +0.0049] | 0.024 |
| A2 instance vs fusion | pinball | -0.0027 | [-0.0077, +0.0023] | 0.058 |
| A2+OTG instance vs fusion | mae | +0.0083 | [-0.0145, +0.0328] | 0.86 |
| A2+OTG instance vs fusion | pinball | +0.0046 | [-0.0060, +0.0182] | 0.87 |
| ladder regime vs fixed | mae | +0.0051 | [-0.0057, +0.0159] | 0.91 |
| ladder regime vs fixed | pinball | +0.0021 | [-0.0012, +0.0068] | 0.94 |
| ladder instance vs regime | mae | +0.0045 | [-0.0033, +0.0128] | 0.93 |
| ladder instance vs regime | pinball | +0.0011 | [-0.0011, +0.0034] | 0.92 |
| ladder OTG vs instance | mae | +0.0414 | [+0.0007, +0.0949] | 0.99 |
| ladder OTG vs instance | pinball | +0.0338 | [+0.0111, +0.0660] | 1 |
| context vs base head | mae | -0.1181 | [-0.1905, -0.0551] | 9.6e-07 |
| context vs base head | pinball | -0.0542 | [-0.0836, -0.0300] | 1.1e-08 |

## T5 — DAM price

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| permuted_context | 0.916 | 0.892 | 0.288 | 0.751 | 0.857 |
| null_context | 0.918 | 0.894 | 0.289 | 0.748 | 0.849 |
| permuted_context+otg | 0.927 | 0.902 | 0.300 | 0.771 | 0.866 |
| null_context+otg | 0.927 | 0.903 | 0.293 | 0.781 | 0.876 |
| null_context:base | 0.940 | 0.915 | 0.279 | 0.730 | 0.852 |
| permuted_context:base | 0.944 | 0.919 | 0.278 | 0.731 | 0.848 |
| mcag_fixed+otg | 0.948 | 0.923 | 0.288 | 0.703 | 0.827 |
| mcag_regime+otg | 0.949 | 0.924 | 0.288 | 0.700 | 0.825 |
| fusion_fixed+otg | 0.949 | 0.925 | 0.286 | 0.679 | 0.824 |
| mcag_instance+otg | 0.964 | 0.939 | 0.292 | 0.696 | 0.825 |
| market_only+otg | 0.966 | 0.941 | 0.291 | 0.728 | 0.852 |
| fusion_fixed:base | 0.970 | 0.945 | 0.296 | 0.706 | 0.819 |
| mcag_fixed:base | 0.982 | 0.957 | 0.302 | 0.708 | 0.809 |
| mcag_regime:base | 0.983 | 0.957 | 0.301 | 0.707 | 0.810 |
| market_only:base | 0.989 | 0.963 | 0.299 | 0.706 | 0.823 |
| mcag_instance:base | 0.992 | 0.966 | 0.304 | 0.703 | 0.811 |
| market_only | 1.371 | 1.335 | 0.281 | 0.736 | 0.862 |
| mcag_fixed | 1.565 | 1.524 | 0.269 | 0.754 | 0.868 |
| mcag_regime | 1.569 | 1.528 | 0.270 | 0.753 | 0.868 |
| mcag_instance | 1.604 | 1.562 | 0.276 | 0.758 | 0.869 |
| fusion_fixed | 1.739 | 1.694 | 0.319 | 0.722 | 0.830 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.1347 | [-0.1971, -0.0738] | 1.2e-11 |
| A2 instance vs fusion | pinball | -0.0429 | [-0.0671, -0.0236] | 5.2e-10 |
| A2+OTG instance vs fusion | mae | +0.0148 | [-0.0005, +0.0305] | 0.99 |
| A2+OTG instance vs fusion | pinball | +0.0055 | [+0.0011, +0.0102] | 1 |
| ladder regime vs fixed | mae | +0.0041 | [-0.0058, +0.0144] | 0.89 |
| ladder regime vs fixed | pinball | +0.0005 | [-0.0024, +0.0038] | 0.69 |
| ladder instance vs regime | mae | +0.0355 | [+0.0022, +0.0702] | 1 |
| ladder instance vs regime | pinball | +0.0057 | [-0.0040, +0.0148] | 0.97 |
| ladder OTG vs instance | mae | -0.6438 | [-0.8178, -0.4410] | 3.2e-25 |
| ladder OTG vs instance | pinball | +0.0156 | [-0.0155, +0.0538] | 0.9 |
| context vs base head | mae | -0.0269 | [-0.0911, +0.0430] | 0.13 |
| context vs base head | pinball | -0.0118 | [-0.0293, +0.0050] | 0.021 |
| A3 real vs null | mae | +0.7188 | [+0.4963, +0.9257] | 1 |
| A3 real vs null | pinball | -0.0053 | [-0.0407, +0.0268] | 0.32 |
| A3 real vs null+otg | mae | +0.0376 | [-0.0340, +0.1184] | 0.92 |
| A3 real vs null+otg | pinball | -0.0023 | [-0.0176, +0.0142] | 0.35 |
| A3 real vs permuted | mae | +0.7207 | [+0.4976, +0.9284] | 1 |
| A3 real vs permuted | pinball | -0.0043 | [-0.0407, +0.0285] | 0.36 |
| A3 real vs permuted+otg | mae | +0.0381 | [-0.0422, +0.1267] | 0.91 |
| A3 real vs permuted+otg | pinball | -0.0086 | [-0.0256, +0.0093] | 0.088 |
| A4 all vs market-only | mae | +0.2667 | [+0.0561, +0.4397] | 1 |
| A4 all vs market-only | pinball | +0.0029 | [-0.0395, +0.0389] | 0.59 |
| A4 all vs market-only+otg | mae | -0.0011 | [-0.0699, +0.0689] | 0.48 |
| A4 all vs market-only+otg | pinball | +0.0003 | [-0.0136, +0.0144] | 0.53 |

## L1 source loss (RE and weather blocks missing on the development test)

| tid | model | mase | mase_source_loss | mean_diff | lo | hi | p_instance_degrades_less |
|---|---|---|---|---|---|---|---|
| T1 | mcag_instance | 0.9394 | 0.9526 |  |  |  |  |
| T1 | fusion_fixed | 0.9349 | 0.9518 |  |  |  |  |
| T1 | instance − fusion (degradation difference) |  |  | -0.0037 | -0.0062 | -0.0012 | 0.0003336 |
| T2 | mcag_instance | 0.967 | 0.9784 |  |  |  |  |
| T2 | fusion_fixed | 0.966 | 0.9738 |  |  |  |  |
| T2 | instance − fusion (degradation difference) |  |  | 0.0035 | 0.0013 | 0.006 | 0.9999 |

## F5 price of adaptivity — PART (validation) vs realised gain (development test)

Sign agreement: 29/45 cells.

| tid | H | refinement | part_G_validation | predicted_refine | realised_gain_dev | agree |
|---|---|---|---|---|---|---|
| T1 | 1 | regime vs fixed | 0.00502 | True | 0.0028 | True |
| T1 | 1 | instance vs fixed | -0.0358 | False | -0.00905 | True |
| T1 | 1 | OTG vs fixed | -0.00288 | False | -0.0025 | True |
| T1 | 2 | regime vs fixed | 0.00082702 | True | 0.00534 | True |
| T1 | 2 | instance vs fixed | -0.07171 | False | -0.02325 | True |
| T1 | 2 | OTG vs fixed | -0.00604 | False | -0.01145 | True |
| T1 | 3 | regime vs fixed | -0.00608 | False | 0.00295 | False |
| T1 | 3 | instance vs fixed | -0.04791 | False | -0.01598 | True |
| T1 | 3 | OTG vs fixed | -0.00991 | False | -0.01549 | True |
| T2 | 1 | regime vs fixed | -0.00072376 | False | 0.00353 | False |
| T2 | 1 | instance vs fixed | -0.01291 | False | 0.00103 | False |
| T2 | 1 | OTG vs fixed | -0.00303 | False | -0.01256 | True |
| T2 | 2 | regime vs fixed | -0.00076667 | False | 0.00226 | False |
| T2 | 2 | instance vs fixed | -0.01992 | False | -0.01894 | True |
| T2 | 2 | OTG vs fixed | -0.00439 | False | -0.02354 | True |
| T2 | 3 | regime vs fixed | -0.00324 | False | 0.00133 | False |
| T2 | 3 | instance vs fixed | -0.01452 | False | -0.0236 | True |
| T2 | 3 | OTG vs fixed | -0.00464 | False | -0.03209 | True |
| T3 | 1 | regime vs fixed | -0.00034847 | False | 0.00395 | False |
| T3 | 1 | instance vs fixed | -0.05658 | False | -0.00888 | True |
| T3 | 1 | OTG vs fixed | -0.00234 | False | 0.0046 | False |
| T3 | 2 | regime vs fixed | -0.00122 | False | 0.00304 | False |
| T3 | 2 | instance vs fixed | -0.06761 | False | -0.00568 | True |
| T3 | 2 | OTG vs fixed | -0.0015 | False | 0.00048871 | False |
| T3 | 3 | regime vs fixed | -0.0049 | False | 0.00516 | False |
| T3 | 3 | instance vs fixed | -0.08992 | False | -0.00234 | True |
| T3 | 3 | OTG vs fixed | -3.3315e-05 | False | -0.00151 | True |
| T4 | 1 | regime vs fixed | 0.00271 | True | -0.04279 | False |
| T4 | 1 | instance vs fixed | -0.12473 | False | -0.10571 | True |
| T4 | 1 | OTG vs fixed | -0.0029 | False | -1.26589 | True |
| T4 | 2 | regime vs fixed | 0.01257 | True | -0.03244 | False |
| T4 | 2 | instance vs fixed | -0.09641 | False | -0.06339 | True |
| T4 | 2 | OTG vs fixed | 0.00066574 | True | -1.59035 | False |
| T4 | 3 | regime vs fixed | 0.01638 | True | -0.06552 | False |
| T4 | 3 | instance vs fixed | -0.11149 | False | -0.08087 | True |
| T4 | 3 | OTG vs fixed | 0.00316 | True | -2.28675 | False |
| T5 | 1 | regime vs fixed | 0.04623 | True | -0.06961 | False |
| T5 | 1 | instance vs fixed | -0.47341 | False | -0.21017 | True |
| T5 | 1 | OTG vs fixed | 0.02762 | True | 1.70571 | True |
| T5 | 2 | regime vs fixed | 0.00669 | True | 0.04303 | True |
| T5 | 2 | instance vs fixed | -0.26086 | False | -0.12867 | True |
| T5 | 2 | OTG vs fixed | 0.01819 | True | 2.17342 | True |
| T5 | 3 | regime vs fixed | -0.00322 | False | -0.01679 | True |
| T5 | 3 | instance vs fixed | -0.17319 | False | -0.09636 | True |
| T5 | 3 | OTG vs fixed | 0.01554 | True | 2.26237 | True |
