# O3 development results — LA-MCAG (development test FY2024-25)

MASE uses train-period seasonal-naive scales; rel. SN is the scaled MAE ratio to the seasonal naive on the same rows. Intervals: rolling-180 conformal at C1 for every variant. `+otg` = online time-adaptive gate; `:base` = base head without context.

## T1 — energy met

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| null_context | 0.928 | 0.698 | 0.289 | 0.793 | 0.895 |
| market_only+otg | 0.933 | 0.702 | 0.296 | 0.788 | 0.886 |
| permuted_context | 0.933 | 0.702 | 0.291 | 0.794 | 0.894 |
| market_only | 0.935 | 0.704 | 0.293 | 0.796 | 0.893 |
| null_context+otg | 0.936 | 0.704 | 0.295 | 0.782 | 0.883 |
| permuted_context+otg | 0.939 | 0.707 | 0.298 | 0.783 | 0.883 |
| mcag_regime+otg | 0.945 | 0.711 | 0.294 | 0.784 | 0.881 |
| mcag_instance+otg | 0.948 | 0.713 | 0.296 | 0.784 | 0.880 |
| mcag_fixed+otg | 0.948 | 0.713 | 0.295 | 0.782 | 0.881 |
| mcag_regime | 0.954 | 0.718 | 0.287 | 0.790 | 0.890 |
| fusion_fixed+otg | 0.954 | 0.718 | 0.298 | 0.789 | 0.883 |
| no_carbon+otg | 0.957 | 0.720 | 0.298 | 0.785 | 0.881 |
| mcag_fixed | 0.960 | 0.722 | 0.288 | 0.787 | 0.888 |
| mcag_instance | 0.961 | 0.723 | 0.289 | 0.786 | 0.887 |
| no_carbon | 0.973 | 0.732 | 0.292 | 0.786 | 0.888 |
| market_only:base | 0.991 | 0.746 | 0.314 | 0.786 | 0.886 |
| fusion_fixed | 1.003 | 0.755 | 0.299 | 0.780 | 0.886 |
| fusion_fixed:base | 1.003 | 0.755 | 0.318 | 0.786 | 0.883 |
| null_context:base | 1.024 | 0.771 | 0.322 | 0.784 | 0.884 |
| permuted_context:base | 1.043 | 0.784 | 0.330 | 0.778 | 0.881 |
| mcag_regime:base | 1.065 | 0.801 | 0.338 | 0.777 | 0.879 |
| mcag_fixed:base | 1.066 | 0.802 | 0.338 | 0.777 | 0.879 |
| mcag_instance:base | 1.069 | 0.804 | 0.339 | 0.777 | 0.879 |
| no_carbon:base | 1.070 | 0.805 | 0.339 | 0.778 | 0.879 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0419 | [-0.0567, -0.0269] | 4.3e-15 |
| A2 instance vs fusion | pinball | -0.0094 | [-0.0128, -0.0060] | 4.0e-14 |
| A2+OTG instance vs fusion | mae | -0.0067 | [-0.0127, -0.0005] | 0.0032 |
| A2+OTG instance vs fusion | pinball | -0.0021 | [-0.0042, -0.0003] | 0.002 |
| ladder regime vs fixed | mae | -0.0060 | [-0.0085, -0.0037] | 3.1e-11 |
| ladder regime vs fixed | pinball | -0.0008 | [-0.0013, -0.0003] | 1.1e-04 |
| ladder instance vs regime | mae | +0.0075 | [+0.0036, +0.0122] | 1 |
| ladder instance vs regime | pinball | +0.0020 | [+0.0011, +0.0033] | 1 |
| ladder OTG vs instance | mae | -0.0138 | [-0.0220, -0.0032] | 7.7e-05 |
| ladder OTG vs instance | pinball | +0.0065 | [+0.0034, +0.0105] | 1 |
| context vs base head | mae | -0.1209 | [-0.1469, -0.0974] | 1.7e-30 |
| context vs base head | pinball | -0.0437 | [-0.0512, -0.0363] | 3.3e-41 |
| A3 real vs null | mae | +0.0443 | [+0.0142, +0.0698] | 1 |
| A3 real vs null | pinball | +0.0024 | [-0.0045, +0.0086] | 0.83 |
| A3 real vs null+otg | mae | +0.0193 | [-0.0034, +0.0391] | 0.99 |
| A3 real vs null+otg | pinball | +0.0025 | [-0.0024, +0.0068] | 0.92 |
| A3 real vs permuted | mae | +0.0392 | [+0.0090, +0.0643] | 1 |
| A3 real vs permuted | pinball | +0.0008 | [-0.0060, +0.0070] | 0.63 |
| A3 real vs permuted+otg | mae | +0.0157 | [-0.0060, +0.0351] | 0.97 |
| A3 real vs permuted+otg | pinball | -0.0000 | [-0.0046, +0.0042] | 0.5 |
| A4 all vs market-only | mae | +0.0374 | [+0.0092, +0.0619] | 1 |
| A4 all vs market-only | pinball | -0.0009 | [-0.0073, +0.0052] | 0.36 |
| A4 all vs market-only+otg | mae | +0.0221 | [+0.0018, +0.0401] | 1 |
| A4 all vs market-only+otg | pinball | +0.0014 | [-0.0030, +0.0052] | 0.81 |
| A6 all vs no carbon | mae | -0.0010 | [-0.0073, +0.0063] | 0.37 |
| A6 all vs no carbon | pinball | -0.0005 | [-0.0017, +0.0011] | 0.22 |
| A6 all vs no carbon+otg | mae | -0.0025 | [-0.0077, +0.0034] | 0.13 |
| A6 all vs no carbon+otg | pinball | -0.0009 | [-0.0021, +0.0004] | 0.034 |

## T2 — actual drawal

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| null_context | 0.947 | 0.722 | 0.291 | 0.796 | 0.896 |
| market_only | 0.950 | 0.725 | 0.292 | 0.797 | 0.896 |
| permuted_context | 0.951 | 0.725 | 0.291 | 0.796 | 0.896 |
| null_context+otg | 0.951 | 0.725 | 0.297 | 0.781 | 0.881 |
| fusion_fixed+otg | 0.952 | 0.726 | 0.294 | 0.788 | 0.883 |
| mcag_regime+otg | 0.954 | 0.727 | 0.293 | 0.782 | 0.882 |
| mcag_fixed+otg | 0.954 | 0.728 | 0.294 | 0.782 | 0.881 |
| market_only+otg | 0.954 | 0.728 | 0.298 | 0.779 | 0.880 |
| mcag_instance+otg | 0.954 | 0.728 | 0.294 | 0.782 | 0.881 |
| permuted_context+otg | 0.956 | 0.729 | 0.297 | 0.784 | 0.884 |
| mcag_fixed | 0.960 | 0.733 | 0.287 | 0.791 | 0.896 |
| mcag_regime | 0.964 | 0.735 | 0.287 | 0.791 | 0.896 |
| mcag_instance | 0.966 | 0.737 | 0.288 | 0.791 | 0.896 |
| no_carbon+otg | 0.969 | 0.739 | 0.297 | 0.782 | 0.881 |
| fusion_fixed | 0.975 | 0.744 | 0.290 | 0.795 | 0.898 |
| market_only:base | 0.997 | 0.761 | 0.310 | 0.790 | 0.888 |
| fusion_fixed:base | 0.999 | 0.762 | 0.311 | 0.792 | 0.888 |
| no_carbon | 1.002 | 0.764 | 0.294 | 0.786 | 0.890 |
| null_context:base | 1.020 | 0.778 | 0.317 | 0.788 | 0.887 |
| permuted_context:base | 1.021 | 0.778 | 0.318 | 0.789 | 0.887 |
| mcag_fixed:base | 1.047 | 0.798 | 0.327 | 0.786 | 0.884 |
| mcag_regime:base | 1.047 | 0.798 | 0.327 | 0.786 | 0.884 |
| mcag_instance:base | 1.048 | 0.799 | 0.327 | 0.786 | 0.884 |
| no_carbon:base | 1.049 | 0.800 | 0.327 | 0.786 | 0.884 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0095 | [-0.0169, -0.0024] | 8.1e-05 |
| A2 instance vs fusion | pinball | -0.0025 | [-0.0041, -0.0011] | 8.0e-06 |
| A2+OTG instance vs fusion | mae | +0.0026 | [-0.0018, +0.0068] | 0.93 |
| A2+OTG instance vs fusion | pinball | +0.0000 | [-0.0012, +0.0010] | 0.5 |
| ladder regime vs fixed | mae | +0.0037 | [+0.0006, +0.0072] | 1 |
| ladder regime vs fixed | pinball | +0.0000 | [-0.0007, +0.0008] | 0.52 |
| ladder instance vs regime | mae | +0.0016 | [-0.0007, +0.0038] | 0.95 |
| ladder instance vs regime | pinball | +0.0010 | [+0.0004, +0.0016] | 1 |
| ladder OTG vs instance | mae | -0.0118 | [-0.0198, -0.0029] | 2.1e-04 |
| ladder OTG vs instance | pinball | +0.0064 | [+0.0037, +0.0093] | 1 |
| context vs base head | mae | -0.0933 | [-0.1112, -0.0751] | 5.6e-33 |
| context vs base head | pinball | -0.0334 | [-0.0383, -0.0284] | 2.0e-48 |
| A3 real vs null | mae | +0.0190 | [+0.0012, +0.0341] | 1 |
| A3 real vs null | pinball | -0.0026 | [-0.0067, +0.0014] | 0.054 |
| A3 real vs null+otg | mae | +0.0056 | [-0.0092, +0.0187] | 0.83 |
| A3 real vs null+otg | pinball | -0.0022 | [-0.0059, +0.0012] | 0.053 |
| A3 real vs permuted | mae | +0.0151 | [-0.0018, +0.0293] | 0.99 |
| A3 real vs permuted | pinball | -0.0032 | [-0.0074, +0.0007] | 0.024 |
| A3 real vs permuted+otg | mae | +0.0004 | [-0.0126, +0.0122] | 0.53 |
| A3 real vs permuted+otg | pinball | -0.0021 | [-0.0055, +0.0013] | 0.062 |
| A4 all vs market-only | mae | +0.0158 | [-0.0032, +0.0313] | 0.98 |
| A4 all vs market-only | pinball | -0.0042 | [-0.0085, -0.0000] | 0.0066 |
| A4 all vs market-only+otg | mae | +0.0026 | [-0.0128, +0.0157] | 0.67 |
| A4 all vs market-only+otg | pinball | -0.0034 | [-0.0075, +0.0002] | 0.012 |
| A6 all vs no carbon | mae | -0.0372 | [-0.0484, -0.0273] | 2.9e-19 |
| A6 all vs no carbon | pinball | -0.0061 | [-0.0088, -0.0039] | 1.9e-11 |
| A6 all vs no carbon+otg | mae | -0.0129 | [-0.0191, -0.0077] | 4.3e-09 |
| A6 all vs no carbon+otg | pinball | -0.0019 | [-0.0035, -0.0006] | 0.0017 |

## T3 — conventional generation

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| mcag_instance+otg | 0.782 | 0.793 | 0.242 | 0.789 | 0.886 |
| mcag_instance | 0.782 | 0.794 | 0.233 | 0.790 | 0.891 |
| fusion_fixed+otg | 0.783 | 0.795 | 0.242 | 0.795 | 0.894 |
| mcag_regime+otg | 0.784 | 0.795 | 0.241 | 0.789 | 0.886 |
| mcag_fixed+otg | 0.785 | 0.796 | 0.241 | 0.787 | 0.886 |
| mcag_regime | 0.785 | 0.796 | 0.233 | 0.789 | 0.891 |
| mcag_fixed | 0.787 | 0.798 | 0.234 | 0.791 | 0.891 |
| fusion_fixed | 0.790 | 0.801 | 0.235 | 0.798 | 0.895 |
| no_carbon+otg | 0.790 | 0.801 | 0.244 | 0.791 | 0.888 |
| no_carbon | 0.792 | 0.804 | 0.236 | 0.792 | 0.892 |
| fusion_fixed:base | 0.808 | 0.819 | 0.254 | 0.797 | 0.890 |
| no_carbon:base | 0.829 | 0.841 | 0.263 | 0.793 | 0.886 |
| mcag_instance:base | 0.830 | 0.842 | 0.263 | 0.792 | 0.885 |
| mcag_fixed:base | 0.830 | 0.842 | 0.263 | 0.792 | 0.885 |
| mcag_regime:base | 0.830 | 0.843 | 0.263 | 0.792 | 0.885 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0073 | [-0.0121, -0.0026] | 1.3e-04 |
| A2 instance vs fusion | pinball | -0.0011 | [-0.0024, +0.0001] | 0.016 |
| A2+OTG instance vs fusion | mae | -0.0012 | [-0.0060, +0.0032] | 0.28 |
| A2+OTG instance vs fusion | pinball | -0.0004 | [-0.0018, +0.0009] | 0.22 |
| ladder regime vs fixed | mae | -0.0019 | [-0.0039, -0.0001] | 0.0062 |
| ladder regime vs fixed | pinball | -0.0009 | [-0.0014, -0.0004] | 5.3e-07 |
| ladder instance vs regime | mae | -0.0026 | [-0.0046, -0.0004] | 0.0015 |
| ladder instance vs regime | pinball | +0.0004 | [-0.0002, +0.0010] | 0.93 |
| ladder OTG vs instance | mae | -0.0005 | [-0.0063, +0.0060] | 0.42 |
| ladder OTG vs instance | pinball | +0.0080 | [+0.0050, +0.0116] | 1 |
| context vs base head | mae | -0.0475 | [-0.0701, -0.0298] | 1.4e-09 |
| context vs base head | pinball | -0.0215 | [-0.0284, -0.0158] | 1.2e-19 |
| A6 all vs no carbon | mae | -0.0044 | [-0.0080, -0.0004] | 0.0037 |
| A6 all vs no carbon | pinball | -0.0011 | [-0.0019, -0.0002] | 0.0035 |
| A6 all vs no carbon+otg | mae | -0.0039 | [-0.0073, -0.0003] | 0.0044 |
| A6 all vs no carbon+otg | pinball | -0.0011 | [-0.0021, -0.0001] | 0.002 |

## T4 — RE generation

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| mcag_instance | 0.855 | 0.782 | 0.271 | 0.797 | 0.889 |
| mcag_fixed | 0.860 | 0.786 | 0.272 | 0.798 | 0.891 |
| mcag_instance+otg | 0.862 | 0.788 | 0.281 | 0.795 | 0.890 |
| fusion_fixed+otg | 0.862 | 0.788 | 0.285 | 0.794 | 0.887 |
| mcag_regime | 0.862 | 0.788 | 0.272 | 0.798 | 0.889 |
| fusion_fixed | 0.863 | 0.789 | 0.268 | 0.803 | 0.892 |
| mcag_fixed+otg | 0.865 | 0.791 | 0.284 | 0.785 | 0.882 |
| mcag_regime+otg | 0.865 | 0.791 | 0.282 | 0.789 | 0.886 |
| fusion_fixed:base | 0.898 | 0.821 | 0.298 | 0.793 | 0.888 |
| mcag_regime:base | 0.910 | 0.832 | 0.304 | 0.792 | 0.888 |
| mcag_fixed:base | 0.911 | 0.833 | 0.303 | 0.792 | 0.888 |
| mcag_instance:base | 0.912 | 0.834 | 0.304 | 0.791 | 0.888 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | -0.0084 | [-0.0174, +0.0016] | 0.012 |
| A2 instance vs fusion | pinball | +0.0024 | [-0.0003, +0.0052] | 0.99 |
| A2+OTG instance vs fusion | mae | -0.0005 | [-0.0099, +0.0090] | 0.45 |
| A2+OTG instance vs fusion | pinball | -0.0047 | [-0.0077, -0.0017] | 8.8e-06 |
| ladder regime vs fixed | mae | +0.0024 | [-0.0004, +0.0051] | 0.99 |
| ladder regime vs fixed | pinball | +0.0000 | [-0.0006, +0.0007] | 0.53 |
| ladder instance vs regime | mae | -0.0070 | [-0.0110, -0.0037] | 1.6e-06 |
| ladder instance vs regime | pinball | -0.0014 | [-0.0030, -0.0002] | 0.0048 |
| ladder OTG vs instance | mae | +0.0065 | [-0.0007, +0.0142] | 0.99 |
| ladder OTG vs instance | pinball | +0.0103 | [+0.0064, +0.0149] | 1 |
| context vs base head | mae | -0.0504 | [-0.0791, -0.0219] | 2.0e-07 |
| context vs base head | pinball | -0.0228 | [-0.0329, -0.0135] | 1.6e-11 |

## T5 — DAM price

| variant | MASE | rel. SN | pinball | cov80 | cov90 |
|---|---|---|---|---|---|
| market_only+otg | 0.806 | 0.823 | 0.262 | 0.790 | 0.890 |
| market_only | 0.819 | 0.836 | 0.258 | 0.789 | 0.895 |
| permuted_context | 0.864 | 0.882 | 0.277 | 0.830 | 0.896 |
| permuted_context+otg | 0.866 | 0.885 | 0.279 | 0.819 | 0.898 |
| fusion_fixed+otg | 0.871 | 0.889 | 0.289 | 0.594 | 0.719 |
| mcag_instance+otg | 0.871 | 0.890 | 0.278 | 0.645 | 0.766 |
| null_context:base | 0.877 | 0.896 | 0.270 | 0.845 | 0.931 |
| null_context+otg | 0.878 | 0.897 | 0.273 | 0.787 | 0.882 |
| mcag_regime+otg | 0.880 | 0.899 | 0.268 | 0.672 | 0.783 |
| mcag_fixed+otg | 0.880 | 0.899 | 0.268 | 0.668 | 0.782 |
| permuted_context:base | 0.882 | 0.901 | 0.268 | 0.847 | 0.939 |
| null_context | 0.882 | 0.901 | 0.289 | 0.818 | 0.894 |
| market_only:base | 0.884 | 0.903 | 0.270 | 0.843 | 0.937 |
| mcag_fixed:base | 0.885 | 0.904 | 0.270 | 0.842 | 0.937 |
| mcag_regime:base | 0.886 | 0.904 | 0.271 | 0.841 | 0.936 |
| fusion_fixed:base | 0.887 | 0.906 | 0.271 | 0.845 | 0.940 |
| mcag_instance:base | 0.890 | 0.909 | 0.273 | 0.839 | 0.938 |
| fusion_fixed | 1.267 | 1.294 | 0.245 | 0.823 | 0.903 |
| mcag_regime | 1.348 | 1.377 | 0.268 | 0.794 | 0.882 |
| mcag_fixed | 1.373 | 1.402 | 0.268 | 0.801 | 0.887 |
| mcag_instance | 1.379 | 1.408 | 0.255 | 0.806 | 0.892 |

| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |
|---|---|---|---|---|
| A2 instance vs fusion | mae | +0.1124 | [+0.0647, +0.1612] | 1 |
| A2 instance vs fusion | pinball | +0.0109 | [+0.0033, +0.0200] | 1 |
| A2+OTG instance vs fusion | mae | +0.0007 | [-0.0131, +0.0153] | 0.55 |
| A2+OTG instance vs fusion | pinball | -0.0112 | [-0.0176, -0.0055] | 1.6e-07 |
| ladder regime vs fixed | mae | -0.0249 | [-0.0396, -0.0096] | 1.6e-08 |
| ladder regime vs fixed | pinball | +0.0002 | [-0.0025, +0.0026] | 0.6 |
| ladder instance vs regime | mae | +0.0308 | [-0.0064, +0.0662] | 1 |
| ladder instance vs regime | pinball | -0.0124 | [-0.0198, -0.0056] | 2.1e-06 |
| ladder OTG vs instance | mae | -0.5098 | [-0.6594, -0.3692] | 4.6e-24 |
| ladder OTG vs instance | pinball | +0.0221 | [-0.0114, +0.0564] | 0.97 |
| context vs base head | mae | -0.0184 | [-0.0875, +0.0508] | 0.26 |
| context vs base head | pinball | +0.0049 | [-0.0142, +0.0268] | 0.74 |
| A3 real vs null | mae | +0.5632 | [+0.3930, +0.7387] | 1 |
| A3 real vs null | pinball | -0.0268 | [-0.0551, +0.0024] | 0.0028 |
| A3 real vs null+otg | mae | -0.0022 | [-0.0599, +0.0546] | 0.47 |
| A3 real vs null+otg | pinball | +0.0122 | [-0.0072, +0.0337] | 0.95 |
| A3 real vs permuted | mae | +0.5817 | [+0.3956, +0.7765] | 1 |
| A3 real vs permuted | pinball | -0.0147 | [-0.0426, +0.0166] | 0.066 |
| A3 real vs permuted+otg | mae | +0.0100 | [-0.0581, +0.0752] | 0.64 |
| A3 real vs permuted+otg | pinball | +0.0064 | [-0.0158, +0.0299] | 0.77 |
| A4 all vs market-only | mae | +0.6263 | [+0.4451, +0.8245] | 1 |
| A4 all vs market-only | pinball | +0.0044 | [-0.0161, +0.0268] | 0.72 |
| A4 all vs market-only+otg | mae | +0.0701 | [-0.0138, +0.1526] | 0.98 |
| A4 all vs market-only+otg | pinball | +0.0235 | [-0.0042, +0.0526] | 0.99 |

## L1 source loss (RE and weather blocks missing on the development test)

| tid | model | mase | mase_source_loss | mean_diff | lo | hi | p_instance_degrades_less |
|---|---|---|---|---|---|---|---|
| T1 | mcag_instance | 0.9609 | 0.9685 |  |  |  |  |
| T1 | fusion_fixed | 1.0026 | 0.9898 |  |  |  |  |
| T1 | instance − fusion (degradation difference) |  |  | 0.0203 | 0.0114 | 0.0303 | 1.0 |
| T2 | mcag_instance | 0.9656 | 0.9762 |  |  |  |  |
| T2 | fusion_fixed | 0.9753 | 0.9748 |  |  |  |  |
| T2 | instance − fusion (degradation difference) |  |  | 0.0111 | 0.0057 | 0.0167 | 1.0 |

## F5 price of adaptivity — PART (validation) vs realised gain (development test)

Sign agreement: 19/45 cells.

| tid | H | refinement | part_G_validation | predicted_refine | realised_gain_dev | agree |
|---|---|---|---|---|---|---|
| T1 | 1 | regime vs fixed | -0.0037 | False | 0.00214 | False |
| T1 | 1 | instance vs fixed | -0.05068 | False | -0.01489 | True |
| T1 | 1 | OTG vs fixed | 0.00112 | True | -0.00193 | False |
| T1 | 2 | regime vs fixed | -0.00809 | False | 0.01516 | False |
| T1 | 2 | instance vs fixed | -0.06818 | False | -0.01097 | True |
| T1 | 2 | OTG vs fixed | 0.00277 | True | 0.00401 | True |
| T1 | 3 | regime vs fixed | -0.00898 | False | 0.02413 | False |
| T1 | 3 | instance vs fixed | -0.06965 | False | -0.00931 | True |
| T1 | 3 | OTG vs fixed | 0.00311 | True | 0.00156 | True |
| T2 | 1 | regime vs fixed | -0.00242 | False | 0.00189 | False |
| T2 | 1 | instance vs fixed | -0.02193 | False | -0.00639 | True |
| T2 | 1 | OTG vs fixed | 0.00216 | True | -0.0048 | False |
| T2 | 2 | regime vs fixed | -0.00486 | False | -0.01048 | True |
| T2 | 2 | instance vs fixed | 0.0004203 | True | -0.01654 | False |
| T2 | 2 | OTG vs fixed | 0.00045372 | True | -0.00839 | False |
| T2 | 3 | regime vs fixed | -0.00666 | False | -0.02113 | True |
| T2 | 3 | instance vs fixed | -0.02745 | False | -0.02608 | True |
| T2 | 3 | OTG vs fixed | 5.6335e-05 | True | -0.00668 | False |
| T3 | 1 | regime vs fixed | -0.00529 | False | 0.01217 | False |
| T3 | 1 | instance vs fixed | -0.07842 | False | 0.00717 | False |
| T3 | 1 | OTG vs fixed | -0.00317 | False | -0.02399 | True |
| T3 | 2 | regime vs fixed | -0.00639 | False | 0.00728 | False |
| T3 | 2 | instance vs fixed | -0.09862 | False | 0.01062 | False |
| T3 | 2 | OTG vs fixed | -0.00345 | False | -0.02303 | True |
| T3 | 3 | regime vs fixed | -0.00778 | False | 0.00381 | False |
| T3 | 3 | instance vs fixed | -0.1182 | False | 0.00723 | False |
| T3 | 3 | OTG vs fixed | -0.00528 | False | -0.02429 | True |
| T4 | 1 | regime vs fixed | 0.03456 | True | -0.00455 | False |
| T4 | 1 | instance vs fixed | -0.05648 | False | 0.01182 | False |
| T4 | 1 | OTG vs fixed | -0.01001 | False | -0.02247 | True |
| T4 | 2 | regime vs fixed | 0.01818 | True | -0.00373 | False |
| T4 | 2 | instance vs fixed | -0.83939 | False | 0.00944 | False |
| T4 | 2 | OTG vs fixed | -0.02168 | False | -0.03026 | True |
| T4 | 3 | regime vs fixed | 0.00256 | True | -0.00834 | False |
| T4 | 3 | instance vs fixed | -0.64502 | False | 0.01325 | False |
| T4 | 3 | OTG vs fixed | -0.03989 | False | -0.03305 | True |
| T5 | 1 | regime vs fixed | 0.02771 | True | 0.10537 | True |
| T5 | 1 | instance vs fixed | -0.22291 | False | -0.03837 | True |
| T5 | 1 | OTG vs fixed | -0.20061 | False | 1.72665 | False |
| T5 | 2 | regime vs fixed | -0.03519 | False | 0.0491 | False |
| T5 | 2 | instance vs fixed | 0.1414 | True | -0.07894 | False |
| T5 | 2 | OTG vs fixed | -0.1989 | False | 1.57414 | False |
| T5 | 3 | regime vs fixed | 0.02757 | True | 0.05412 | True |
| T5 | 3 | instance vs fixed | 0.35485 | True | 0.16562 | True |
| T5 | 3 | OTG vs fixed | -0.17341 | False | 1.38747 | False |
