# RefusedH100 run summary

Written 2026-09-17T18:37:27 on master.

| stage | queue | state | seconds | host | PBS job | log |
|---|---|---|---|---|---|---|
| env_check | cpu | done | 2.1 | master | 31785.master | logs/stages/env_check.log |
| leakage_test | cpu | done | 13.7 | master | 31785.master | logs/stages/leakage_test.log |
| o2_samples | cpu | done | 19.1 | master | 31785.master | logs/stages/o2_samples.log |
| o2_patchtst | gpu | done | 365.7 | master | 31770.master | logs/stages/o2_patchtst.log |
| o2_tft | gpu | done | 896.9 | master | 31770.master | logs/stages/o2_tft.log |
| o2_bilstm | gpu | done | 132.1 | master | 31770.master | logs/stages/o2_bilstm.log |
| o2_chronos | gpu | done | 60.5 | master | 31770.master | logs/stages/o2_chronos.log |
| o3_main | gpu | done | 1547.3 | master | 31772.master | logs/stages/o3_main.log |
| o3_ablation | gpu | done | 625.2 | master | 31772.master | logs/stages/o3_ablation.log |
| o3_ablation2 | gpu | done | 229.0 | master | 31772.master | logs/stages/o3_ablation2.log |
| o2_lgbm | cpu | done | 1599.6 | master | 31772.master | logs/stages/o2_lgbm.log |
| o2_evaluate | cpu | done | 1062.6 | master | 31772.master | logs/stages/o2_evaluate.log |
| o4 | cpu | done | 464.1 | master | 31772.master | logs/stages/o4.log |
| o5_rules | cpu | done | 458.3 | master | 31772.master | logs/stages/o5_rules.log |
| o5_ppo | gpu | done | 1727.2 | master | 31772.master | logs/stages/o5_ppo.log |
| o3_evaluate | cpu | done | 354.5 | master | 31783.master | logs/stages/o3_evaluate.log |
| data_paper_numbers | cpu | done | 0.0 | master | 31785.master | logs/stages/data_paper_numbers.log |
| data_paper_figures | cpu | done | 9.6 | master | 31785.master | logs/stages/data_paper_figures.log |
| build_b01_psp | cpu | done | 36.4 | master | 31785.master | logs/stages/build_b01_psp.log |
| build_b02_dsm | cpu | done | 186.5 | master | 31772.master | logs/stages/build_b02_dsm.log |
| build_b03_npp | cpu | done | 62.3 | master | 31772.master | logs/stages/build_b03_npp.log |
| build_b04_re | cpu | done | 38.6 | master | 31772.master | logs/stages/build_b04_re.log |
| build_b05_co2 | cpu | done | 4.3 | master | 31772.master | logs/stages/build_b05_co2.log |
| build_b06_imd | cpu | done | 4.5 | master | 31772.master | logs/stages/build_b06_imd.log |
| build_b07_panel | cpu | done | 7.0 | master | 31785.master | logs/stages/build_b07_panel.log |
| build_b08_validation | cpu | done | 7.3 | master | 31785.master | logs/stages/build_b08_validation.log |
| build_verify | cpu | done | 0.0 | master | 31785.master | logs/stages/build_verify.log |
| prereg_fill | cpu | done | 0.0 | master | 31783.master | logs/stages/prereg_fill.log |
| report | cpu | done | 0.0 | master | 31783.master | logs/stages/report.log |

## Environment (env_check)

* host master, Python 3.10.21, CUDA available False, panel checksum ok: True
* numpy 2.2.6, pandas 2.3.3, pyarrow 25.0.1, scipy 1.15.3, scikit-learn 1.7.2, statsmodels 0.14.6, lightgbm 4.7.0, torch 2.5.1+cu121, pytorch-lightning 2.5.6, neuralforecast 3.2.2, chronos-forecasting 2.3.2, transformers 5.3.0, stable_baselines3 2.7.1, gymnasium 1.2.3, matplotlib 3.10.8, openpyxl 3.1.5, xlrd 2.0.2, pymupdf 1.28.2

## Panel rebuild

**identical panel (content SHA-256 match)**

## Result files

| file | bytes | sha256 (first 16) |
|---|---:|---|
| results/environment.json | 970 | 6ec2539dc7943a35 |
| results/dev/o3/eval/f5.csv | 3433 | fa689a7318914e5e |
| results/dev/o3/eval/l1.csv | 565 | 77c342b98a509334 |
| results/dev/o3/eval/pairs.csv | 69673 | ad73bdab6e4e0e9a |
| results/dev/o3/eval/point.csv | 28552 | cd0690713fca15df |
| results/dev/o3/logs/T1__fusion_fixed.json | 768 | 6336111a5350714d |
| results/dev/o3/logs/T1__market_only.json | 488 | 733546afc73b5d73 |
| results/dev/o3/logs/T1__mcag_fixed.json | 766 | 194e58f2722620c6 |
| results/dev/o3/logs/T1__mcag_instance.json | 770 | 7808a4cc026d5a18 |
| results/dev/o3/logs/T1__mcag_regime.json | 767 | 5eed395b88fb3051 |
| results/dev/o3/logs/T1__no_carbon.json | 572 | 8a14531e70ed5f8f |
| results/dev/o3/logs/T1__null_context.json | 593 | 5cceb6a9adaaf5e7 |
| results/dev/o3/logs/T1__permuted_context.json | 597 | c252a719f515e527 |
| results/dev/o3/logs/T2__fusion_fixed.json | 767 | 99bef9f8f23429ab |
| results/dev/o3/logs/T2__market_only.json | 493 | 2751a6338cd8f93b |
| results/dev/o3/logs/T2__mcag_fixed.json | 766 | d9ec8dad10d75874 |
| results/dev/o3/logs/T2__mcag_instance.json | 770 | e49adbb12bae4524 |
| results/dev/o3/logs/T2__mcag_regime.json | 767 | 9c070e371b7ba4fe |
| results/dev/o3/logs/T2__no_carbon.json | 573 | 55fc587ba9391169 |
| results/dev/o3/logs/T2__null_context.json | 593 | a5fb3a03ffcdb1f5 |
| results/dev/o3/logs/T2__permuted_context.json | 598 | b6f4e2403e62e0cb |
| results/dev/o3/logs/T3__fusion_fixed.json | 766 | 3ebf12d3f439939a |
| results/dev/o3/logs/T3__mcag_fixed.json | 765 | 5c02fa8e56934e90 |
| results/dev/o3/logs/T3__mcag_instance.json | 767 | e84a292eca8a79b0 |
| results/dev/o3/logs/T3__mcag_regime.json | 766 | 4ee259d1801a2981 |
| results/dev/o3/logs/T3__no_carbon.json | 572 | 1cdfa91513b4610d |
| results/dev/o3/logs/T4__fusion_fixed.json | 766 | 309008846c6279b1 |
| results/dev/o3/logs/T4__mcag_fixed.json | 764 | fb61b2a93ab05040 |
| results/dev/o3/logs/T4__mcag_instance.json | 765 | f4f46089764af26c |
| results/dev/o3/logs/T4__mcag_regime.json | 765 | ef7c4159bc050a5b |
| results/dev/o3/logs/T5__fusion_fixed.json | 702 | 54a8a8d2e667c734 |
| results/dev/o3/logs/T5__market_only.json | 485 | 898648a9d9aeedcc |
| results/dev/o3/logs/T5__mcag_fixed.json | 704 | 6e3cac958e9d9227 |
| results/dev/o3/logs/T5__mcag_instance.json | 702 | aad78ba8976d46e7 |
| results/dev/o3/logs/T5__mcag_regime.json | 705 | 8ea3dd3f340fdf76 |
| results/dev/o3/logs/T5__null_context.json | 527 | 107ff7cc44e98f49 |
| results/dev/o3/logs/T5__permuted_context.json | 534 | d6bc8ea5b56bcaf9 |
| results/dev/o2/eval/latency.csv | 3114 | dc65551e898e9333 |
| results/dev/o2/eval/point.csv | 23229 | e70938f15ab08bd2 |
| results/dev/o2/eval/prob.csv | 32844 | c1c221dae069c82e |
| results/dev/o2/eval/seeds.csv | 5816 | 32b65d52f1555059 |
| results/dev/o2/eval/selection.json | 7779 | 71d2ec7992376bce |
| results/dev/o2/eval/tests.csv | 28219 | c886b27488c089d2 |
| results/dev/o2/logs/bilstm_T1.json | 563 | 266139979eb53de6 |
| results/dev/o2/logs/bilstm_T2.json | 561 | 05d540542f4705ae |
| results/dev/o2/logs/bilstm_T3.json | 561 | d2400cbb329f9b92 |
| results/dev/o2/logs/bilstm_T4.json | 558 | 57f6b7185cdb1c53 |
| results/dev/o2/logs/bilstm_T5.json | 558 | d6ae569cab9bf01f |
| results/dev/o2/logs/lgbm_T1.json | 1984 | 4f98c0c281d2db4c |
| results/dev/o2/logs/lgbm_T2.json | 1989 | e2ea3ce8a1c717b0 |
| results/dev/o2/logs/lgbm_T3.json | 1976 | 0ee567521605f84d |
| results/dev/o2/logs/lgbm_T4.json | 1931 | 7f6e1993a9070cbb |
| results/dev/o2/logs/lgbm_T5.json | 1844 | 13822c6597b3778c |
| results/dev/o2/logs/nf_patchtst_T1.json | 1102 | caa5e44e00ae63b9 |
| results/dev/o2/logs/nf_patchtst_T2.json | 1102 | 0ea2f9e5272843da |
| results/dev/o2/logs/nf_patchtst_T3.json | 1102 | 892907bf39774e6d |
| results/dev/o2/logs/nf_patchtst_T4.json | 1142 | 9f2ee77d5cf3e2e7 |
| results/dev/o2/logs/nf_patchtst_T5.json | 742 | 93c06bff25d1adcc |
| results/dev/o2/logs/nf_tft_T1.json | 1100 | 8e3591a17ca8a961 |
| results/dev/o2/logs/nf_tft_T2.json | 1100 | c455f95c512c49ce |
| results/dev/o2/logs/nf_tft_T3.json | 1100 | 3a11ca4266cce3fb |
| results/dev/o2/logs/nf_tft_T4.json | 1140 | ffe4ff0719922a70 |
| results/dev/o2/logs/nf_tft_T5.json | 737 | b3266976e465258e |
| results/dev/o5/frozen_params.json | 196 | 5251d233e4550391 |
| results/dev/o5/ladder.csv | 126 | 6633d37437c5e64f |
| results/dev/o5/metrics.csv | 2400 | b3054b3a5d28fd28 |
| results/dev/o5/ppo_metrics.csv | 876 | cb7987e2ab108430 |
| results/dev/o5/tests.csv | 1966 | 1693de22c6550216 |
| results/dev/o5/tuning.csv | 3865 | 27dce96e98e814f0 |
| results/dev/o4/placebo.csv | 1595 | 7cbcb470f4866909 |
| results/dev/o4/strata.csv | 395 | 8e58977da4f442a2 |
| results/dev/o4/tests.csv | 786 | f91c35d19e79b2ac |
| results/dev/o4/variance.csv | 116 | 3bec57878fb638e4 |
| results/dev/o4/weights.json | 300 | f93c9f25744f7f49 |
| results/tables/dev_o2_summary.md | 9210 | ecf54330f4671962 |
| results/tables/dev_o3_summary.md | 18786 | 9fe3ab6e38a095ee |
| results/tables/dev_o4_summary.md | 1506 | bf625e261498b915 |
| results/tables/dev_o5_summary.md | 4065 | d1243af955e6b9fe |
| results/paper/data_descriptor/numbers.tex | 6063 | 90d52a432dd9f2aa |
| results/paper/data_descriptor/numbers_manifest.json | 20206 | 297fc3892871901f |
