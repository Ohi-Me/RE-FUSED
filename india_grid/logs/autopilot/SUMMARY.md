# Autopilot run 2026-09-21T15:20:47+05:30

Job `32123.master` on `master`, GPU `MIG-2f219700-5dc6-59eb-81ef-2bb9414d3ec0`.

## Stage records

* `c_paper`: done 2.2 s, 2026-09-21T15:20:26
* `c_supp`: done 0.3 s, 2026-09-21T15:20:29
* `c_strata`: done 1.0 s, 2026-09-21T15:20:30
* `c_integrity`: done 1.1 s, 2026-09-21T15:20:31
* `c_leakage`: done 13.3 s, 2026-09-21T15:20:46

## Files written

```
-rw------- 1 <hpc-user> <hpc-user> 1415 Sep 21 15:20 t_arms.tex
-rw------- 1 <hpc-user> <hpc-user> 3877 Sep 21 15:20 t_armtests.tex
-rw------- 1 <hpc-user> <hpc-user> 1546 Sep 21 15:20 t_assessment.tex
-rw------- 1 <hpc-user> <hpc-user> 1335 Sep 21 15:20 t_ensemble.tex
-rw------- 1 <hpc-user> <hpc-user> 3595 Sep 21 15:20 t_gating.tex
-rw------- 1 <hpc-user> <hpc-user> 3257 Sep 21 15:20 t_hyp_F1.tex
-rw------- 1 <hpc-user> <hpc-user> 2838 Sep 21 15:20 t_hyp_F2.tex
-rw------- 1 <hpc-user> <hpc-user>  948 Sep 21 15:20 t_hyp_F3.tex
-rw------- 1 <hpc-user> <hpc-user> 1715 Sep 21 15:20 t_hyp_F4.tex
-rw------- 1 <hpc-user> <hpc-user> 1039 Sep 21 15:20 t_hyp_F5.tex
-rw------- 1 <hpc-user> <hpc-user> 7887 Sep 21 15:20 t_hyp_all.tex
-rw------- 1 <hpc-user> <hpc-user> 2366 Sep 21 15:20 t_latency.tex
-rw------- 1 <hpc-user> <hpc-user> 1900 Sep 21 15:20 t_members.tex
-rw------- 1 <hpc-user> <hpc-user> 1669 Sep 21 15:20 t_members_prob.tex
-rw------- 1 <hpc-user> <hpc-user> 4411 Sep 21 15:20 t_o2tests.tex
-rw------- 1 <hpc-user> <hpc-user>  970 Sep 21 15:20 t_part.tex
-rw------- 1 <hpc-user> <hpc-user>  702 Sep 21 15:20 t_season_accuracy.tex
-rw------- 1 <hpc-user> <hpc-user> 1199 Sep 21 15:20 t_seeds.tex
-rw------- 1 <hpc-user> <hpc-user> 1250 Sep 21 15:20 t_sourceloss.tex
-rw------- 1 <hpc-user> <hpc-user>  816 Sep 21 15:20 t_state_accuracy.tex
-rw------- 1 <hpc-user> <hpc-user>  981 Sep 21 15:20 t_state_regret.tex
-rw------- 1 <hpc-user> <hpc-user> 1074 Sep 21 15:20 t_tuning.tex
-rw------- 1 <hpc-user> <hpc-user> 2045 Sep 21 15:20 t_variants.tex
-rw------- 1 <hpc-user> <hpc-user> 1124 Sep 21 15:20 tables_manifest.json
results/audit:
total 24
-rw------- 1 <hpc-user> <hpc-user>   825 Sep 21 15:20 INTEGRITY.md
-rw------- 1 <hpc-user> <hpc-user> 10980 Sep 21 15:20 integrity.json
drwx------ 2 <hpc-user> <hpc-user>  4096 Sep 20 23:53 rescore
drwx------ 2 <hpc-user> <hpc-user>  4096 Sep 20 23:53 strata

results/audit/strata:
total 28
-rw------- 1 <hpc-user> <hpc-user> 1188 Sep 21 15:20 accuracy_horizon.csv
-rw------- 1 <hpc-user> <hpc-user> 1575 Sep 21 15:20 accuracy_season.csv
-rw------- 1 <hpc-user> <hpc-user> 9513 Sep 21 15:20 accuracy_state.csv
-rw------- 1 <hpc-user> <hpc-user> 5631 Sep 21 15:20 regret_state.csv
```

## Integrity

# Integrity and reproducibility check

Run 2026-09-21T15:20:30+0530 on `master` (PBS job 32123.master), phase `confirm`.

| Check | Result |
|---|---|
| Pre-registration hash matches `PREREG_SHA256.txt` | yes |
| Frozen choice files and scorer (42 files) unchanged | yes |
| Panel content hash matches `CHECKSUM.txt` | yes (104550 rows x 107 columns) |
| Git tag `prereg-v1` present | yes (4d3807ed275342c1f83f2e7009f094e838f20bad) |
| Re-scoring the stored results reproduces `hypotheses.csv` | yes, identical |

Supported checks in the archived table: 35 of 52; in the re-scored table: 35.

The scoring script is `codes/scripts/confirm/c01_hypotheses.py`, sha256 `f07727d831f03717f351d75c1c9554dd6e1c3d4e8d9dfe8f725beeb0df5d713a`, which is the hash written into the pre-registration before the evaluation data were opened.
