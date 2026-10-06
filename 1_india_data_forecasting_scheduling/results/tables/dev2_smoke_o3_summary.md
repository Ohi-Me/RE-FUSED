# O3 round 2 — staleness-matched dropout and PART as a decision rule

Development test FY2024-25, reported only (protocol `codes/design/06_dev2_protocol.md`, section 4).

## Variants (development test)

| tid | model | mase | rmsse | pinball | cov80 | cov90 | n |
|---|---|---|---|---|---|---|---|
| T5 | mcag_instance | 1.3793 | 1.0422 | 0.2555 | 0.8059 | 0.8919 | 14196 |
| T5 | fusion_fixed | 1.2671 | 0.956 | 0.2446 | 0.8227 | 0.9034 | 14196 |
| T5 | mcag_instance_sd | 0.9058 | 0.722 | 0.2736 | 0.782 | 0.86 | 14196 |

## Paired tests (all horizons pooled)

| tid | comparison | metric | H | mean_a | mean_b | mean_diff | lo | hi | p_a_better | n_days |
|---|---|---|---|---|---|---|---|---|---|---|
| T5 | sd vs round 1 (instance) | mae | all | 0.9059 | 1.3791 | -0.4749 | -0.6263 | -0.3228 | 7.025e-19 | 365 |
| T5 | sd vs round 1 (instance) | pinball | all | 0.2736 | 0.2555 | 0.0181 | -0.0099 | 0.0428 | 0.9798 | 365 |

## PART decision rule (N9)

Total realised development-test gain collected (MSE, scaled units; higher is better): rule +0.2462, sign_only +0.2462, always +4.9452, never +0.0000, oracle +5.0625.

Sign agreement, all cells: 4/9; cells with a clear realised gain: 3/6.

| tid | H | refinement | G_val | G_lower80 | G_upper80 | refine_rule | refine_sign | realised_gain_dev | gain_lo | gain_hi | clear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T5 | 1 | regime vs fixed | 0.0277 | 0.0192 | 0.0373 | True | True | 0.1054 | 0.0415 | 0.1759 | True |
| T5 | 1 | instance vs fixed | -0.2229 | -0.2501 | -0.1082 | False | False | -0.0384 | -0.1848 | 0.0921 | False |
| T5 | 1 | OTG vs fixed | -0.2006 | -0.1969 | -0.182 | False | False | 1.7267 | 1.2059 | 2.343 | True |
| T5 | 2 | regime vs fixed | -0.0352 | -0.0377 | 0.0006092 | False | False | 0.0491 | -0.0156 | 0.1162 | False |
| T5 | 2 | instance vs fixed | 0.1414 | 0.0713 | 0.746 | True | True | -0.0789 | -0.2233 | 0.0466 | False |
| T5 | 2 | OTG vs fixed | -0.1989 | -0.2337 | -0.2098 | False | False | 1.5741 | 1.0535 | 2.1836 | True |
| T5 | 3 | regime vs fixed | 0.0276 | 0.0195 | 0.0663 | True | True | 0.0541 | 0.0017 | 0.1164 | True |
| T5 | 3 | instance vs fixed | 0.3549 | 0.4241 | 0.8215 | True | True | 0.1656 | 0.0442 | 0.2897 | True |
| T5 | 3 | OTG vs fixed | -0.1734 | -0.2136 | -0.1794 | False | False | 1.3875 | 0.9288 | 1.9545 | True |
