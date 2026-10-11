# Protocol: source fusion for day-ahead scheduling (TPWRS follow-up 2), version 1

Status: STOPPED AFTER DEVELOPMENT (11 Oct 2026, job 35030; no variant beat G0, see results/extra/SOURCE_FUSION_README.md), so it was not frozen and the evaluation period was not used. Written 11 Oct 2026 before any run of `codes/scripts/extra/x09_source_fusion.py`. It is frozen (hashed
into `SOURCE_FUSION_SHA256.txt` and committed) after the development run and before the one run on the evaluation
period; the script refuses that period until the hash matches. Nothing here changes the pre-registered results of

## 1. Question and gap

level. They used one information source, the calibrated forecast distribution. The State's own recorded schedule
(H_pub, day t-2) is a second source: it carries what the State knows about its contracts and plans. In the
development year (FY2024-25) the unprojected recorded schedule had lower settlement regret than every forecast rule; in
the evaluation period the forecast rules were better. Two sources whose ranking changes over time are the textbook case
for forecast combination (Bates and Granger 1969; Timmermann 2006) and for online aggregation with Fixed Share
(Herbster and Warmuth 1998). The question: does blending the two schedules lower settlement regret below the better
single source?

## 2. Model

S_{n,t} = w_{n,t} A2_{n,t} + (1 - w_{n,t}) H_{n,t}, with A2 the paper's frozen settlement-aware rule before projection
and H the recorded schedule of day t-2. Scored projected onto the schedule limits and not projected; each version tuned
on its own objective.

| Id | Weight |
|---|---|
| G0 | 0 or 1: the source with lower regret on the fitting year |
| G1 | one weight, grid 0:0.05:1, chosen on the fitting year |
| G2 | one weight per State, same grid, chosen on the fitting year |
| G3 | online per State: Fixed Share over 11 blends (w = 0:0.1:1), loss of day t known at t+2, losses scaled by the State's mean fitting-year loss; eta in {0.1, 0.3, 1, 3, 10} and share in {0, 0.01, 0.05, 0.2} chosen on the fitting year |
| G4 | logistic w(z) on the follow-up-1 features plus the 7- and 28-day mean of (regret of H minus regret of A2), lagged two days; trained on the fitting-year settlement (last 20 % of fitting days for early stopping), 3 seeds averaged |

## 3. Data and evaluation

Development: fit FY2023-24, score FY2024-25. Frozen run: fit FY2024-25, score 1 Apr 2025 - 29 Aug 2026 (exploratory:
this period was opened by the pre-registered study). Settlements: stylised and DSM-closer (dsm_nominal).

Primary comparison: each of G1-G4 against G0, mean system regret per day, paired 7-day block bootstrap (2,000
resamples), one-sided HLN p, Holm over the four, separately per settlement and projection. A variant "beats" G0 only
if p_Holm < 0.05 and the interval excludes 0. Secondary: each variant against A2 and against H_pub; CVaR90 and J.

## 4. What would count against the idea

If no variant beats G0 in the frozen run, blending the State's schedule with the forecast rule does not pay, and this
is reported. If a variant beats G0 in development but not in the frozen run, it is reported as not replicated.
