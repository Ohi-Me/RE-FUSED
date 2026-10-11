# Exploratory: gated correction of the States' own drawal schedules (not pre-registered)

Requested by the ECM review (10 Oct 2026). Script `code/scripts/india/explore_schedule_correction.py`, H100 job
34921 (10 Oct 2026, 19:38 IST, about 3 minutes), log `logs/explore_schedule.out`. The cluster checkout was at
commit 997787d with this script copied in; nothing else differed. The State-panel test period (1 Apr 2025 to
31 Aug 2026) had already been opened by the pre-registered studies, so this is exploratory by construction. No
pre-registered result is changed.

## Design

* Data: the official State-day panel, 34 control areas; target = actual drawal X (GWh/day).
* The panel's `dev_drawal_schedule_gwh` is the recorded schedule S: X - S equals the published deviation
  `dev_od_ud_gwh` exactly (checked on 100,160 State-days). Its description in `FIELD_REGISTRY.csv` repeats the
  deviation text and should be corrected.
* Grid-India publishes only the final schedule, which includes revisions made during the day. Two baselines:
    persistence rule).
  * `sched_final`: S of day t (NOT available day-ahead; a ceiling that asks whether even the revised schedule has a
    correctable error).
  * `snaive7`: the seasonal naive used in the pre-registered State-panel ladders, for reference.
* Corrector features: lags of X and of the schedule error X - S at 2, 3, 7, 14 days, a 7-day mean of X - S ending at
  t-2, the t-2 schedule, calendar terms and an area code. Same engine and settings as the earlier ladders
  (`ladder.run_ladder`, three seeds), split train < Apr 2024, gate fitting FY2024-25, test Apr 2025 - Aug 2026.
* Metric: deviation energy sum |X - F| over areas, GWh per year, mean of three seeds; differences with 7-day
  moving-block bootstrap 95 % intervals (2,000 resamples) against no correction and against one gate per area.

## Results (`summary.csv`)

| Baseline | Uncorrected (GWh/yr) | One gate per area: change | Finer or time-adaptive designs against one gate per area |
|---|---|---|---|
| `sched_persist` (operational) | 82,835 | -5,185 [-6,198, -4,172] (6.3 %, squared-error skill 0.14) | only area x weekend better, -259 [-446, -96]; expanding gate -117 [-235, +0.4]; per-instance, rolling and forgetting gates not distinguishable or worse; Fixed Share worse, +1,787 [+1,059, +2,433] |
| `sched_final` (not operational) | 16,416 | -4,642 [-4,998, -4,313] (28 %, skill 0.47) | time-adaptive gates slightly better (-75 to -166; e.g. rolling 90 d -166 [-271, -67]); per-instance +255 [+171, +333]; Fixed Share +302 [+232, +384] |
| `snaive7` (reference) | 119,296 | -37,678 [-43,211, -32,657] (32 %) | none clearly better; per-instance +883 [+212, +1,574]; Fixed Share +7,195 [+5,895, +8,436] |

## Reading

* The pattern of the paper holds on the States' own schedules: one coarse gate removes most of what correction can
  remove, finer gates add little, and Fixed Share (the design that helped weekly persistence on the hourly series)
  adds error here.
* Even the final, revised schedules carry a correctable error: a gate on a corrector that sees only errors published
  two days earlier removes 28 % of their deviation energy. This points to persistent schedule bias. Because the final
  schedule is not known day-ahead, this is not an operational saving.
* Limits: daily net energy, not the 15-minute blocks that are settled; schedules include revisions; one corrector
  family; exploratory.
