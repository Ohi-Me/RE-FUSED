# Methodology

This is a short description of what we did, in the order we did it. The papers give the full details; this file is
meant to help someone find their way through the code and the results.

## 1. Building the panel

We downloaded the official reports (see `data_sources.md`), parsed each source into an interim table at its own
resolution, and joined them into one table with one row per State control area and day. The build scripts are
`code/preprocessing/build/b01` to `b08`: one parser per source, then the join (`b07`), then the validation numbers
(`b08`). The build writes a content hash of the finished panel. We rebuilt the panel from the raw files on a
second machine and got the same hash.

## 2. Respecting publication delays

For every column we measured how many days after the data day its value becomes public. A forecast issued at noon
on day t may only use a column up to day t minus its delay. The same rule builds the training rows, so the models
are trained on the same kind of information they would have in operation. A leakage test
(`code/reproducibility/tests/test_leakage.py`) changes values that should not be visible and checks that the
features do not move.

## 3. What we forecast

Five targets, each one to three days ahead:

1. energy met by each State
2. actual drawal of each State from the grid
3. conventional generation of each State
4. renewable generation of each State
5. the day-ahead market price of each bid area

Accuracy is measured with MASE against a seasonal naive forecast, and uncertainty with the pinball loss and the
coverage of 80 % and 90 % intervals.

## 4. Periods

* Training: FY2018-19 to FY2022-23 in development, one year more in the final run.
* Calibration year: FY2023-24 in development, FY2024-25 in the final run. All choices were made here.
* Development test: FY2024-25, used only during development.
* Confirmatory evaluation: 1 April 2025 to 31 August 2026. This period stayed locked in the code until the
  pre-registration was frozen, and it was scored once.

## 5. Models and combination

We trained gradient-boosted trees (LightGBM, XGBoost), recurrent and attention models (BiLSTM, TFT, PatchTST),
multilayer and convolutional models with covariates (N-HiTS, TiDE, BiTCN, N-BEATSx), and used two pretrained models
(Chronos-Bolt and Chronos-2). Trained models use five seeds. The forecasts are then combined in three ways: an
equal-weight average of the best few, stacked weights fitted on the calibration year, and online weights that are
updated only from losses that have already been published. The final forecaster of each target was chosen on the
calibration year and frozen. Intervals are calibrated with rolling conformal shifts.

Code: `code/forecasting/o2` (first round) and `code/forecasting/dev2` (second round and combination).

## 6. Context gating and the carbon layer

LA-MCAG gives each block of context (market, conventional supply, renewables, system state, weather, carbon) its
own encoder and a gate that sees how old that block's data are. We compare it with a fixed-fusion model of the same
size, test real context against null and shuffled context, test what happens when two sources disappear, and test
whether a carbon-intensity block helps. Code: `code/forecasting/o3`.

## 7. Deviation assessment

We built a deviation measure that adds ex-ante context (renewable ramps, system stress, carbon intensity and forecast
uncertainty) and splits each deviation into the part that was visible when the schedule was set and the surprise
part. It is compared with the current size-times-rate measure by how well it orders observed consequences within
groups of similar-sized deviations. Code: `code/analysis/o4` and `code/forecasting/dev2/d2_08_o4.py`.

## 8. Day-ahead scheduling

Using the calibrated forecast of each State's drawal, we compare scheduling rules: the forecast median, a fixed
risk level, a risk level that depends on uncertainty and system stress, joint scheduling of all States against the
system-wide tail, and a policy learned with PPO. All of them respect the State's historical schedule range and ramp
limit. They are scored by the regret under a simplified deviation settlement. We also test whether a better
forecast, with the decision rule held fixed, lowers the regret. Code: `code/analysis/o5` and
`code/forecasting/dev2/d2_09_o5.py`, `d2_11_ppo.py`.

## 9. Pre-registration and scoring

Before opening the evaluation period we wrote down {n_checks} checks with their decision rules
(`code/reproducibility/design/02_preregistration.md`), hashed the file, the {n_frozen} frozen choice files and the
scoring script, and tagged the code. The scoring script (`code/analysis/confirm/c01_hypotheses.py`) then produced
`results/hypotheses.csv`. Comparisons use Diebold-Mariano tests with the HLN correction, seven-day block bootstrap
intervals, Holm correction within each family of checks and Benjamini-Yekutieli correction across families.
Changes made after the freeze are listed in `code/reproducibility/design/07_confirm_deviations.md`.

## 10. From results to papers

Every number, table and figure in our papers is written by a script from the result files
(`code/analysis/paper`). A check fails the build if a paper uses a number that no script produced. The generated
files are in `results/tables/` and `figures/`.
