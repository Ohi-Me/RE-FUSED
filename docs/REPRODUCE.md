# How to rerun the work

All results in this repository were produced on an NVIDIA H100 cluster (one MIG instance and at most 8 CPU cores
per job, PBS scheduler). Any Linux machine with a CUDA GPU will do; without a GPU the neural models are slow.

## 1. Environment

```bash
conda create -n refused python=3.10 -y && conda activate refused
pip install -r India_Grid_Study/requirements-h100.txt -c India_Grid_Study/hpc/constraints.txt
pip install -r Forecast_Correction_India/env/requirements-lock.txt -c Forecast_Correction_India/env/constraints.txt
```

The constraints files stop `neuralforecast` and `chronos-forecasting` from replacing the CUDA build of PyTorch.

## 2. Data

```bash
cd India_Grid_Study
python codes/scripts/acquire/grid_india.py psp --from 2017-18 --to 2026-27   # Daily PSP reports
python codes/scripts/acquire/grid_india.py dsm                               # DSM rate and price files
python codes/scripts/acquire/npp.py --start 2018-03-22 --end 2026-09-11      # CEA generation reports
python codes/scripts/acquire/cea.py re                                       # CEA renewable reports
python codes/scripts/acquire/cea.py co2                                      # CEA CO2 database
python codes/scripts/acquire/imd.py --years 2017-2025                        # IMD gridded weather
python run_all.py stage build_b01_psp build_b02_dsm build_b03_npp build_b04_re build_b05_co2 \
    build_b06_imd build_b07_panel build_b08_validation build_verify   # compares with data/processed/CHECKSUM.txt
python codes/scripts/build/b09_psp_timeseries.py   # all-India 15-minute and hourly series
python codes/scripts/build/b10_hourly_prices.py    # hourly national market prices
```

Each downloaded file is checked against the SHA-256 in its source's `MANIFEST.csv`. An agency can revise or
remove a file; such a file then shows up as a mismatch in the build log.

If you only want to check the results, skip this step: the processed data are already in
`India_Grid_Study/data/processed/`.

## 3. The India programme (forecasting, assessment, scheduling)

```bash
cd India_Grid_Study
python run_all.py status        # list of stages and which are done
python run_all.py submit        # on a PBS cluster: every job, with dependencies
python run_all.py local         # anywhere else: all stages one after another
```

The confirmatory stages refuse to run unless the pre-registration hash, its git tag and `REFUSED_CONFIRMATORY=1`
all match. `codes/scripts/audit/` rescoring reproduces the hypothesis table from the stored predictions.

## 4. The forecast-correction study on Indian data

```bash
cd Forecast_Correction_India
qsub -o logs/india_dev.out hpc/india_dev.pbs           # development
qsub -o logs/india_daily.out hpc/india_daily.pbs       # State daily ladders (exploratory)
qsub -o logs/india_confirm.out hpc/india_confirm.pbs   # the pre-registered run
```

The pre-registered run needs `REFUSED_INDIA_CONFIRMATORY=1` and the tag `prereg-india-v1`; the job file sets the
flag. Its verdicts are written to `results/india/confirm/analysis/verdicts.json`.

## 5. What will and will not match exactly

- Panel content hash, gradient boosting and every statistic computed from stored predictions: exact.
- GPU-trained neural networks: not bit-for-bit reproducible, so retrained forecasts differ in the last digits.
  Reported results are means over seeds.
- Changing the number of CPU threads can change the last digits of floating-point sums.
