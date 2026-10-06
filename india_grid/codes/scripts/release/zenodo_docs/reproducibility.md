# Reproducibility

There are three levels at which this work can be checked, from a few seconds to a few days of computing.

## Level 1: check the files (no computing)

Every file in the record has its SHA-256 hash in `MANIFEST.sha256`:

```bash
sha256sum -c MANIFEST.sha256        # Linux
shasum -a 256 -c MANIFEST.sha256    # macOS
```

The panel also has a content hash in `data/processed_panel/CHECKSUM.txt`. It is computed on the table as read
from the file, so it does not depend on how the Parquet file was written:

```python
import hashlib, pandas as pd
p = pd.read_parquet("data/processed_panel/refused_state_day.parquet")
print(hashlib.sha256(pd.util.hash_pandas_object(p, index=False).values.tobytes()).hexdigest())
```

## Level 2: check the pre-registration and re-score the results (minutes)

One script runs all the checks of levels 1 and 2 and writes a short report:

```bash
python code/reproducibility/release/check_zenodo_package.py . ../refused_check
```

It checks every file against `MANIFEST.sha256`, the panel against its content hash, the pre-registration and the
files it froze against their recorded hashes, and then re-scores the {n_checks} checks from the stored results and
compares them with `results/hypotheses.csv`. The same steps by hand are below.

The code expects the project layout in which it was written. This script copies the record's files back into
that layout in a new folder:

```bash
python code/reproducibility/release/rebuild_layout.py ../refused_work
cd ../refused_work
pip install -r requirements-h100.txt      # Python 3.10
```

Then:

* `codes/design/PREREG_SHA256.txt` holds the hash of `codes/design/02_preregistration.md`, and the
  pre-registration lists the hashes of the {n_frozen} frozen choice files and of the scoring script. All of them
  can be checked with `sha256sum`.
* The {n_checks} checks can be scored again from the stored results:

```bash
REFUSED_PHASE=confirm python codes/scripts/confirm/c01_hypotheses.py
```

The new `results/confirm2/hypotheses.csv` should be identical to `results/hypotheses.csv` in this record. We
ran this check ourselves after the confirmatory run, on the stored results; the report is `results/INTEGRITY.md`.

## Level 3: rerun everything (days, with a GPU)

1. Download the raw reports with the scripts in `codes/scripts/acquire/`. The manifests in
   `data/source_manifests/` let you check each file against the hash we recorded.
2. Build the panel with `python run_all.py stage build_b01_psp ... build_b08_validation`, or all stages with
   `python run_all.py local`. The content hash of the rebuilt panel should match `CHECKSUM.txt`.
3. Run the development and confirmatory stages with `run_all.py`. `python run_all.py plan` lists every stage and
   the order they run in.

We ran all training and evaluation on one NVIDIA H100 multi-instance GPU slice (about 47 GB of GPU memory) through
PBS batch jobs. The job files themselves are specific to our cluster and are not part of this record. Every stage
can be run directly with `python run_all.py stage <name>`. For a PBS cluster, `python run_all.py submit --dry-run`
writes the job scripts it would submit; the queue, memory and conda settings come from a `.env` file (see
`settings()` in `run_all.py`). The tree models run on a CPU; the deep and pretrained models need a GPU
to finish in reasonable time. Five seeds were used for every trained model, and results will match ours closely
but not bit for bit on different hardware.

### About the blinding guard

`codes/refused/guard.py` refuses to load the evaluation period unless the pre-registration hash matches and the git
tag `prereg-v1` exists. We added it to stop ourselves from looking at the evaluation data before the freeze. A copy
of this record has no git history, so before rerunning the confirmatory stages run `python run_all.py stage
c_freeze`: it checks the pre-registration against its recorded hash, creates a git repository, commits the code and
tags it `prereg-v1`. The confirmatory stages (`c_...`) then run with `REFUSED_CONFIRMATORY=1`, which `run_all.py`
sets for them. For us the order mattered; for someone checking the work afterwards it does not.

## What is not in this record

The record holds what is needed to check and rerun the study. Some files we used during the work are kept in our
own archive and not here: the model forecasts themselves (about 4 GB), the training logs of each model, the scripts
that typeset our manuscripts, the rehearsal run we made before the freeze, the script that filled the tables of
the pre-registration draft, the draft itself, and our cluster job files. The two stages that use them
(`prereg_fill` and `c_rehearsal`) are therefore not runnable from this record; neither is needed for the checks
above or for rerunning the models.

## Software

Python 3.10 with the versions in `requirements-h100.txt`. PyTorch 2.5.1 with CUDA 12.1 is installed first and
`neuralforecast` afterwards without dependencies, as `code/reproducibility/environment/setup_env.sh` does
(`rebuild_layout.py` copies it to `hpc/`, where the driver expects it). The pretrained Chronos models are
downloaded from Hugging Face on first use.
