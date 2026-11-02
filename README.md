# IPMF-PdM-Framework — reproducibility-focused repository

This repository contains a transparent conventional LSTM baseline and the
available exploratory notebooks/results for NASA C-MAPSS FD001/FD004.

> **Important scope statement:** The runnable Python baseline in this release is
> a conventional two-layer LSTM. It is **not** the complete hybrid
> Attention-LSTM + XGBoost residual-correction model described in the manuscript.
> The original repository did not contain an executable implementation of that
> hybrid architecture. This release does not fabricate the missing implementation
> or reconstruct missing historical seed-level results.

## Repository layout

```text
src/ipmf/run_lstm_baseline.py             Reproducible conventional LSTM baseline
experiments/notebooks/                    Available data-check and exploratory notebooks
experiments/results/                      Available exploratory CSV outputs
examples/synthetic_demo/                  Clearly labelled synthetic illustration only
legacy_archive/original_repository/       Original uploaded repository files, unchanged
docs/REPRODUCIBILITY_AND_PROVENANCE.md    Scope, provenance, and limitations
outputs/                                  Generated results (not committed by default)
```

## Environment

Recommended: Python 3.11 or 3.12. TensorFlow 2.16.1 supports Python 3.12.
Install:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Data preparation

Download the official NASA C-MAPSS dataset independently. The runner expects
one of the following file sets inside a data directory:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

or the corresponding `FD004` files. Standard NASA whitespace-delimited files
are supported. CSV files with 26 columns are also accepted.

## Run the conventional LSTM baseline

From the repository root:

```bash
python src/ipmf/run_lstm_baseline.py --data-dir /path/to/CMAPSS --subset FD001 --seed 42
python src/ipmf/run_lstm_baseline.py --data-dir /path/to/CMAPSS --subset FD004 --seed 42
```

On Windows, quote the path if it contains spaces:

```powershell
python src/ipmf/run_lstm_baseline.py --data-dir "C:\Users\YourName\CMAPSS" --subset FD004 --seed 42
```

The script:
- splits training engines into train/validation groups before creating windows;
- fits the scaler on training engines only;
- uses the official test RUL file;
- retains short test trajectories by left-padding with the first observed row;
- reports the number of short test engines and all evaluated test engines;
- saves one metrics CSV and one per-engine prediction CSV under `outputs/`.

By default, official test RUL targets are not clipped. Add
`--clip-test-targets` only when reproducing a protocol that explicitly clips
test targets to the training RUL cap. Record this choice when reporting metrics.

## Exploratory notebooks and CSV files

The notebooks in `experiments/notebooks/` cover:
- FD001/FD004 data checks;
- padding/masking sensitivity;
- FD004 multi-seed validation and short-trajectory prediction stability.

The associated CSVs in `experiments/results/` are exploratory outputs. Read each
notebook's markdown and code cells to understand its exact preprocessing,
target cap, seeds, and whether test labels were accessed. Do not treat
prediction-stability statistics as test accuracy.

## Historical files and synthetic illustration

Original scripts and the two original result CSVs are preserved unchanged in
`legacy_archive/original_repository/`. The original CSVs contain two records
each, not ten. They are retained for provenance, not certified as complete
raw seed-level evidence.

The confusion-matrix generator has been moved to
`examples/synthetic_demo/` and renamed/documented as synthetic. It does not
train a classifier or produce observed experimental results. Do not use its
figure as a measured confusion matrix.

## What is still required for a full manuscript reproduction

A faithful reproduction of the manuscript's main hybrid results requires:
1. the exact Attention-LSTM + XGBoost residual-correction implementation;
2. the original preprocessing, split, target, and training configuration;
3. original per-seed predictions/metrics or reruns from the documented protocol;
4. a real fault-classification dataset, labels, classifier training code, and
   held-out predictions for a genuine confusion matrix;
5. exact scripts/configuration for any autoencoder, scheduler, and edge-device
   experiments claimed in the paper.

Until those artifacts are supplied and verified, this repository should be
described as a baseline plus exploratory-reimplementation repository, not as a
complete reproduction package for every manuscript result.

## Citation and data provenance

C-MAPSS data are provided by NASA. Please cite the dataset/source as required by
the official distribution and cite the relevant manuscript separately.
