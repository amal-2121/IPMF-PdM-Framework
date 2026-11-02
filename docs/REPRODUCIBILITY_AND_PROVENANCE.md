# Reproducibility and provenance notes

## Implemented in this repository release
- A conventional LSTM regression baseline (`src/ipmf/run_lstm_baseline.py`).
- Group-level train/validation split before sequence generation.
- Scaler fitted only on training engines.
- Left-padding by repeating the first observation for short test trajectories.
- Three exploratory notebooks copied from the project workspace (when available).
- CSV outputs from exploratory FD004 masking/padding/RUL-cap experiments (when available).

## Not implemented / not independently reproducible from this ZIP
- The manuscript's full Attention-LSTM + XGBoost residual-correction hybrid.
- The original ten-run seed-level records underlying the historical hybrid summary.
- A trained 1D-CNN + Attention fault classifier with genuine observed labels/predictions.
- An LSTM autoencoder with a reconstruction decoder/loss.
- Industrial edge deployment measurements or hardware logs.

## Historical result files
The two original result CSVs are retained unchanged under
`legacy_archive/original_repository/`. They contain only two rows each in the
provided archive, despite the original script listing ten seeds. Their full
ten-run provenance cannot be recovered from the supplied repository ZIP.
They must not be represented as ten-run raw data.

## Synthetic demonstration
`examples/synthetic_demo/confusion_matrix_demo.py` is explicitly synthetic.
Its output is not model performance and must not be used as an experimental
figure or evidence in the manuscript.

## Data
NASA C-MAPSS data are not redistributed here. Obtain the official dataset from
NASA and place the files in a local directory. The runner accepts standard
whitespace-delimited `.txt` files and compatible `.csv` files.
