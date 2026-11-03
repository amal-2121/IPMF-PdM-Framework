# Revised Attention-LSTM + XGBoost evaluation

This folder contains the revised five-seed comparison for NASA C-MAPSS FD001 and FD004. It is kept separate from the repository's historical IPMF scripts and reported results because the model and experiment protocol are not identical.

## Contents

- `attention_lstm_xgboost.py`: transparent implementation of an Attention-LSTM backbone with XGBoost residual correction.
- `final_metrics_by_seed.csv`: per-seed metrics for Attention-LSTM and Hybrid.
- `final_metrics_summary.csv`: mean and standard deviation across five seeds.
- `final_alpha_summary_by_seed.csv`: selected correction coefficient and calibration-group counts.

## Evaluation summary

Seeds: 42, 7, 21, 100, 2025. Sequence length: 30 cycles. Training RUL target cap: 125. Official test RUL labels were not clipped in the reported evaluation. Short FD004 test trajectories were padded by repeating their first observed row.

| Dataset | Model | MAE (mean ± SD) | RMSE (mean ± SD) | NASA score (mean ± SD) |
|---|---|---:|---:|---:|
| FD001 | Attention-LSTM | 12.9165 ± 0.7114 | 16.9436 ± 0.7921 | 507.2491 ± 85.7300 |
| FD001 | Hybrid | 13.0481 ± 0.6833 | 16.9476 ± 0.8599 | 491.7130 ± 88.2721 |
| FD004 | Attention-LSTM | 25.0319 ± 0.9279 | 32.5672 ± 1.1125 | 14811.8207 ± 3022.1379 |
| FD004 | Hybrid | 24.7889 ± 0.9274 | 32.0696 ± 0.7322 | 13308.3017 ± 2679.5639 |

The five-seed paired comparison does not show a statistically significant MAE difference between Hybrid and Attention-LSTM on either subset. The hybrid's MAE is slightly higher on FD001 and slightly lower on FD004 in the aggregate.

## Method outline

1. Split training engines into fit, validation, and calibration groups (70/15/15 for FD001; 174/37/38 for FD004 in the recorded runs).
2. Fit the scaler and Attention-LSTM using fit engines, with validation engines used for early stopping.
3. Generate out-of-sample base predictions on held-out calibration engines.
4. Train XGBoost on residuals from one calibration-engine partition; choose alpha on a separate calibration-engine partition.
5. Apply the selected residual correction to the official test trajectories and calculate MAE, RMSE, and NASA score.

This is a transparent implementation and is not claimed to be byte-for-byte or protocol-identical to earlier unpublished IPMF code.

## Reproduction

Install Python packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, and `xgboost`.

Place the NASA C-MAPSS files in a data directory using names such as `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` (or the FD004 equivalents). CSV variants are also accepted.

Example:

```bash
python attention_lstm_xgboost.py --data-dir /path/to/CMAPSSData --subset FD001 --seed 42 --epochs 60 --patience 8 --output-dir outputs
```

Repeat for seeds `42 7 21 100 2025` and subsets `FD001 FD004`. The recorded five-seed runs used 60 epochs maximum, batch size 256, patience 8, learning rate 0.001, 500 XGBoost estimators, max depth 3, and XGBoost learning rate 0.03. Actual epochs trained varied with early stopping. The script's general CLI defaults differ for epochs and patience, so pass the recorded values explicitly to reproduce this run configuration.
