# IPMF – Intelligent Predictive Maintenance Framework

Code accompanying the paper: *"An Intelligent Predictive Maintenance Framework Using Machine Learning for Smart Manufacturing Environments"*

## Results on C-MAPSS FD001
- MAE: 11.67 ± 0.20
- RMSE: 16.12 ± 0.58

## How to run
1. Install Python 3.12 and create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python run_ipmf.py`

## Repository structure
- `run_ipmf.py` – main training and evaluation script (10 seeds)
- `generate_figures.py` – produces loss curve and scatter plot
- `generate_fig1.py` – produces architecture diagram
- `FD001_results.csv` – detailed results per seed

## License
Apache 2.0

## Revised Attention-LSTM + XGBoost evaluation (five seeds)

A separate, reproducible evaluation is provided under `revised_evaluation/`. It compares the Attention-LSTM backbone with the hybrid residual-correction model on C-MAPSS FD001 and FD004 using seeds 42, 7, 21, 100, and 2025. The original files and historical results above are retained unchanged; these are different experiment versions and should not be conflated.

### Five-seed mean ± standard deviation

| Dataset | Model | MAE | RMSE | NASA score |
|---|---|---:|---:|---:|
| FD001 | Attention-LSTM | 12.9165 ± 0.7114 | 16.9436 ± 0.7921 | 507.2491 ± 85.7300 |
| FD001 | Hybrid | 13.0481 ± 0.6833 | 16.9476 ± 0.8599 | 491.7130 ± 88.2721 |
| FD004 | Attention-LSTM | 25.0319 ± 0.9279 | 32.5672 ± 1.1125 | 14811.8207 ± 3022.1379 |
| FD004 | Hybrid | 24.7889 ± 0.9274 | 32.0696 ± 0.7322 | 13308.3017 ± 2679.5639 |

The five-seed comparison does not establish a statistically significant paired MAE improvement for the hybrid on either subset. See the evaluation folder for per-seed results, alpha-selection details, code, and protocol notes.
