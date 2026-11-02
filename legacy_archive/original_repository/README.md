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
