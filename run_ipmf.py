# -*- coding: utf-8 -*-
# run_ipmf.py - Corrected for CSV files with header and numeric conversion

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r"C:\Users\twin\Downloads\archive"
SUBSET = "FD004"          # Change to "FD004" for the second experiment
SEQ_LEN = 30
MAX_RUL = 125
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]   # 10 seeds

# ============================================================
# FUNCTIONS
# ============================================================
def load_data(subset, data_path):
    """Load train, test and RUL dataframes from CSV files (with header)."""
    col_names = ['unit_number', 'time_cycles'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)]
    
    train_file = os.path.join(data_path, f"train_{subset}.csv")
    test_file = os.path.join(data_path, f"test_{subset}.csv")
    rul_file = os.path.join(data_path, f"RUL_{subset}.csv")
    
    # Read with header=0 because the CSV files have column names in first row
    train_df = pd.read_csv(train_file, header=0)
    test_df = pd.read_csv(test_file, header=0)
    rul_test = pd.read_csv(rul_file, header=0)
    
    # Rename columns to match expected names (the CSV headers might differ)
    train_df.columns = col_names
    test_df.columns = col_names
    rul_test.columns = ['rul']
    
    # Convert all columns to numeric (coerce errors to NaN, then drop NaNs if any)
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    for col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Drop any rows with NaN (should not happen if data is clean)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    return train_df, test_df, rul_test

def remove_constant_sensors(df):
    """Drop columns with zero standard deviation (must be numeric)."""
    constant = [col for col in df.columns if df[col].std() == 0]
    return df.drop(columns=constant), constant

def compute_rul(df, max_rul_cap=MAX_RUL):
    """Add RUL column (remaining useful life) to training dataframe."""
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    df = df.drop(columns=['max_cycles'])
    df['RUL'] = df['RUL'].clip(upper=max_rul_cap)
    return df

def create_sequences(data, feature_cols, seq_length):
    """Create sliding windows for LSTM training."""
    X, y = [], []
    grouped = data.groupby('unit_number')
    for _, group in grouped:
        group = group.sort_values('time_cycles')
        seqs = group[feature_cols].values
        targets = group['RUL'].values
        if len(seqs) <= seq_length:
            continue
        for i in range(len(seqs) - seq_length):
            X.append(seqs[i:i+seq_length])
            y.append(targets[i+seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def prepare_test_data(test_df, rul_df, feature_cols, seq_length):
    """Create test sequences (last window of each engine) and true RUL."""
    X_test = []
    y_true = []
    grouped = test_df.groupby('unit_number')
    for unit_id, group in grouped:
        group = group.sort_values('time_cycles')
        seqs = group[feature_cols].values
        if len(seqs) >= seq_length:
            X_test.append(seqs[-seq_length:])
            y_true.append(rul_df.iloc[unit_id-1, 0])
    X_test = np.array(X_test, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    y_true = np.clip(y_true, 0, MAX_RUL)
    return X_test, y_true

def build_model(input_shape):
    """Create the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def run_single_seed(seed):
    """Train and evaluate model for one random seed."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    train_df, test_df, rul_test = load_data(SUBSET, DATA_PATH)
    
    # Remove constant sensors
    train_df, const_cols = remove_constant_sensors(train_df)
    test_df = test_df.drop(columns=const_cols)
    
    # Compute RUL for training
    train_df = compute_rul(train_df, MAX_RUL)
    
    # Feature columns (exclude identifiers and RUL)
    feature_cols = [col for col in train_df.columns 
                    if col not in ['unit_number', 'time_cycles', 'RUL']]
    
    # Standardize features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols, SEQ_LEN)
    X_test, y_test = prepare_test_data(test_df, rul_test, feature_cols, SEQ_LEN)
    
    # Build and train model
    model = build_model((SEQ_LEN, X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_split=0.2,
              epochs=100,
              batch_size=256,
              callbacks=[early_stop],
              verbose=0)
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred = np.clip(y_pred, 0, MAX_RUL)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse}

# ============================================================
# MAIN LOOP
# ============================================================
if __name__ == "__main__":
    print(f"Running experiments for {SUBSET} with {len(SEEDS)} seeds...")
    print(f"Data path: {DATA_PATH}")
    print("-" * 50)
    
    results = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"Seed {i:2d}/{len(SEEDS)} (value={seed})...", end=' ', flush=True)
        res = run_single_seed(seed)
        results.append(res)
        print(f"MAE = {res['MAE']:.2f}, RMSE = {res['RMSE']:.2f}")
    
    mae_vals = [r['MAE'] for r in results]
    rmse_vals = [r['RMSE'] for r in results]
    
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS FOR {SUBSET}")
    print(f"MAE  = {np.mean(mae_vals):.2f} ± {np.std(mae_vals):.2f}")
    print(f"RMSE = {np.mean(rmse_vals):.2f} ± {np.std(rmse_vals):.2f}")
    print("=" * 50)
    
    out_df = pd.DataFrame(results)
    out_df.insert(0, 'seed', SEEDS)
    csv_file = f"{SUBSET}_results.csv"
    out_df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {os.path.abspath(csv_file)}")