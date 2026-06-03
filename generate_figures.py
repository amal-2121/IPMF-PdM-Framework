# generate_figures.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ========== CONFIGURATION ==========
DATA_PATH = r"C:\Users\twin\Downloads\archive"
SUBSET = "FD001"
SEQ_LEN = 30
MAX_RUL = 125
SEED = 42   # use the same seed as one of your runs

# ========== LOAD DATA (same as before) ==========
col_names = ['unit_number', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
train_df = pd.read_csv(os.path.join(DATA_PATH, f"train_{SUBSET}.csv"), header=0)
test_df = pd.read_csv(os.path.join(DATA_PATH, f"test_{SUBSET}.csv"), header=0)
rul_test = pd.read_csv(os.path.join(DATA_PATH, f"RUL_{SUBSET}.csv"), header=0)
train_df.columns = col_names
test_df.columns = col_names
rul_test.columns = ['rul']

for col in train_df.columns:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
train_df = train_df.dropna()
test_df = test_df.dropna()

# Remove constant sensors
constant = [col for col in train_df.columns if train_df[col].std() == 0]
train_df = train_df.drop(columns=constant)
test_df = test_df.drop(columns=constant)

# Compute RUL for training
max_cycles = train_df.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycles']
train_df = train_df.merge(max_cycles, on='unit_number')
train_df['RUL'] = train_df['max_cycles'] - train_df['time_cycles']
train_df = train_df.drop(columns=['max_cycles'])
train_df['RUL'] = train_df['RUL'].clip(upper=MAX_RUL)

# Feature columns
feature_cols = [col for col in train_df.columns if col not in ['unit_number', 'time_cycles', 'RUL']]

# Standardize
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Create sequences
def create_sequences(data, feat_cols, seq_len):
    X, y = [], []
    grouped = data.groupby('unit_number')
    for _, group in grouped:
        group = group.sort_values('time_cycles')
        seqs = group[feat_cols].values
        targets = group['RUL'].values
        if len(seqs) <= seq_len:
            continue
        for i in range(len(seqs) - seq_len):
            X.append(seqs[i:i+seq_len])
            y.append(targets[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def prepare_test_data(test_df, rul_df, feat_cols, seq_len):
    X_test, y_true = [], []
    grouped = test_df.groupby('unit_number')
    for unit_id, group in grouped:
        group = group.sort_values('time_cycles')
        seqs = group[feat_cols].values
        if len(seqs) >= seq_len:
            X_test.append(seqs[-seq_len:])
            y_true.append(rul_df.iloc[unit_id-1, 0])
    X_test = np.array(X_test, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    y_true = np.clip(y_true, 0, MAX_RUL)
    return X_test, y_true

X_train, y_train = create_sequences(train_df, feature_cols, SEQ_LEN)
X_test, y_test = prepare_test_data(test_df, rul_test, feature_cols, SEQ_LEN)

# Build model (same as your working IPMF model – simple LSTM, not attention+XGBoost)
# Because that's what gave you the 11.67 MAE and the history
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Set seed
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Train with history
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred = model.predict(X_test, verbose=0)
y_pred = np.clip(y_pred, 0, MAX_RUL)

# ========== PLOT 1: Loss curve ==========
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss (FD001)')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
print("Saved loss_curve.png")

# ========== PLOT 2: Scatter plot ==========
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, MAX_RUL], [0, MAX_RUL], 'r--', label='Perfect Prediction')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('Predicted vs True Remaining Useful Life (FD001)')
plt.legend()
plt.grid(True)
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
print("Saved scatter_plot.png")

print("Done! Both figures saved in the current directory.")