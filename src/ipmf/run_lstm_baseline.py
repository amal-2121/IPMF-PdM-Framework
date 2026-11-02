#!/usr/bin/env python3
"""
Reproducible conventional LSTM baseline for NASA C-MAPSS FD001/FD004.

This is NOT the Attention-LSTM + XGBoost hybrid IPMF implementation.
It is a transparent baseline/reimplementation utility. It:
  * reads standard NASA whitespace-delimited .txt files (and compatible CSVs),
  * splits training engines before window generation to avoid engine leakage,
  * fits the scaler on training engines only,
  * left-pads short test trajectories by repeating their first observed row,
  * evaluates every test engine, including trajectories shorter than seq_len.

Expected files in --data-dir:
  train_FD001.txt, test_FD001.txt, RUL_FD001.txt
(or corresponding FD004 files). CSV variants are also accepted.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

COLS = ["unit_number", "time_cycles"] + [f"op_setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install requirements.txt first.") from exc

def read_table(path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Read NASA .txt whitespace files or CSV files, with or without headers."""
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    if path.suffix.lower() == ".txt":
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] > 26:  # NASA text files often have trailing empty columns
            df = df.iloc[:, :26]
        if expected_columns is not None:
            if df.shape[1] != len(expected_columns):
                raise ValueError(f"{path.name}: expected {len(expected_columns)} columns, got {df.shape[1]}")
            df.columns = expected_columns
        return df
    # CSV: first try a header; if its width does not match, retry headerless.
    df = pd.read_csv(path)
    if expected_columns is not None and df.shape[1] != len(expected_columns):
        df = pd.read_csv(path, header=None)
    if expected_columns is not None:
        if df.shape[1] != len(expected_columns):
            raise ValueError(f"{path.name}: expected {len(expected_columns)} columns, got {df.shape[1]}")
        df.columns = expected_columns
    return df

def load_cmapss(data_dir: Path, subset: str):
    train = read_table(data_dir / f"train_{subset}.txt", COLS) if (data_dir / f"train_{subset}.txt").exists() else read_table(data_dir / f"train_{subset}.csv", COLS)
    test = read_table(data_dir / f"test_{subset}.txt", COLS) if (data_dir / f"test_{subset}.txt").exists() else read_table(data_dir / f"test_{subset}.csv", COLS)
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not rul_path.exists():
        rul_path = data_dir / f"RUL_{subset}.csv"
    rul = read_table(rul_path)
    if rul.shape[1] != 1:
        # Official RUL files are one-column; tolerate whitespace files read as one column.
        rul = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_values = pd.to_numeric(rul.iloc[:, 0], errors="raise").to_numpy(dtype=np.float32)
    for frame in (train, test):
        for c in COLS:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
        if frame.isna().any().any():
            raise ValueError(f"Missing values detected in {subset} input data.")
    return train, test, rul_values

def add_train_rul(train: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    out = train.copy()
    terminal = out.groupby("unit_number")["time_cycles"].transform("max")
    out["RUL"] = terminal - out["time_cycles"]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out

def select_features(train: pd.DataFrame):
    candidates = [c for c in COLS[2:] if train[c].nunique(dropna=False) > 1]
    return candidates

def make_train_windows(frame: pd.DataFrame, feature_cols: list[str], seq_len: int):
    xs, ys, groups = [], [], []
    for unit, g in frame.groupby("unit_number", sort=True):
        g = g.sort_values("time_cycles")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        target = g["RUL"].to_numpy(dtype=np.float32)
        if len(arr) < seq_len:
            continue
        for end in range(seq_len, len(arr) + 1):
            xs.append(arr[end-seq_len:end])
            ys.append(target[end-1])
            groups.append(int(unit))
    if not xs:
        raise ValueError("No training windows were generated. Check sequence length and data.")
    return np.stack(xs), np.asarray(ys), np.asarray(groups)

def make_test_windows(frame: pd.DataFrame, feature_cols: list[str], rul: np.ndarray, seq_len: int):
    units = sorted(frame["unit_number"].unique())
    if len(units) != len(rul):
        raise ValueError(f"Test engines ({len(units)}) do not match RUL labels ({len(rul)}).")
    xs, ys = [], []
    for unit, true_rul in zip(units, rul):
        g = frame.loc[frame.unit_number == unit].sort_values("time_cycles")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        if len(arr) == 0:
            raise ValueError(f"Engine {unit} has no observations.")
        # Repeat the first observed row on the left; retains all engines without a zero-mask ambiguity.
        if len(arr) < seq_len:
            pad = np.repeat(arr[[0]], seq_len-len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        else:
            arr = arr[-seq_len:]
        xs.append(arr)
        ys.append(float(true_rul))
    return np.stack(xs), np.asarray(ys, dtype=np.float32), units

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred.reshape(-1) - y_true.reshape(-1)
    return float(np.sum(np.where(d < 0, np.exp(-d/13.0)-1.0, np.exp(d/10.0)-1.0)))

def build_model(seq_len: int, n_features: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def run(args):
    set_seed(args.seed)
    train, test, rul = load_cmapss(Path(args.data_dir), args.subset)
    train = add_train_rul(train, args.max_rul)
    features = select_features(train)

    # Split by engine BEFORE window construction. No engine contributes to both partitions.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_units, val_units = next(splitter.split(train, groups=train["unit_number"]))
    train_ids = set(train.iloc[train_units]["unit_number"].unique())
    val_ids = set(train.iloc[val_units]["unit_number"].unique())
    train_part = train[train.unit_number.isin(train_ids)].copy()
    val_part = train[train.unit_number.isin(val_ids)].copy()

    scaler = StandardScaler()
    scaler.fit(train_part[features])
    for frame in (train_part, val_part, test):
        frame.loc[:, features] = scaler.transform(frame[features])

    x_train, y_train, _ = make_train_windows(train_part, features, args.seq_len)
    x_val, y_val, _ = make_train_windows(val_part, features, args.seq_len)
    x_test, y_test, test_units = make_test_windows(test, features, rul, args.seq_len)

    model = build_model(args.seq_len, len(features))
    import tensorflow as tf
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2
    )
    pred = model.predict(x_test, verbose=0).reshape(-1)
    pred = np.clip(pred, 0.0, args.max_rul if args.max_rul is not None else np.inf)

    if args.clip_test_targets and args.max_rul is not None:
        y_eval = np.clip(y_test, 0.0, args.max_rul)
    else:
        y_eval = y_test

    result = {
        "subset": args.subset, "seed": args.seed, "n_train_engines": len(train_ids),
        "n_validation_engines": len(val_ids), "n_test_engines": len(test_units),
        "n_short_test_engines": int((test.groupby("unit_number").size() < args.seq_len).sum()),
        "seq_len": args.seq_len, "max_rul_cap": args.max_rul,
        "test_target_clipped": bool(args.clip_test_targets),
        "MAE": float(mean_absolute_error(y_eval, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_eval, pred))),
        "NASA_score": nasa_score(y_eval, pred),
        "epochs_trained": len(history.history["loss"]),
        "validation_MAE": float(min(history.history["val_mae"])),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out / f"{args.subset}_lstm_baseline_seed{args.seed}.csv", index=False)
    pd.DataFrame({"unit_number": test_units, "RUL_true": y_eval, "RUL_pred": pred}).to_csv(
        out / f"{args.subset}_lstm_baseline_predictions_seed{args.seed}.csv", index=False
    )
    print(pd.Series(result).to_string())
    print(f"Saved outputs to {out.resolve()}")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Directory containing NASA C-MAPSS train/test/RUL files")
    p.add_argument("--subset", choices=["FD001", "FD004"], default="FD001")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--max-rul", type=float, default=125.0)
    p.add_argument("--clip-test-targets", action="store_true",
                   help="Clip official test RUL labels to max-rul; default evaluates raw official labels.")
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
