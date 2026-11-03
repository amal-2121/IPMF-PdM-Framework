#!/usr/bin/env python3
"""Attention-LSTM + XGBoost residual-correction model for NASA C-MAPSS.

This is a transparent, reproducible hybrid implementation. It is not asserted
to be byte-for-byte or protocol-identical to any earlier unpublished IPMF code.

Expected files in --data-dir:
  train_FD001.txt, test_FD001.txt, RUL_FD001.txt
(or FD004 equivalents). CSV variants are accepted.

The residual learner is fitted on a held-out calibration group of training
engines. The LSTM never trains on those engines, so residual targets are
generated out-of-sample with respect to the base learner.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf


COLS = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def read_cmapss(path: Path, expected_cols=COLS) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    # NASA files are whitespace-delimited; CSV is also accepted.
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] > 26:
            df = df.iloc[:, :26]
        if df.shape[1] == 26:
            df.columns = expected_cols
            return df
    except Exception:
        pass
    df = pd.read_csv(path)
    if df.shape[1] == 26 and list(df.columns) != expected_cols:
        df.columns = expected_cols
    if list(df.columns) != expected_cols:
        raise ValueError(f"{path.name}: expected 26 columns, got {df.shape[1]}")
    return df


def load_data(data_dir: Path, subset: str):
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not train_path.exists():
        train_path = data_dir / f"train_{subset}.csv"
    if not test_path.exists():
        test_path = data_dir / f"test_{subset}.csv"
    if not rul_path.exists():
        rul_path = data_dir / f"RUL_{subset}.csv"

    train = read_cmapss(train_path)
    test = read_cmapss(test_path)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    if rul_df.shape[1] > 1:
        rul_df = pd.read_csv(rul_path, header=None)
    rul = pd.to_numeric(rul_df.iloc[:, 0], errors="raise").to_numpy(np.float32)

    for frame in (train, test):
        for col in COLS:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[COLS].isna().any().any():
            raise ValueError("Missing values found in C-MAPSS input.")
    test_units = sorted(test.unit_number.unique())
    if len(test_units) != len(rul):
        raise ValueError(f"Test engines ({len(test_units)}) != RUL labels ({len(rul)}).")
    return train, test, rul


def add_train_rul(train: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    out = train.copy()
    terminal = out.groupby("unit_number").time_cycles.transform("max")
    out["RUL"] = terminal - out.time_cycles
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


def feature_columns(train: pd.DataFrame) -> list[str]:
    # Keep all operating settings and sensors; constant columns are removed.
    candidates = COLS[2:]
    return [c for c in candidates if train[c].nunique(dropna=False) > 1]


def make_train_windows(frame: pd.DataFrame, features: list[str], seq_len: int):
    xs, ys, groups = [], [], []
    for unit, g in frame.groupby("unit_number", sort=True):
        g = g.sort_values("time_cycles")
        x = g[features].to_numpy(np.float32)
        y = g["RUL"].to_numpy(np.float32)
        for end in range(seq_len, len(x) + 1):
            xs.append(x[end-seq_len:end])
            ys.append(y[end-1])
            groups.append(int(unit))
    if not xs:
        raise ValueError("No training windows; reduce --seq-len.")
    return np.stack(xs), np.asarray(ys), np.asarray(groups)


def make_test_windows(frame: pd.DataFrame, features: list[str], rul: np.ndarray,
                      seq_len: int):
    xs, meta = [], []
    units = sorted(frame.unit_number.unique())
    for unit, true_rul in zip(units, rul):
        g = frame.loc[frame.unit_number == unit].sort_values("time_cycles")
        arr = g[features].to_numpy(np.float32)
        if len(arr) < seq_len:
            pad = np.repeat(arr[[0]], seq_len-len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        else:
            arr = arr[-seq_len:]
        xs.append(arr)
        meta.append((int(unit), float(true_rul), int(len(g))))
    return np.stack(xs), pd.DataFrame(meta, columns=["unit_number", "RUL_true", "trajectory_length"])


def make_attention_lstm(seq_len: int, n_features: int, learning_rate: float):
    inputs = tf.keras.Input(shape=(seq_len, n_features), name="sensor_sequence")
    x = tf.keras.layers.LSTM(96, return_sequences=True, name="lstm_sequence")(inputs)
    x = tf.keras.layers.Dropout(0.20)(x)
    scores = tf.keras.layers.Dense(1, name="attention_score")(x)
    weights = tf.keras.layers.Softmax(axis=1, name="temporal_attention")(scores)
    context = tf.keras.layers.Multiply()([x, weights])
    context = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1),
                                     name="attention_context")(context)
    embedding = tf.keras.layers.Dense(64, activation="relu", name="hybrid_embedding")(context)
    embedding = tf.keras.layers.Dropout(0.15)(embedding)
    output = tf.keras.layers.Dense(1, name="rul_prediction")(embedding)
    model = tf.keras.Model(inputs, output, name="Attention_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mae"]
    )
    encoder = tf.keras.Model(inputs, model.get_layer("hybrid_embedding").output,
                              name="Attention_LSTM_encoder")
    return model, encoder


def make_residual_features(encoder, x: np.ndarray) -> np.ndarray:
    latent = encoder.predict(x, verbose=0)
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    slope = (x[:, -1, :] - x[:, 0, :]) / max(x.shape[1]-1, 1)
    return np.concatenate([latent, last, mean, std, slope], axis=1)


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    return float(np.sum(np.where(d < 0, np.exp(-d/13.0)-1,
                                 np.exp(d/10.0)-1)))


def split_engine_groups(frame: pd.DataFrame, seed: int,
                        validation_fraction: float, calibration_fraction: float):
    groups = frame.unit_number.to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=validation_fraction + calibration_fraction,
                              random_state=seed)
    fit_idx, hold_idx = next(first.split(frame, groups=groups))
    hold = frame.iloc[hold_idx]
    hold_groups = hold.unit_number.to_numpy()
    relative_cal = calibration_fraction / (validation_fraction + calibration_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_cal, random_state=seed + 1009)
    val_rel, cal_rel = next(second.split(hold, groups=hold_groups))
    fit = frame.iloc[fit_idx].copy()
    val = hold.iloc[val_rel].copy()
    cal = hold.iloc[cal_rel].copy()
    return fit, val, cal


def run(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_raw, test_raw, official_rul = load_data(data_dir, args.subset)
    train = add_train_rul(train_raw, args.max_rul)
    features = feature_columns(train)

    fit_df, val_df, cal_df = split_engine_groups(
        train, args.seed, args.validation_fraction, args.calibration_fraction
    )

    scaler = StandardScaler()
    scaler.fit(fit_df[features])
    for part in (fit_df, val_df, cal_df, test_raw):
        part.loc[:, features] = scaler.transform(part[features])

    x_fit, y_fit, _ = make_train_windows(fit_df, features, args.seq_len)
    x_val, y_val, _ = make_train_windows(val_df, features, args.seq_len)
    x_cal, y_cal, cal_groups = make_train_windows(
        cal_df, features, args.seq_len
    )
    x_test, test_meta = make_test_windows(test_raw, features, official_rul, args.seq_len)

    model, encoder = make_attention_lstm(args.seq_len, len(features), args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(2, args.patience//2),
                                             factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(
        x_fit, y_fit, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=args.verbose
    )

    # LSTM predictions on calibration engines are out-of-sample.
    cal_base = model.predict(x_cal, verbose=0).reshape(-1)
    cal_features = make_residual_features(encoder, x_cal)
    residual = y_cal - cal_base

    # Split calibration engines into XGBoost-fit and alpha-selection groups.
    inner_split = GroupShuffleSplit(
        n_splits=1, test_size=1/3, random_state=args.seed + 2027
    )
    xgb_idx, alpha_idx = next(
        inner_split.split(cal_features, residual, groups=cal_groups)
    )

    residual_model = XGBRegressor(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, objective="reg:squarederror",
        random_state=args.seed, n_jobs=args.xgb_jobs
    )
    residual_model.fit(cal_features[xgb_idx], residual[xgb_idx])

    # Select alpha using held-out calibration engines only.
    alpha_base = cal_base[alpha_idx]
    alpha_correction = residual_model.predict(cal_features[alpha_idx])
    alpha_candidates = np.linspace(0.0, 1.5, 31)

    alpha_maes = []
    for a in alpha_candidates:
        candidate_pred = np.maximum(
            alpha_base + a * alpha_correction, 0.0
        )
        if args.max_rul is not None:
            candidate_pred = np.minimum(candidate_pred, args.max_rul)
        alpha_maes.append(
            mean_absolute_error(y_cal[alpha_idx], candidate_pred)
        )

    best_alpha = float(alpha_candidates[int(np.argmin(alpha_maes))])
    alpha_holdout_mae = float(min(alpha_maes))

    test_base = model.predict(x_test, verbose=0).reshape(-1)
    test_features = make_residual_features(encoder, x_test)
    correction = residual_model.predict(test_features)
    pred = np.maximum(test_base + best_alpha * correction, 0.0)

    y_true = test_meta.RUL_true.to_numpy()
    if args.clip_test_targets and args.max_rul is not None:
        pred = np.clip(pred, 0.0, args.max_rul)
        y_eval = np.clip(y_true, 0.0, args.max_rul)
    else:
        pred = np.maximum(pred, 0.0)
        y_eval = y_true

    metrics = {
        "subset": args.subset, "seed": args.seed,
        "n_fit_engines": int(fit_df.unit_number.nunique()),
        "n_validation_engines": int(val_df.unit_number.nunique()),
        "n_calibration_engines": int(cal_df.unit_number.nunique()),
        "n_test_engines": int(len(test_meta)),
        "n_short_test_engines": int((test_meta.trajectory_length < args.seq_len).sum()),
        "seq_len": args.seq_len, "max_rul_cap": args.max_rul,
        "test_targets_clipped": bool(args.clip_test_targets),
        "Attention_LSTM_MAE": float(mean_absolute_error(y_eval, test_base)),
        "Hybrid_MAE": float(mean_absolute_error(y_eval, pred)),
        "Hybrid_RMSE": float(np.sqrt(mean_squared_error(y_eval, pred))),
        "Hybrid_NASA_score": nasa_score(y_eval, pred),
        "epochs_trained": len(history.history["loss"]),
        "calibration_residual_MAE": float(mean_absolute_error(y_cal, cal_base)),
        "best_alpha": best_alpha,
        "alpha_selection_MAE": alpha_holdout_mae,
        "n_xgb_fit_engines": int(len(np.unique(cal_groups[xgb_idx]))),
        "n_alpha_selection_engines": int(len(np.unique(cal_groups[alpha_idx]))),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / f"{args.subset}_hybrid_seed{args.seed}.csv", index=False)
    predictions = test_meta.copy()
    predictions["Attention_LSTM_pred"] = test_base
    predictions["XGBoost_residual"] = correction
    predictions["Hybrid_pred"] = pred
    predictions.to_csv(out / f"{args.subset}_predictions_seed{args.seed}.csv", index=False)
    model.save(out / f"{args.subset}_attention_lstm_seed{args.seed}.keras")
    residual_model.save_model(out / f"{args.subset}_xgboost_residual_seed{args.seed}.json")
    with open(out / f"{args.subset}_metadata_seed{args.seed}.json", "w") as f:
        json.dump({"features": features, "metrics": metrics, "arguments": vars(args)}, f, indent=2)
    print(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subset", choices=["FD001", "FD004"], default="FD001")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--max-rul", type=float, default=125.0)
    p.add_argument("--clip-test-targets", action="store_true",
                   help="Clip official test RUL labels for capped-target evaluation; report this choice.")
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--calibration-fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--xgb-estimators", type=int, default=500)
    p.add_argument("--xgb-max-depth", type=int, default=3)
    p.add_argument("--xgb-learning-rate", type=float, default=0.03)
    p.add_argument("--xgb-jobs", type=int, default=-1)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
