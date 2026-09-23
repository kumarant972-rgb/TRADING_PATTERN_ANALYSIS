import os
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

CLASS_NAMES = ["DOWN", "SIDEWAYS", "UP"]
WINDOW_SIZE = 60

# Yahoo Finance does not provide a native 4h interval, so 4h is
# constructed by resampling 1h candles.
TIMEFRAMES = {
    "30m": {"download_interval": "30m", "period": "60d",  "future": 3, "threshold": 0.004},
    "1h":  {"download_interval": "1h",  "period": "730d", "future": 3, "threshold": 0.006},
    "4h":  {"download_interval": "1h",  "period": "730d", "future": 2, "threshold": 0.010},
    "1d":  {"download_interval": "1d",  "period": "5y",   "future": 3, "threshold": 0.020},
}


def artifact_paths(interval):
    safe = interval.replace("m", "min").replace("h", "hr").replace("d", "day")
    return (
        f"cnn_model_{safe}.keras",
        f"scaler_{safe}.pkl",
        f"config_{safe}.pkl",
    )


def _flatten_ohlcv(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing OHLCV columns: " + ", ".join(missing))
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def download_data(symbol, interval):
    if interval not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {interval}")

    cfg = TIMEFRAMES[interval]
    fetch_interval = cfg["download_interval"]
    period = cfg["period"]

    df = yf.download(
        symbol.upper(),
        period=period,
        interval=fetch_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(
            f"No market data returned for {symbol.upper()} at {interval}. "
            f"Yahoo request used period={period}, interval={fetch_interval}."
        )

    df = _flatten_ohlcv(df)

    if interval == "4h":
        # Resample 1h data to 4-hour candles.
        df = df.resample("4h", origin="start_day", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    return df


def create_features(df):
    df = df.copy()
    close = df["Close"]

    df["return"] = close.pct_change()
    df["hl_range"] = (df["High"] - df["Low"]) / (close + 1e-10)
    df["body"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-10)
    df["upper_wick"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / (close + 1e-10)
    df["lower_wick"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    ) / (close + 1e-10)

    df["vol_change"] = df["Volume"].pct_change()
    df["vol_sma5"] = df["Volume"].rolling(5).mean() / (df["Volume"] + 1)

    for w in [5, 10, 20, 50]:
        df[f"sma{w}"] = close.rolling(w).mean() / (close + 1e-10) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26) / (close + 1e-10)
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / (bb_mid + 1e-10)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / (close + 1e-10)

    feature_cols = [
        "return", "hl_range", "body", "upper_wick", "lower_wick",
        "vol_change", "vol_sma5", "sma5", "sma10", "sma20", "sma50",
        "rsi", "macd", "signal", "hist", "bb_width", "atr"
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, feature_cols


def create_labels(df, interval):
    cfg = TIMEFRAMES[interval]
    future = cfg["future"]
    threshold = cfg["threshold"]

    out = df.copy()
    out["fwd_ret"] = out["Close"].shift(-future) / out["Close"] - 1
    out["target"] = np.select(
        [out["fwd_ret"] < -threshold, out["fwd_ret"] > threshold],
        [0, 2],
        default=1,
    )
    return out.dropna()


def build_sequences(df, feature_cols, window=WINDOW_SIZE):
    X, y, endpoints = [], [], []
    feat = df[feature_cols].values.astype("float32")
    target = df["target"].values.astype("int64")

    for i in range(window, len(df)):
        X.append(feat[i-window:i])
        y.append(int(target[i]))
        endpoints.append(i)

    return np.asarray(X, dtype="float32"), np.asarray(y), np.asarray(endpoints)


def build_cnn(window, n_features):
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(window, n_features)),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.40),
        Dense(64, activation="relu"),
        Dropout(0.30),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(symbol="AAPL", interval="1d", epochs=40):
    symbol = symbol.upper()
    raw = download_data(symbol, interval)
    df, features = create_features(raw)
    df = create_labels(df, interval)

    X, y, endpoints = build_sequences(df, features)
    if len(X) < 300:
        raise ValueError(
            f"Only {len(X)} sequences available for {symbol} {interval}. "
            "Use a symbol/timeframe with more history."
        )

    # Chronological holdout. An embargo equal to the forecast horizon
    # prevents the last training labels from reaching into the test period.
    split = int(len(X) * 0.80)
    future = TIMEFRAMES[interval]["future"]

    train_mask = endpoints < max(WINDOW_SIZE, split - future)
    test_mask = endpoints >= split

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    if len(X_tr) < 100 or len(X_te) < 50:
        raise ValueError("Not enough train/test samples after the time-series split.")

    if len(np.unique(y_tr)) < 3:
        raise ValueError("Training data does not contain DOWN, SIDEWAYS and UP classes.")

    scaler = StandardScaler()
    train_shape = X_tr.shape
    test_shape = X_te.shape

    X_tr = scaler.fit_transform(
        X_tr.reshape(-1, train_shape[-1])
    ).reshape(train_shape)
    X_te = scaler.transform(
        X_te.reshape(-1, test_shape[-1])
    ).reshape(test_shape)

    model = build_cnn(WINDOW_SIZE, len(features))

    classes = np.unique(y_tr)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_tr
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_prob = model.predict(X_te, verbose=0)
    preds = np.argmax(test_prob, axis=1)

    accuracy = accuracy_score(y_te, preds)
    avg_confidence = float(np.max(test_prob, axis=1).mean() * 100)

    model_file, scaler_file, config_file = artifact_paths(interval)
    model.save(model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump({
        "symbol": symbol,
        "interval": interval,
        "window": WINDOW_SIZE,
        "future": future,
        "threshold": TIMEFRAMES[interval]["threshold"],
        "features": features,
    }, config_file)

    report = classification_report(
        y_te, preds, target_names=CLASS_NAMES, zero_division=0
    )
    matrix = confusion_matrix(y_te, preds).tolist()

    return {
        "symbol": symbol,
        "interval": interval,
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "accuracy": float(accuracy * 100),
        "average_confidence": avg_confidence,
        "classification_report": report,
        "confusion_matrix": matrix,
        "history": history.history,
        "artifacts": [model_file, scaler_file, config_file],
    }


def load_artifacts(interval):
    model_file, scaler_file, config_file = artifact_paths(interval)
    missing = [p for p in [model_file, scaler_file, config_file] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "No trained model for this timeframe. Train "
            f"{interval} first. Missing: {', '.join(missing)}"
        )
    return load_model(model_file), joblib.load(scaler_file), joblib.load(config_file)


def predict(symbol="AAPL", interval="1d"):
    model, scaler, cfg = load_artifacts(interval)

    raw = download_data(symbol, interval)
    df, _ = create_features(raw)

    if len(df) < cfg["window"]:
        raise ValueError(
            f"Not enough candles for {interval}. "
            f"Need {cfg['window']}, got {len(df)}."
        )

    latest = df[cfg["features"]].tail(cfg["window"]).values
    latest = scaler.transform(latest)
    prob = model.predict(np.expand_dims(latest, axis=0), verbose=0)[0]

    idx = int(np.argmax(prob))
    confidence = float(prob[idx] * 100)

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "label": CLASS_NAMES[idx],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(prob[i] * 100) for i in range(3)
        },
        "latest_close": float(raw["Close"].iloc[-1]),
        "timestamp": str(raw.index[-1]),
        "chart_data": raw.tail(100),
    }
