import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st

from trading_analyzer import (
    TIMEFRAMES,
    predict,
    train,
    artifact_paths,
)

st.set_page_config(
    page_title="Trading Chart Pattern Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading Chart Pattern Analyzer")
st.caption(
    "CNN-based OHLCV classifier: DOWN / SIDEWAYS / UP. "
    "Backtest accuracy and confidence are shown separately."
)

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Stock Symbol", "TSLA").strip().upper()
    timeframe = st.selectbox(
        "Timeframe",
        ["30m", "1h", "4h", "1d"],
        index=0,
        help="4h is built by resampling 1-hour candles.",
    )
    mode = st.radio("Mode", ["Train / Backtest", "Predict / Chart"])

    if mode == "Train / Backtest":
        epochs = st.slider("Maximum training epochs", 10, 60, 30)
        run_train = st.button("🚀 Train model", type="primary", use_container_width=True)
    else:
        run_predict = st.button("📊 Predict", type="primary", use_container_width=True)

    st.divider()
    st.info(
        "Yahoo Finance intraday history is limited. "
        "The app automatically uses a suitable history window for each timeframe."
    )

if mode == "Train / Backtest" and run_train:
    st.subheader(f"Training {symbol} — {timeframe}")
    status = st.empty()
    progress = st.progress(0)

    try:
        status.info(
            f"Downloading {symbol} data for {timeframe} and training the CNN..."
        )
        progress.progress(10)

        result = train(symbol=symbol, interval=timeframe, epochs=epochs)
        progress.progress(100)
        status.success("Training completed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Backtest accuracy", f"{result['accuracy']:.2f}%")
        c2.metric("Avg. model confidence", f"{result['average_confidence']:.2f}%")
        c3.metric("Train samples", result["train_samples"])
        c4.metric("Test samples", result["test_samples"])

        st.subheader("What these numbers mean")
        st.write(
            "Backtest accuracy is the percentage of chronological holdout samples "
            "classified correctly. Model confidence is the average maximum softmax "
            "probability on those holdout samples. Neither number guarantees future returns."
        )

        st.subheader("Classification report")
        st.code(result["classification_report"])

        st.subheader("Confusion matrix")
        st.write(result["confusion_matrix"])

        st.subheader("Training artifacts")
        for artifact in result["artifacts"]:
            st.write(f"✅ `{artifact}`")

        hist = result["history"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["accuracy"], label="Train accuracy")
        ax.plot(hist["val_accuracy"], label="Validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{symbol} {timeframe} — CNN accuracy")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as exc:
        progress.empty()
        status.error(f"Training failed: {exc}")
        st.exception(exc)

if mode == "Predict / Chart" and run_predict:
    try:
        with st.spinner(f"Downloading {symbol} {timeframe} data and predicting..."):
            result = predict(symbol=symbol, interval=timeframe)

        label = result["label"]
        confidence = result["confidence"]
        emoji = {"UP": "🟢", "DOWN": "🔴", "SIDEWAYS": "🟡"}[label]

        if confidence >= 80:
            confidence_band = "High model confidence"
        elif confidence >= 60:
            confidence_band = "Moderate model confidence"
        else:
            confidence_band = "Low model confidence"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", f"{emoji} {label}")
        c2.metric("Confidence", f"{confidence:.1f}%")
        c3.metric("Latest close", f"{result['latest_close']:.2f}")
        c4.metric("Timeframe", result["interval"])

        st.progress(
            min(max(confidence / 100, 0), 1),
            text=f"{confidence_band}: {confidence:.1f}%",
        )

        st.subheader("Model probabilities")
        p1, p2, p3 = st.columns(3)
        p1.metric("🔴 DOWN", f"{result['probabilities']['DOWN']:.1f}%")
        p2.metric("🟡 SIDEWAYS", f"{result['probabilities']['SIDEWAYS']:.1f}%")
        p3.metric("🟢 UP", f"{result['probabilities']['UP']:.1f}%")

        st.caption(
            "Confidence is the model's softmax probability for the selected class; "
            "it is not a probability that the stock will actually rise or fall."
        )

        st.subheader(f"Candlestick chart — {symbol} ({timeframe})")
        chart = result["chart_data"].copy()
        fig, _ = mpf.plot(
            chart,
            type="candle",
            style="charles",
            volume=True,
            title=f"{symbol} | {timeframe} | {label} ({confidence:.1f}%)",
            figsize=(12, 7),
            returnfig=True,
            tight_layout=True,
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.caption(f"Latest market candle: {result['timestamp']}")

    except Exception as exc:
        st.error(str(exc))
        st.info(
            "If this timeframe has not been trained yet, switch to "
            "'Train / Backtest' and train it first."
        )

st.divider()
st.caption(
    "Educational/research software only. Backtest accuracy and model confidence "
    "are historical/model metrics, not guaranteed future performance."
)
