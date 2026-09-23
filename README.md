# Trading Chart Pattern Analyzer

CNN-based OHLCV classifier with separate models for 30m, 1h, 4h and 1d.

## Local run

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run app.py
```

## Training

Open **Train / Backtest**, choose a stock and timeframe, then train each timeframe separately.

The app reports:
- chronological holdout backtest accuracy
- average softmax model confidence
- DOWN / SIDEWAYS / UP probabilities
- classification report
- confusion matrix

## Timeframes

- 30m: Yahoo 30-minute data, 60-day history
- 1h: Yahoo 1-hour data, 730-day history
- 4h: 1-hour data resampled into 4-hour candles
- 1d: daily data, 5-year history

A separate model/scaler/config is saved for every timeframe.

## Important

Backtest accuracy is historical validation only. Confidence is the model's softmax probability, not a guaranteed future-market probability or return forecast.
