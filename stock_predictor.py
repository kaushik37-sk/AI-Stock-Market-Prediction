!pip install ta

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

def download_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print("Error: No data downloaded. Check the ticker symbol and date range.")
            return None
        
        df.ffill(inplace=True)
        
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze()).rsi()
        df['MACD'] = ta.trend.MACD(df['Close'].squeeze()).macd()
        df['EMA'] = ta.trend.EMAIndicator(df['Close'].squeeze(), window=20).ema_indicator()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()
        bb = ta.volatility.BollingerBands(df['Close'].squeeze())
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def prepare_data(df, feature='Close', window_size=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[[feature]])
    X, y = [], []
    for i in range(window_size, len(df_scaled)):
        X.append(df_scaled[i-window_size:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(60, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        Bidirectional(LSTM(60, return_sequences=False)),
        Dropout(0.3),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, X, scaler, steps=30):
    predictions = []
    current_input = X[-1].reshape(1, X.shape[1], 1)
    
    for _ in range(steps):
        next_pred = model.predict(current_input)[0][0]
        predictions.append(next_pred)
        
        current_input = np.append(current_input[:,1:,:], [[[next_pred]]], axis=1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Example usage
ticker = "AAPL"
start = "2023-01-01"
end = "2024-01-01"
data = download_stock_data(ticker, start, end)

if data is not None:
    X, y, scaler = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = build_lstm_model((X.shape[1], 1))
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=20, batch_size=32, callbacks=[early_stopping])
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    future_preds = predict_future(model, X, scaler, steps=30)
    
    plt.figure(figsize=(12,6))
    plt.plot(data.index[-len(predictions):], data['Close'].values[-len(predictions):], label='Actual Price')
    plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price')
    plt.axvline(x=data.index[-1], color='r', linestyle='--', label='Prediction Start')
    plt.plot(pd.date_range(data.index[-1], periods=30, freq='D'), future_preds, label='Future Predictions', linestyle='dashed')
    plt.legend()
    plt.show()
