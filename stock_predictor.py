import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Define stock tickers
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = "2010-01-01"  # Extended dataset for long-term analysis

# Initialize results dictionary
all_results = {}

for ticker in tickers:
    print(f"Processing {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Create features
    data["Moving_Avg"] = data["Close"].rolling(window=10).mean()
    data["Volatility"] = data["Close"].pct_change().rolling(window=10).std()
    data.dropna(inplace=True)

    # Define features and target variable
    X = data[["Moving_Avg", "Volatility"]]
    y = data["Close"].shift(-1).dropna()
    X = X.iloc[:len(y), :]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}

    # Train Logistic Regression (for price movement classification)
    y_class = (data["Close"].shift(-1) > data["Close"]).astype(int).dropna()
    X_class = X.iloc[:len(y_class), :]
    min_length = min(len(X_class), len(y_class))
    X_class = X_class.iloc[:min_length, :]
    y_class = y_class.iloc[:min_length]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_c, y_train_c)
    y_pred_logistic = logistic_model.predict(X_test_c)
    accuracy_logistic = accuracy_score(y_test_c, y_pred_logistic)

    # Train GDA Model
    qda_model = QDA()
    qda_model.fit(X_train_c, y_train_c)
    y_pred_qda = qda_model.predict(X_test_c)
    accuracy_qda = accuracy_score(y_test_c, y_pred_qda)

    # Store results
    results["Logistic Regression"] = {"Accuracy": accuracy_logistic}
    results["GDA"] = {"Accuracy": accuracy_qda}
    all_results[ticker] = results

    # Future Predictions
    future_year = 2030  # Change this to predict for any future year
    future_dates = pd.date_range(start=end_date, periods=(future_year - datetime.today().year) * 252, freq='B')
    future_X = pd.DataFrame(index=future_dates)
    future_X["Moving_Avg"] = data["Close"].rolling(window=10).mean().iloc[-1]
    future_X["Volatility"] = data["Close"].pct_change().rolling(window=10).std().iloc[-1]
    future_X_scaled = scaler.transform(future_X)
    future_predictions = models["XGBoost"].predict(future_X_scaled)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Actual Price", linestyle="-", color="blue")
    plt.plot(future_dates, future_predictions, label=f"Predicted Price for {future_year}", linestyle="dashed", color="green")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Stock Price Trends with Future Predictions")
    plt.legend()
    plt.grid()
    plt.show()

# Print results
for ticker, metrics in all_results.items():
    print(f"\nResults for {ticker}:")
    for model, values in metrics.items():
        print(f"{model} - {values}")
