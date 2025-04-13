AI Stock Market Prediction
Overview
This project utilizes machine learning techniques, specifically LSTM (Long Short-Term Memory) neural networks, to predict future stock prices based on historical data. The model uses various technical indicators (such as RSI, MACD, EMA, OBV, and Bollinger Bands) to enhance the prediction's accuracy. The stock data is sourced from Yahoo Finance using the yfinance library.

Features
Downloads stock data for a given ticker symbol from Yahoo Finance.

Calculates multiple technical indicators for improved prediction.

Prepares the data by scaling it and creating sliding windows for training the model.

Builds and trains a Bidirectional LSTM model to predict stock prices.

Predicts future stock prices and visualizes the results.

Requirements
Before running the script, make sure to install the following libraries:

bash
Copy
Edit
pip install numpy pandas yfinance ta matplotlib scikit-learn tensorflow
Usage
Step 1: Download Stock Data
The script begins by asking the user for the stock ticker symbol, start date, and end date for downloading historical stock data. For example, you can input AAPL for Apple, or TSLA for Tesla.

bash
Copy
Edit
Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT): AAPL
Enter the start date (YYYY-MM-DD): 2020-01-01
Enter the end date (YYYY-MM-DD): 2022-01-01
Step 2: Data Preprocessing
The stock data is cleaned and preprocessed:

Missing values are forward-filled.

Technical indicators are calculated (RSI, MACD, EMA, OBV, and Bollinger Bands).

The data is scaled using MinMaxScaler for model training.

Step 3: Build and Train the Model
An LSTM model is built with two Bidirectional LSTM layers and two Dropout layers to reduce overfitting. The model is trained using the adam optimizer and mean_squared_error loss function. Early stopping is employed to prevent overtraining and ensure the best model is used.

Step 4: Prediction and Visualization
Once trained, the model predicts the stock prices for the test data. It also predicts the next 30 days of stock prices (future predictions). The results are plotted for visual comparison, showing both actual and predicted stock prices, with future predictions shown in a dashed line.

python
Copy
Edit
plt.figure(figsize=(12,6))
plt.plot(data.index[-len(predictions):], data['Close'].values[-len(predictions):], label='Actual Price')
plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price')
plt.axvline(x=data.index[-1], color='r', linestyle='--', label='Prediction Start')
plt.plot(pd.date_range(data.index[-1], periods=30, freq='D'), future_preds, label='Future Predictions', linestyle='dashed')
plt.title(f"Stock Price Prediction for {ticker}")
plt.legend()
plt.show()
Step 5: Results
The results will display a graph with the actual stock prices versus predicted prices, followed by predicted values for the next 30 days.

File Breakdown
main.py: The main script that performs all the tasks: downloading stock data, preprocessing, model building, training, and visualization.

README.md: Documentation that explains how to set up and run the project.

requirements.txt: A list of required Python libraries for the project.

Future Enhancements
Model Optimization: Implement hyperparameter tuning (e.g., grid search or random search) to improve model performance.

Additional Features: Incorporate more features like sentiment analysis or news-based predictions.

Deployment: Make the model accessible via a web interface or as an API for real-time stock prediction.

License
This project is licensed under the MIT License - see the LICENSE file for details.
