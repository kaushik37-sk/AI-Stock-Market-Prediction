**AI-Driven Stock Market Prediction Project**

**Project Overview:**
This project utilizes machine learning techniques to predict future stock prices based on historical data. The model incorporates various regression and classification methods to enhance prediction accuracy and analyze stock market trends.

**Key Features:**
- **Stock Data Retrieval:** Uses Yahoo Finance (yfinance) API to collect historical stock data from 2010 to the present.
- **Feature Engineering:** Computes moving averages and volatility metrics to serve as model inputs.
- **Machine Learning Models:** Implements Linear Regression, Random Forest, XGBoost, Logistic Regression, and Quadratic Discriminant Analysis (QDA).
- **Future Predictions:** Allows forecasting stock prices for any future year (e.g., 2026, 2030, etc.).
- **Visualization:** Generates graphs comparing actual historical prices with predicted future prices.

**Methodology:**
1. **Data Collection:** Historical stock data is retrieved for multiple major companies (AAPL, GOOGL, MSFT, AMZN, TSLA).
2. **Preprocessing & Feature Engineering:** Moving averages and volatility are calculated to represent stock trends.
3. **Model Training:**
   - Regression models predict stock price movements.
   - Classification models predict whether a stock price will increase or decrease.
4. **Prediction for Future Years:** The user can specify any future year, and the model will generate stock price forecasts.
5. **Performance Evaluation:** Metrics like Mean Squared Error (MSE), R-squared (R2), and Accuracy are computed for model assessment.

**Results:**
- **Regression Models:** Provided strong R-squared values, indicating high accuracy in price trend predictions.
- **Classification Models:** Demonstrated reasonable accuracy in predicting price movement direction.
- **Future Predictions:** XGBoost was found to be the most effective model for forecasting stock prices beyond 2026.

**Conclusion:**
This AI-powered stock market prediction model provides insights into future stock trends based on historical data. While stock markets are inherently unpredictable, machine learning models can identify patterns and trends that aid in investment decision-making. The model is highly adaptable and allows future-year predictions, making it a valuable tool for financial analysis.

**Future Enhancements:**
- Expanding feature engineering to incorporate sentiment analysis from financial news.
- Integrating deep learning techniques such as LSTMs for improved time-series predictions.
- Extending the dataset to include global economic indicators for more robust modeling.

