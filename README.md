# AI-Stock-Market-Prediction
An AI-powered stock market prediction model with graphs and results
AI-Powered Stock Price Prediction
This project uses Machine Learning to predict stock prices and classify stock movements (whether the price will go up or down). The main goal is to evaluate the performance of different machine learning models and use them to make informed predictions for investors.

1. Project Overview
Stock market prediction is a challenging yet essential area of financial analysis. By using historical stock data and several machine learning algorithms, this project predicts stock price trends for major tech companies. We implemented and tested the following models:

Linear Regression

Random Forest

XGBoost

Logistic Regression (for classification)

Quadratic Discriminant Analysis (QDA)

2. Project Objectives
Predict the future stock prices of major tech companies.

Evaluate the performance of various regression and classification models.

Use machine learning to classify whether a stock’s price will increase or decrease.

3. Data Collection
The dataset for this project is obtained using the Yahoo Finance API (yfinance), which provides real-time stock data. The stocks analyzed are:

Apple (AAPL)

Google (GOOGL)

Microsoft (MSFT)

Amazon (AMZN)

Tesla (TSLA)

Data used for the analysis includes daily closing prices, moving averages, and volatility measures.

4. Models & Methodology
Regression Models for Price Prediction
Linear Regression: Attempts to model the relationship between the stock price and other features.

Random Forest: An ensemble model that averages over several decision trees to reduce overfitting.

XGBoost: A highly efficient gradient boosting model that performs well on structured data.

Classification Models for Price Movement
Logistic Regression: Used to predict whether the price of the stock will go up or down.

Quadratic Discriminant Analysis (QDA): A probabilistic classifier used for categorizing stock price movements.

5. Results & Insights
Here are the results for each stock and model:

Apple (AAPL):
Linear Regression: MSE: 42.09, R²: 0.95

Random Forest: MSE: 40.84, R²: 0.95

XGBoost: MSE: 52.19, R²: 0.94

Logistic Regression: Accuracy: 59.09%

GDA: Accuracy: 59.09%

Google (GOOGL):
Linear Regression: MSE: 30.90, R²: 0.97

Random Forest: MSE: 24.38, R²: 0.97

XGBoost: MSE: 31.42, R²: 0.97

Logistic Regression: Accuracy: 59.09%

GDA: Accuracy: 52.73%

Microsoft (MSFT):
Linear Regression: MSE: 77.10, R²: 0.98

Random Forest: MSE: 67.60, R²: 0.98

XGBoost: MSE: 94.53, R²: 0.97

Logistic Regression: Accuracy: 57.27%

GDA: Accuracy: 54.55%

Amazon (AMZN):
Linear Regression: MSE: 29.05, R²: 0.98

Random Forest: MSE: 36.81, R²: 0.98

XGBoost: MSE: 42.79, R²: 0.98

Logistic Regression: Accuracy: 57.27%

GDA: Accuracy: 56.36%

Tesla (TSLA):
Linear Regression: MSE: 441.02, R²: 0.90

Random Forest: MSE: 573.22, R²: 0.88

XGBoost: MSE: 667.53, R²: 0.86

Logistic Regression: Accuracy: 50.91%

GDA: Accuracy: 54.55%

Key Takeaways:
XGBoost performed well overall, delivering high R² scores in most cases, but it was not always the top performer.

Random Forest was a strong competitor, often providing comparable results to XGBoost.

Linear Regression worked best for predicting stock price trends, especially in Google and Microsoft.

Logistic Regression and GDA were more suitable for classifying price movements but struggled to achieve high accuracy.

6. Visualizations
Several graphs were created to visualize the stock price trends and model performances:

Stock Price History: Line charts displaying price trends for each company.

Model Performance: Bar charts comparing MSE and R² scores across different models.

Accuracy Comparison: Bar charts comparing classification models (Logistic Regression vs. GDA).

7. Future Enhancements
Try deep learning models like LSTMs for better time-series forecasting.

Add more macroeconomic features such as inflation or interest rates for improved predictions.

Implement hyperparameter tuning to optimize the model performance further.

8. Conclusion
This project demonstrated the application of machine learning models to predict stock prices and classify price movements. While no model can guarantee perfect predictions in such a volatile market, these techniques provide useful insights that can assist in making more informed investment decisions.

Tools & Technologies Used:
Python Libraries: Pandas, NumPy, Matplotlib, Seaborn


## 📝 Code  
You can find the full implementation here: [stock_predictor.py](stock_predictor.py)

Machine Learning Libraries: Scikit-Learn, XGBoost

Yahoo Finance API: For real-time stock data

Data Visualization: Matplotlib, Seaborn for creating insightful graphs
