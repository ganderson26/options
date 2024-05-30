# Started with the following ChatGPT prompt:
# write a script that predicts stock price over the next few days


# Install Python
#https://www.python.org/downloads/

# Install packages from command line
#pip install pandas scikit-learn yfinance


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Function to get historical stock price data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data and train the model
def predict_stock_price(ticker, start_date, end_date, prediction_days):
    # Getting historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Feature engineering: using previous 'n' days' closing prices as features
    for i in range(1, prediction_days+1):
        stock_data[f'Close_Lag_{i}'] = stock_data['Close'].shift(i)
    
    # Dropping rows with NaN values
    stock_data.dropna(inplace=True)
    
    # Splitting data into features and target variable
    X = stock_data.drop(['Close'], axis=1)
    y = stock_data['Close']
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicting stock prices for the next 'prediction_days' days
    last_days_data = X[-prediction_days:]
    predicted_prices = model.predict(last_days_data)
    
    return predicted_prices

# Example usage
if __name__ == "__main__":
    # Ticker symbol of the stock you want to predict
    ticker = "DELL"
    # Date range for historical data (format: "YYYY-MM-DD")
    start_date = "2024-01-01"
    end_date = "2024-05-24"
    # Number of days to predict into the future
    prediction_days = 5
    
    # Predict stock prices
    predicted_prices = predict_stock_price(ticker, start_date, end_date, prediction_days)
    
    # Print the predicted prices for the next 'prediction_days' days
    print(ticker + f" Prediction for the next {prediction_days} days starting " + start_date + " and ending " + end_date)

    for i in range(prediction_days):
        print(f"Predicted price for Day {i+1}: {predicted_prices[i]}")
        print(f"Predicted price for Day {prediction_days}: {predicted_prices[i]}")
