import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Alpha Vantage API Key and symbol (replace 'YOUR_API_KEY' with your actual API key)
API_KEY = 'YOUR_API_KEY'
symbol = 'AAPL'  # Apple Inc. stock symbol

# Initialize Alpha Vantage API
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Get historical stock data (adjust the time period as needed)
data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')

# Extract dates and closing prices
dates = list(data.index)
closing_prices = list(data['4. close'])

# Prepare the data for prediction (simple lag feature)
lag_days = 5  # Number of lag days for prediction
features = []
targets = []

for i in range(lag_days, len(dates)):
    features.append(closing_prices[i - lag_days:i])
    targets.append(closing_prices[i])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values')
plt.plot(predictions, label='Predictions')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
