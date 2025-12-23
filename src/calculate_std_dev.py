import pandas as pd
import numpy as np
import yfinance as yf
import statistics
import math

def flatten_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]

def calculate_future_price_std_dev(prices, time_period_years):
    """
    Calculates the standard deviation of stock price for a point in the future.

    Args:
        prices: A list of historical stock prices.
        time_period_years: The time period in years to project into the future.

    Returns:
        A tuple containing the annualized volatility and the estimated future price range (low, high).
    """

    print(prices)
    daily_returns = np.diff(prices) / prices[:-1]
    daily_std_dev = statistics.stdev(daily_returns)
    annualized_volatility = daily_std_dev * math.sqrt(252)

    current_price = prices[-1]
    price_range_change = current_price * annualized_volatility * math.sqrt(time_period_years)

    price_range_low = current_price - price_range_change
    price_range_high = current_price + price_range_change

    return annualized_volatility, (price_range_low, price_range_high)


def calculate_historical_std_dev(price_data, date, window=30):
  """
  Calculates the historical standard deviation of a stock price
  for a given date, using a rolling window.

  Args:
    price_data (pd.Series): A Pandas Series with dates as index and stock prices as values.
    date (str or datetime): The date for which to calculate the standard deviation.
    window (int): The number of preceding days to use for the calculation (default is 30).

  Returns:
    float: The standard deviation of the stock price for the given date,
           or NaN if there is not enough data.
  """
  
  if date not in price_data.index:
    raise ValueError(f"Date {date} not found in price data.")
  
  # Get the data up to the specified date
  data_up_to_date = price_data.loc[:date]

  # Ensure there's enough data for the window
  if len(data_up_to_date) < window:
    return np.nan

  # Calculate the rolling standard deviation
  std_dev = data_up_to_date.tail(window).std()
  return std_dev

# Example usage:
# Example Usage
ticker = "PLTR"
start_date = "2024-04-11"
end_date = "2025-04-11"  
# Assuming 'stock_prices.csv' has columns 'Date' and 'Close'
#df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=['Date'])
# Get current stock price
stock_data = yf.download(ticker, start=start_date, end=end_date)
df = pd.DataFrame(stock_data)

print(df)


# Calculate the standard deviation for a specific date
date_of_interest = '2025-04-04'
std_dev = calculate_historical_std_dev(df['Close'], date_of_interest)

if not pd.isna(std_dev).any():
  print(f"Standard deviation of stock price on {date_of_interest}: {std_dev}")
else:
  print(f"Not enough data to calculate standard deviation on {date_of_interest}")


# Calculate future std dev
just_price = df['Close'].values.tolist()
flat_list = flatten_list_comprehension(just_price)
future_std_dev = calculate_future_price_std_dev(flat_list, 1)
print("future std dev", future_std_dev)
