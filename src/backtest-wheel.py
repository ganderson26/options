# Started with the following ChatGPT prompt:
#write a script that will backtest selling puts at 1 standard deviation and if assigned sell calls for 12 months

# Install Python
#https://www.python.org/downloads/

# Install packages from command line
#pip install pandas numpy yfinance


import pandas as pd
import numpy as np
import yfinance as yf  # You may need to install this package: pip install yfinance

# Function to get historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(stock_data)

    return stock_data['Adj Close']

# Function to backtest the strategy
def backtest_strategy(cash, std_dev_num, trade_days, ticker, start_date, end_date):
    # Get historical stock data
    stock_prices = get_stock_data(ticker, start_date, end_date)

    # Calculate daily returns
    daily_returns = stock_prices.pct_change()

    # Calculate 1 standard deviation
    std_dev = daily_returns.std()

    # Set up initial parameters
    put_strike = stock_prices[-1] * (std_dev_num - std_dev)  # Put strike at standard deviation below current price
    call_strike = stock_prices[-1] * (std_dev_num + std_dev)  # Call strike at standard deviation above current price
    positions = []
    
    # Backtest the strategy
    for i in range(len(stock_prices)):
        if i % trade_days == 0:  # Trade every 21 days (monthly)
            # Sell put option
            put_option_price = max(put_strike - stock_prices[i], 0)

            option_chain = yf.Ticker(ticker_symbol).option_chain(put_option_price)
            print(option_chain)

            cash -= put_option_price
            positions.append(-1)
            

            # Check if put option is assigned
            if stock_prices[i] < put_strike:
                # If assigned, sell call option
                call_option_price = max(stock_prices[i] - call_strike, 0)
                cash += call_option_price
                positions[-1] += 1

    # Calculate final portfolio value
    final_value = cash + sum([pos * stock_prices.iloc[-1] for pos in positions])

    return final_value

# Define parameters
cash = 100000  # Initial cash amount
std_dev_num = 1
trade_days = 21
ticker_symbol = 'AAPL'
start_date = '2019-01-01'
end_date = '2020-01-01'

# Backtest the strategy
portfolio_value = backtest_strategy(cash, std_dev_num, trade_days, ticker_symbol, start_date, end_date)

print("Wheel Backtest for " + ticker_symbol + " from " + start_date + " to " + end_date + 
      f" and starting with ${cash:.2f}" + f" standard deviation number {std_dev_num:.0f}" +
      f" trading every {trade_days:.0f} days")

print(f"Final Portfolio Value: ${portfolio_value:.2f}")
