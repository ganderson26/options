import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def backtest_puts_and_calls(symbol, start_date, end_date):
    # Download historical price data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate daily returns
    data['Returns'] = data['Adj Close'].pct_change()

    # Calculate 1 standard deviation of daily returns
    std_dev = data['Returns'].std()

    # Define the put-selling strategy: Sell puts to collect premium when daily return is below -1 standard deviation
    data['PutSignal'] = np.where(data['Returns'] < -std_dev, 1, 0)

    # Define the call-selling strategy: Sell calls if assigned from put-selling strategy for 12 months
    #data['CallSignal'] = data['PutSignal'].rolling(window=252).sum()  # Assuming 252 trading days in a year

    # Calculate strategy returns from collecting premium
    option_chain = yf.Ticker(symbol).option_chain(data['PutSignal'])
    print(option_chain)
    #data['PutPremium'] = -data['PutSignal'] * data['Returns']
    #data['PutPremium'] = -data['PutSignal'].rolling(window=252 * 12).sum() * data['Returns']
    print("Put Premium:", data['PutPremium'])
    #data['CallPremium'] = -data['CallSignal'].shift(1) * data['Returns']
    #data['StrategyReturns'] = data['PutPremium'] + data['CallPremium']

    data['StrategyReturns'] = data['PutPremium']

    # Calculate cumulative returns
    data['CumulativeReturns'] = (1 + data['StrategyReturns']).cumprod()

    # Print cumulative returns
    print("Cumulative Returns:", data['CumulativeReturns'][-1])

    # Plot strategy returns
    data[['CumulativeReturns', 'Adj Close']].plot(figsize=(10, 6))
    plt.show()

# Example usage
symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
backtest_puts_and_calls(symbol, start_date, end_date)
