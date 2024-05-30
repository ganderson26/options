import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_puts_using_option_chain(symbol, start_date, end_date, duration_months=12):
    # Download historical price data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Define the put-selling strategy: Sell puts to collect premium when daily return is below -1 standard deviation
    data['Returns'] = data['Adj Close'].pct_change()
    std_dev = data['Returns'].std()
    data['PutSignal'] = np.where(data['Returns'] < -std_dev, 1, 0)

    # Initialize a DataFrame to store option premiums
    option_premiums = pd.DataFrame(index=data.index)

    # Loop through each day and retrieve option chain data
    for date in data.index:
        try:
            option_chain = yf.Ticker(symbol).option_chain(date.strftime('%Y-%m-%d'))
            puts = option_chain.puts
            print("puts" + puts)
            # Assuming you want to sell the put with the strike closest to 1 standard deviation below the stock price
            put_strike = data.loc[date, 'Adj Close'] * (1 - std_dev)
            put_option = puts.loc[puts['strike'].sub(put_strike).abs().idxmin()]

            # Collect premium from selling the put option
            option_premium = put_option['lastPrice'] * 100  # Assuming one contract represents 100 shares
            print("option_premium:" + option_premium)
            option_premiums.loc[date, 'PutPremium'] = -option_premium if data.loc[date, 'PutSignal'] == 1 else 0
        except:
            option_premiums.loc[date, 'PutPremium'] = 0

    # Calculate cumulative returns
    option_premiums['CumulativeReturns'] = (1 + option_premiums['PutPremium']).cumprod()

    # Print cumulative returns
    print("Cumulative Returns:", option_premiums['CumulativeReturns'][-1])

    # Plot strategy returns
    option_premiums[['CumulativeReturns']].plot(figsize=(10, 6), title='Put Premiums Backtest')
    plt.show()

# Example usage
symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
backtest_puts_using_option_chain(symbol, start_date, end_date)
