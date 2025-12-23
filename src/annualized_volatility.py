import pandas as pd
import numpy as np
import yfinance as yf

# 1. Get historical stock data
ticker_symbol = 'AAPL'
data = yf.download(ticker_symbol, period='1y', auto_adjust=True)['Close']

# 2. Calculate daily logarithmic returns
log_returns = np.log(data / data.shift(1)).dropna()

# 3. Compute the standard deviation of the returns
daily_volatility = log_returns.std()

# 4. Annualize the volatility
trading_days = 252  # Standard for most markets
annualized_volatility = daily_volatility * np.sqrt(trading_days)

# Display the result
print("Annualized Volatility:", annualized_volatility)
