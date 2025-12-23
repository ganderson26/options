import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option premium.

    Args:
        S: Current stock price.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate.
        sigma: Volatility of the underlying asset.
        option_type: 'call' or 'put'.

    Returns:
        The option premium.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    #d1 = (np.log(S / K) + (r + sigma**2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        premium = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        premium = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return premium

def calculate_historical_volatility(ticker, period="1y"):
    """
    Calculates historical volatility of a stock.

    Args:
        ticker: Stock ticker symbol.
        period: Period for which to calculate volatility (e.g., "1y", "6mo", "3mo").

    Returns:
        The annualized historical volatility.
    """
    data = yf.download(ticker, period=period)
    data['daily_returns'] = data['Close'].pct_change()
    return np.std(data['daily_returns']) * np.sqrt(252) # Annualize volatility

# Example Usage
ticker = "PLTR"
strike_price = 260
expiry_date_str = "2025-04-11"   # "2025-04-11"
risk_free_rate = 0.04 # 4%
option_type = "call"
start_date = '2025-03-30'  # '2025-03-30'
end_date = '2025-04-11'  # '2025-04-11'

# Convert expiration date to datetime object
expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
purchase_date = datetime.strptime(start_date, "%Y-%m-%d")
#time_to_expiry = (expiry_date - datetime.now()).days / 365
# Looks like 12 days is the magic number
time_to_expiry = (expiry_date - purchase_date).days / 365

##time_to_expiry = 16 / 365
#print(expiry_date)
#print(purchase_date)
print(expiry_date - purchase_date)
#print((expiry_date - purchase_date) / 365)
print(time_to_expiry) # Looks correct

# Get current stock price
stock_data = yf.download(ticker, start=start_date, end=end_date)
print(stock_data)
current_price = stock_data['Close'].iloc[-1]
print(current_price)
std_dev = np.std(current_price, ddof=1)
print(std_dev)

# Calculate historical volatility
historical_volatility = calculate_historical_volatility(ticker, period="6mo")
print(historical_volatility) # Looks correct, https://www.alphaquery.com/stock/PLTR/volatility-option-statistics/30-day/historical-volatility

# Calculate option premium
option_premium = black_scholes(current_price, strike_price, time_to_expiry, risk_free_rate, historical_volatility, option_type)

print("Option Premium: ", option_premium)
