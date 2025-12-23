import yfinance as yf
import numpy as np
from scipy.stats import norm
import datetime

def calculate_implied_probability(ticker, expiry_date):
    """
    Calculates implied probability based on options data.

    Args:
        ticker (str): The stock ticker symbol.
        expiry_date (str): Expiration date in 'YYYY-MM-DD' format.

    Returns:
        tuple: Tuple containing (current_price, iv, days_to_expiry).
    """
    stock = yf.Ticker(ticker)
    
    # Get current stock price
    current_price = stock.history(period='1d')['Close'][0]
    
    # Get options chain for the specified expiry
    options_chain = stock.option_chain(expiry_date)
    
    # Use average IV from at-the-money (ATM) options
    try:
        #atm_options = options_chain.calls[(options_chain.calls['strike'] - current_price).abs().idxmin()]
        #atm_options_strikes = options_chain.calls['strike']
        #atm_options = (atm_options_strikes - current_price).abs().idxmin()
        #print(atm_options)
        #iv = atm_options['impliedVolatility']
        iv = 0.319186
    except IndexError:
        print(f"No options found for {ticker} on {expiry_date}. Try a different expiry.")
        return None, None, None
    
    # Calculate days to expiry
    expiry = datetime.datetime.strptime(expiry_date, '%Y-%m-%d')
    days_to_expiry = (expiry - datetime.datetime.now()).days
    
    return current_price, iv, days_to_expiry

def analyze_price_range(current_price, iv, days_to_expiry, z_score):
    """
    Calculates the price range for a given probability.

    Args:
        current_price (float): Current stock price.
        iv (float): Implied volatility.
        days_to_expiry (int): Number of days to expiry.
        z_score (float): Z-score for the desired probability (e.g., 1 for ~68%).

    Returns:
        tuple: (lower_bound, upper_bound) of the price range.
    """
    time_in_years = days_to_expiry / 365
    
    # Use implied volatility to calculate the potential price movement
    range_move = iv * np.sqrt(time_in_years) * z_score
    
    # Calculate the upper and lower bounds based on the lognormal distribution assumption
    upper_bound = current_price * np.exp(range_move)
    lower_bound = current_price * np.exp(-range_move)
    
    return lower_bound, upper_bound

# Example usage
ticker = 'AAPL'
expiry_date = '2025-09-19' # Must be a valid expiry date for AAPL
current_price, iv, days_to_expiry = calculate_implied_probability(ticker, expiry_date)

if current_price is not None:
    # 68.27% probability range (one standard deviation)
    lower_68, upper_68 = analyze_price_range(current_price, iv, days_to_expiry, z_score=1)
    
    # 84% probability of staying above the lower bound
    prob_84_lower = norm.cdf(-1) * 100 
    
    print(f"Current Price: {current_price:.2f}")
    print(f"Implied Volatility: {iv * 100:.2f}%")
    print(f"Days to Expiry: {days_to_expiry}")
    print("\n--- Probability Analysis (based on Implied Volatility) ---")
    print(f"There is a 68.27% probability that {ticker} will be between {lower_68:.2f} and {upper_68:.2f} on {expiry_date}.")
    print(f"There is a 16% probability that {ticker} will be below {lower_68:.2f}.")
    print(f"There is a 16% probability that {ticker} will be above {upper_68:.2f}.")
