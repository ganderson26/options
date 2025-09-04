# get_option_chains.py
#
# Description: 
# Get Option Chain Data and Price, Annualized Volatility, Expiration Date, Days to Expire, 
# Lower and Upper Price at 1 Standard Deviation.
#
# Get Option Chain:
# 1. Get option chain on Friday for the next Friday expiration date. There will be many lastTradeDates. 
#    This may not be a concern since we will be looking for specific Strikes.
# 2. Loop through the array of symbols:
#    a. Get option chain for the current expiration. This should be the 1st element in the array.
#    b. Create 2 DataFrames, one for callas and the other for puts.
#    c. Get current price. For ease of use, save the current price to each row.
#    d. Calculate annualized volatility for the past year. For ease of use, save the current price to each row.
#    e. Calculate the price range for 1 standard deviation, 68.27%. This appears to be what ToS is doing.
#       For ease of use, save the current price to each row.
#    f. Save to database.
#    g. Write message to logfile.
#
# Future Enhancements:
# 1. Calculate the greeks.
# 2. Determine if Friday is a holiday.
#
# Setup cron job for 1 of the following:
# crontab for Monday to Friday at 2:30
# crontab -e
# 30 14 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains_greek.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#
# crontab for Monday to Friday at 12:00
# crontab -e
# 0 12 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains_greek.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#
# crontab for Friday at 12:00
# crontab -e
# 0 12 * * 5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains_greek.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#

import numpy as np
import pandas as pd
import yfinance as yf 
import datetime
from scipy.stats import norm
from sqlalchemy import create_engine

def get_option_chains():
    """
    Get the current option chain. This is the main driving function. It calls most all the other functions.

    Args:
        None.

    Returns:
        option_chains_df: The calculated option chain.
    """
    #symbols =['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    symbols =['AAPL']
    option_chains_df = pd.DataFrame()

    for symbol in symbols:
        # Get ticker
        ticker = yf.Ticker(symbol)
       
        # Get all chains for expiration dates
        expiration_dates = ticker.options

        # Get current expiration
        expiration_date = expiration_dates[0]
        option_chain = ticker.option_chain(expiration_date)

        # Create a DataFrame for calls and puts
        calls_df = pd.DataFrame(option_chain.calls)
        puts_df = pd.DataFrame(option_chain.puts)

        # Get current price
        current_price = get_current_price(ticker)
        calls_df['currentPrice'] = current_price
        puts_df['currentPrice'] = current_price

        # Get annualized_volatility
        annualized_volatility = get_annualized_volatility(symbol)
        calls_df['annualized_volatility'] = annualized_volatility
        puts_df['annualized_volatility'] = annualized_volatility

        # Calculate price range
        # 68.27% probability range (one standard deviation)
        days_to_expiry, expiry_date = get_next_friday_date() # Must be a valid expiry date
        lower_68, upper_68 = analyze_price_range(current_price, annualized_volatility, days_to_expiry, z_score=1)
        calls_df['expiry_date'] = expiry_date
        calls_df['days_to_expiry'] = days_to_expiry
        calls_df['lower_68'] = lower_68
        calls_df['upper_68'] = upper_68
        puts_df['expiry_date'] = expiry_date
        puts_df['days_to_expiry'] = days_to_expiry
        puts_df['lower_68'] = lower_68
        puts_df['upper_68'] = upper_68

        # Calculate delta per strike
        for index, row in puts_df.iterrows():
            strike = row['strike']
            iv = round(annualized_volatility, 2)
            #iv = .2012
            volume = row['volume']
            bid = row['bid']
        
            r = 0.05 # Risk-free rate (5%)
            t = float(days_to_expiry)
            T = t/365

            #delta = round(black_scholes_delta(current_price, strike, T, r, iv, option_type="put"), 2)
            #print("Euro", strike, iv, volume, bid, r, t, T, delta)

            # Try american
            N = 100 # Number of steps in the binomial tree
     
            delta_put = round(american_option_delta_binomial(current_price, strike, T, r, iv, N, "put"), 2)
            #print("AMER PUT", strike, iv, volume, bid, r, t, T, delta_put)
            puts_df.at[index, 'delta'] = delta_put

            delta_call = round(american_option_delta_binomial(current_price, strike, T, r, iv, N, "call"), 2)
            #print("AMER CALL", strike, iv, volume, bid, r, t, T, delta_call)
            calls_df.at[index, 'delta'] = delta_call


            
        
        # Concatenate the calls and puts DataFrames
        option_chains_df = pd.concat([option_chains_df, calls_df, puts_df])

        #print(option_chains_df)
                    
    return option_chains_df

def get_current_price(ticker):
    """
    Get the current price.

    Args:
        ticker: The ticker for the stock.

    Returns:
        current_price: The current price for the stock when this program executes.
    """
     
    current_price = ticker.info.get('currentPrice')

    #print(current_price)

    return current_price

# Get annualized volatility
def get_annualized_volatility(symbol):
    """
    Calculates the annualized volatility.

    Args:
        symbol: The ticker for the stock.

    Returns:
        annualized_volatility: The volatility for the year.
    """

    # Get historical stock data
    data = yf.download(symbol, period='1y', auto_adjust=True)['Close']
    #print(data)

    # Calculate daily logarithmic returns
    ##log_returns = np.log(data / data.shift(1)).dropna()
    log_returns = np.diff(np.log(data.iloc[:, -1]))

    # Compute the standard deviation of the returns
    ##daily_volatility = log_returns.std()
    daily_volatility = np.std(log_returns)

    # Annualize the volatility
    trading_days = 252  # Standard for most markets
    annualized_volatility = daily_volatility * np.sqrt(trading_days)

    # Get 1st value in Series
    ##return annualized_volatility[0]
    return annualized_volatility

def analyze_price_range(current_price, iv, days_to_expiry, z_score):
    """
    Calculates the price range for a given probability.

    Args:
        current_price: The current price of the stock for when this program executes.
        iv: The annualized volatility.
        days_to_expiry: The number of days till the expiration date.
        z_score: 1, which is the 68.27% probability range (one standard deviation).

    Returns:
        lower_bound: The lower bound price.
        upper_bound: The upper bound price.
    """

    # Use the year
    time_in_years = days_to_expiry / 365
    
    # Use implied volatility to calculate the potential price movement
    range_move = iv * np.sqrt(time_in_years) * z_score
    
    # Calculate the upper and lower bounds based on the lognormal distribution assumption
    upper_bound = current_price * np.exp(range_move)
    lower_bound = current_price * np.exp(-range_move)
    
    return lower_bound, upper_bound

def get_next_friday_date():
    """
    Determine the date of the next Friday which is the option expiration and the number of days from when 
    this program executes which is scheduled to run on the previous Friday. This execution is scheduled 
    in a cron job.

    Args:
        None.

    Returns:
        days_until_friday: Number of days till next expiration date (next Friday).
        next_friday: Date of the next Friday.
    """

    # Get today's date
    today = datetime.date.today()

    # Calculate the number of days until the next Friday
    # weekday() returns 0 for Monday, 1 for Tuesday, ..., 4 for Friday, ..., 6 for Sunday
    # If today is Friday (weekday() == 4), the next Friday is 7 days away.
    # Otherwise, calculate the difference to reach Friday (4) and add 7 if the target day has already passed this week.
    days_until_friday = (4 - today.weekday() + 7) % 7

    # If today is already Friday, days_until_friday will be 0, so set it to 7 to get the *next* Friday.
    if days_until_friday == 0:
        days_until_friday = 7

    # Add the calculated days to today's date
    next_friday = today + datetime.timedelta(days=days_until_friday)

    #print("The date for the next Friday:", next_friday)
    #print("Days till next Friday:", days_until_friday)

    return days_until_friday, next_friday

def save_to_mysql(df):
    """
    Save DataFrame to MySQL.

    Args:
        df: DataFrame.

    Returns:
        None.
    """

    # Replace with your MySQL credentials and database name
    db_connection_str = 'mysql+pymysql://root:Marathon#262@localhost:3306/OPTIONS'
    engine = create_engine(db_connection_str)

    # Name of the table in MySQL
    table_name = 'OPTION_CHAINS'  

    # Will create the following columns
    # contractSymbol,lastTradeDate,strike,lastPrice,bid,ask,change,percentChange,volume,openInterest,impliedVolatility,inTheMoney,contractSize,currency,currentPrice,annualized_volatility,expiry_date,days_to_expire,lower_68,upper_68

    # Use to_sql() to write the DataFrame to the database
    # if_exists options: 'fail', 'replace', 'append'
    # index=False prevents writing the DataFrame index as a column in the table
    # Used 'replace' to initially create the table and column from the DataFrame
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    NOT USED

    Calculates the Black-Scholes delta for a European option.

    Args:
        S (float): Current underlying asset price.
        K (float): Option strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        option_type (str): "call" for a call option, "put" for a put option.

    Returns:
        float: The delta of the option.
    """

    #print('S:', S)
    #print('K:', K)
    #print('T:', T)
    #print('r:', r)
    #print('sigma:', sigma)
    
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return delta

def american_option_delta_binomial(S, K, T, r, sigma, N, option_type='call'):
        """
        Calculating the delta for American options in Python typically requires numerical methods, as 
        there are no simple closed-form solutions like for European options (Black-Scholes). The most 
        common methods are:
        Binomial or Trinomial Tree Models:
            - These models build a lattice of possible stock prices over time.
            - At each node, the option value is calculated by working backward from expiration, considering 
              the possibility of early exercise for American options.
            - The delta can be approximated by observing the change in option price at a given node when the 
              underlying stock price changes by one step in the tree.

        Args:
            S: Current underlying asset price.
            K: Option strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate (annualized).
            N: Number of steps in the binomil tree.
            sigma: Volatility of the underlying asset (annualized).
            option_type: "call" for a call option, "put" for a put option.

        Returns:
            delta: The delta of the option.
        """

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)

        # Initialize option values at expiration
        option_values = np.zeros(N + 1)
        for j in range(N + 1):
            stock_price = S * (u**(N - j)) * (d**j)
            if option_type == 'call':
                option_values[j] = max(0, stock_price - K)
            elif option_type == 'put':
                option_values[j] = max(0, K - stock_price)

        # Work backwards through the tree
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                stock_price = S * (u**(i - j)) * (d**j)
                # Calculate expected value from continuation
                continuation_value = np.exp(-r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
                # Calculate value if exercised early
                if option_type == 'call':
                    exercise_value = max(0, stock_price - K)
                elif option_type == 'put':
                    exercise_value = max(0, K - stock_price)
                # Take the maximum of continuation and exercise values
                option_values[j] = max(continuation_value, exercise_value)

        # Approximate delta using the first two nodes at time 0
        delta = (option_values[0] - option_values[1]) / (S * u - S * d)

        return delta

# Main
# Get option chain
option_chains_df = get_option_chains()
#print(option_chains_df)



# Save to MySQL
save_to_mysql(option_chains_df)

# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

# Write to log file when run as cron job
print("Option Chain Successfully Loaded", current_date)


