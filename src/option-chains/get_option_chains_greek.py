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
    
    # Create a collection of stock symbols to build option chain data
    symbols =['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
       
    # Create an empty DataFrame that will be populated
    option_chains_df = pd.DataFrame()

    # Loop through the collection of symbols
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

        # Add to each row since these will be the same regardless of strike
        calls_df['currentPrice'] = current_price
        puts_df['currentPrice'] = current_price

        # Get annualized_volatility
        annualized_volatility = get_annualized_volatility(symbol)
        
        # Add to each row since these will be the same regardless of strike
        calls_df['annualized_volatility'] = annualized_volatility
        puts_df['annualized_volatility'] = annualized_volatility

        # Calculate price range
        # Get the date for next Friday and the number of days to expiration
        days_to_expiry, expiry_date = get_next_friday_date() 

        # z_score=1 for 68.27% probability range (one standard deviation)
        # Must be a valid expiry date
        lower_68, upper_68 = analyze_price_range(current_price, annualized_volatility, days_to_expiry, z_score=1)
        
        # Add to each row since these will be the same regardless of strike
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
            # Use American Option algorithm
            strike = row['strike']
            iv = round(annualized_volatility, 2)
            volume = row['volume']
            bid = row['bid']
            rate = 0.05 # Risk-free rate (5%)
            time = float(days_to_expiry)/365 # Time to expiration (years)
            steps = 100 # Number of steps in the binomial tree
     
            delta_put = round(american_option_delta_binomial(current_price, strike, time, rate, iv, steps, 'put'), 2)
            # Update row
            puts_df.at[index, 'delta'] = delta_put

            delta_call = round(american_option_delta_binomial(current_price, strike, time, rate, iv, steps, 'call'), 2)
            # Update row
            calls_df.at[index, 'delta'] = delta_call        
        
        # Concatenate the calls and puts DataFrames
        option_chains_df = pd.concat([option_chains_df, calls_df, puts_df])
                    
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

def american_option_delta_binomial(current_price, strike, time, rate, iv, steps, option_type='call'):
        """
        Calculating the delta for American options in Python typically requires numerical methods, as 
        there are no simple closed-form solutions like for European options (Black-Scholes). The most 
        common methods are:
        Binomial or Trinomial Tree Models:
            - The binomial option pricing model uses an iterative procedure, allowing for the specification 
              of nodes, or points in time, between the valuation date and the option's expiration date.
            - These models build a lattice of possible stock prices over time.
            - At each node, the option value is calculated by working backward from expiration, considering 
              the possibility of early exercise for American options.
            - The delta can be approximated by observing the change in option price at a given node when the 
              underlying stock price changes by one step in the tree.

        Least Squares Monte Carlo (LSM):
            - This method simulates multiple stock price paths using Monte Carlo simulation.
            - At each time step, it uses least squares regression to estimate the conditional expected value 
              of continuing the option.
            - The option value is then determined by comparing the continuation value to the exercise value.
            - Delta can be estimated by re-running the LSM simulation with a small perturbation to the initial 
              stock price and observing the change in option price.    

        Finite Difference Methods:
            - These methods discretize the partial differential equation (PDE) that governs option pricing 
              (e.g., Black-Scholes PDE with early exercise conditions).
            - Delta is then obtained directly from the numerical solution of the PDE.  

        Note on Delta Calculation:
        For American options, the delta is more complex than for European options due to the early exercise 
        feature. The delta will reflect the sensitivity to the underlying price while also accounting for the 
        optimal exercise boundary. Increasing the number of steps (N) in binomial/trinomial trees generally 
        improves the accuracy of the delta approximation.                   

        Args:
            current_price: Current underlying asset price.
            strike: Option strike price.
            time: Time to expiration in years.
            rate: Risk-free interest rate (annualized).
            iv: Volatility of the underlying asset (annualized).
            steps: Number of steps in the binomil tree.
            option_type: "call" for a call option, "put" for a put option.

        Returns:
            delta: The delta of the option.
        """

        # Set up the building blocks of the binomial model
        # This is the foundation needed to price an option (and compute delta) with the binomial lattice method

        # Length of each time step in the binomial tree
        # If time=1 year and steps=1000 steps, each step is 1/1000 year
        length_of_each_time = time / steps
        # Up factor: how much the stock price moves up in one step
        # Formula comes from Cox-Ross-Rubinstein (CRR) binomial model
        # If volatility is high or step is long, u is larger
        up_factor = np.exp(iv * np.sqrt(length_of_each_time))
        # Down factor: how much the stock price moves down in one step
        # This ensures the model is recombining (a node reached by up then down is same as down then up)
        down_factor = 1 / up_factor
        # Risk-neutral probability of an up move
        # Ensures the expected stock return under this measure equals the risk-free rate
        risk_neutral_probability = (np.exp(rate * length_of_each_time) - down_factor) / (up_factor - down_factor)

        # Build the option price tree backward (from maturity → present)

        # Initialize option values at expiration
        option_values = np.zeros(steps + 1)
        for j in range(steps + 1):
            stock_price = current_price * (up_factor**(steps - j)) * (down_factor**j)
            if option_type == 'call':
                option_values[j] = max(0, stock_price - strike)
            elif option_type == 'put':
                option_values[j] = max(0, strike - stock_price)

        # Work backwards through the tree
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                stock_price = current_price * (up_factor**(i - j)) * (down_factor**j)
                # Calculate expected value from continuation
                continuation_value = np.exp(-rate * length_of_each_time) * (risk_neutral_probability * option_values[j] + (1 - risk_neutral_probability) * option_values[j + 1])
                # Calculate value if exercised early
                if option_type == 'call':
                    exercise_value = max(0, stock_price - strike)
                elif option_type == 'put':
                    exercise_value = max(0, strike - stock_price)
                # Take the maximum of continuation and exercise values
                option_values[j] = max(continuation_value, exercise_value)

        # Approximate delta using the first two nodes at time 0
        delta = (option_values[0] - option_values[1]) / (current_price * up_factor - current_price * down_factor)

        return delta

# Main
"""
Main entry point to program.
    - Build the option chain
    - Save to MySQL
    - Get the current date
    - Write a log messsage with the current date to the logfile
"""

# Get option chain
option_chains_df = get_option_chains()

# Save to MySQL
save_to_mysql(option_chains_df)

# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

# Write to log file when run as cron job
print("Option Chain Successfully Loaded", current_date)


