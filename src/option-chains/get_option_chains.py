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
# 30 14 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#
# crontab for Monday to Friday at 12:00
# crontab -e
# 0 12 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#
# crontab for Friday at 12:00
# crontab -e
# 0 12 * * 5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/option-chains/get_option_chains.py >> ~/codebase/options-examples/options/src/option-chains/logfile.log 2>&1
#

import numpy as np
import pandas as pd
import yfinance as yf 
import datetime
from sqlalchemy import create_engine

# Get current option chain
def get_option_chains():
    symbols =['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
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
        
        # Concatenate the calls and puts DataFrames
        option_chains_df = pd.concat([option_chains_df, calls_df, puts_df])

        #print(option_chains_df)
                    
    return option_chains_df

# Get current price
def get_current_price(ticker):
    current_price = ticker.info.get('currentPrice')

    #print(current_price)

    return current_price

# Get annualized volatility
def get_annualized_volatility(symbol):

    # Get historical stock data
    data = yf.download(symbol, period='1y', auto_adjust=True)['Close']

    # Calculate daily logarithmic returns
    log_returns = np.log(data / data.shift(1)).dropna()

    # Compute the standard deviation of the returns
    daily_volatility = log_returns.std()

    # Annualize the volatility
    trading_days = 252  # Standard for most markets
    annualized_volatility = daily_volatility * np.sqrt(trading_days)

    # Get 1st value in Series
    return annualized_volatility[0]

# Calculates the price range for a given probability
def analyze_price_range(current_price, iv, days_to_expiry, z_score):
    # Use the year
    time_in_years = days_to_expiry / 365
    
    # Use implied volatility to calculate the potential price movement
    range_move = iv * np.sqrt(time_in_years) * z_score
    
    # Calculate the upper and lower bounds based on the lognormal distribution assumption
    upper_bound = current_price * np.exp(range_move)
    lower_bound = current_price * np.exp(-range_move)
    
    return lower_bound, upper_bound

# Get date for next Friday
def get_next_friday_date():
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

# Save option chain to MySQL
def save_to_mysql(df):
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


