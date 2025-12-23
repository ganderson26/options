# option_chains.py
#
# Description: 
# Get Options Chain for historical data since the chain is not available after expiration.
#
# Steps:
# 1. Get current Option Chains for a collection of Stocks.
# 2. Save to CSV.
#
# crontab -e
# 30 14 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/yahoo_option_chains_mysql.py >> ~/codebase/options-examples/options/src/logfile.log 2>&1
#
# Requirements:
# pip install pymysql

import pandas as pd
import yfinance as yf 
import datetime
from sqlalchemy import create_engine

# Function to get current option chain
def get_option_chains():
    symbols =['AAPL','AMZN','GOOGL','META','MSFT','NVDA', 'TSLA']
    option_chains_df = pd.DataFrame()

    for symbol in symbols:
        # Get ticker
        ticker = yf.Ticker(symbol)
        #print('ticker', ticker)
        #print('info', ticker.info)
        # Get all chains for expiration dates
        expiration_dates = ticker.options
        # Get current expiration
        expiration_date = expiration_dates[0]
        option_chain = ticker.option_chain(expiration_date)
        # Create a DataFrame for calls and puts
        calls_df = pd.DataFrame(option_chain.calls)
        puts_df = pd.DataFrame(option_chain.puts)
        # Concatenate the calls and puts DataFrames
        option_chains_df = pd.concat([option_chains_df, calls_df, puts_df])
    
    return option_chains_df

# Function to save option chain to MySQL
def save_to_mysql(df):
    # Replace with your MySQL credentials and database name
    db_connection_str = 'mysql+pymysql://root:Marathon#262@localhost:3306/OPTIONS'
    engine = create_engine(db_connection_str)

    table_name = 'OPTION_CHAINS'  # Name of the table in MySQL

    #,contractSymbol,lastTradeDate,strike,lastPrice,bid,ask,change,percentChange,volume,openInterest,impliedVolatility,inTheMoney,contractSize,currency

    # Use to_sql() to write the DataFrame to the database
    # if_exists options: 'fail', 'replace', 'append'
    # index=False prevents writing the DataFrame index as a column in the table
    # Used 'replace' to initially create the table and column from the DataFrame
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

# Main
# Get option chain
option_chains_df = get_option_chains()

# Save to MySQL
save_to_mysql(option_chains_df)

# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

# Write to log file when run as cron job
print(f"Option Chain Successfully Loaded", current_date)
