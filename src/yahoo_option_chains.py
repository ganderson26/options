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
# 30 14 * * 1-5 /Users/ganderson/anaconda3/bin/python ~/codebase/options-examples/options/src/yahoo_option_chains.py >> ~/codebase/options-examples/options/src/logfile.log 2>&1
#

import pandas as pd
import yfinance as yf 
import datetime

# Function to get current options chain
def get_option_chains():
    symbols =['AAPL','AMZN','GOOGL','META','MSFT','NVDA', 'TSLA']
    option_chains_df = pd.DataFrame()

    for symbol in symbols:
        # Get ticker
        ticker = yf.Ticker(symbol)
        print('ticker', ticker)
        print('info', ticker.info)
        # Get all chains for expiration dates
        #expiration_dates = ticker.options
        # Get current expiration
        #expiration_date = expiration_dates[0]
        #option_chain = ticker.option_chain(expiration_date)
        # Create a DataFrame for calls and puts
        #calls_df = pd.DataFrame(option_chain.calls)
        #puts_df = pd.DataFrame(option_chain.puts)
        # Concatenate the calls and puts DataFrames
        #option_chains_df = pd.concat([option_chains_df, calls_df, puts_df])
    
    #return option_chains_df
    return

# Main
option_chains_df = get_option_chains()
# Save to csv
# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
# Create the filename with the current date
#filename = f"~/options/option-chains-{current_date}.csv"
#option_chains_df.to_csv(filename)
# Write to log file
print("Option Chain Created", current_date)
