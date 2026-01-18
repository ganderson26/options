# rebuild_database.py
#
# Description: 
# Rebuild database to include 2 standard deviations and filter out rows that are not between the lower and upper 2nd SD. 
#


import numpy as np
import pandas as pd
import yfinance as yf 
import datetime
from scipy.stats import norm
from sqlalchemy import create_engine

def add_2_sd(df):
    
    results = []

    # Loop through the collection
    for index, row in df.iterrows():
    
        lower_95, upper_95 = analyze_price_range(row['currentPrice'], row['annualized_volatility'], row['days_to_expiry'], z_score=2)
        
        # Add to each row since these will be the same regardless of strike
        df.at[index, 'lower_95'] = lower_95
        df.at[index, 'upper_95'] = upper_95

    # Filter for only rows where between lower_95 and upper_95 (2 standard deviations)
    df = df[(df['strike'] >= df['lower_95']) & (df['strike'] <= df['upper_95'])]

    return df

def analyze_price_range(current_price, iv, days_to_expiry, z_score):

    # Use the year
    time_in_years = days_to_expiry / 365
    
    
    # Use implied volatility to calculate the potential price movement
    range_move = iv * np.sqrt(time_in_years) * z_score
    
    
    # Calculate the upper and lower bounds based on the lognormal distribution assumption
    upper_bound = current_price * np.exp(range_move)
    lower_bound = current_price * np.exp(-range_move)
    
    return lower_bound, upper_bound

def read_from_mysql():

    # Replace with your MySQL credentials and database name
    db_connection_str = 'mysql+pymysql://root:Marathon#262@localhost:3306/OPTIONS'
    engine = create_engine(db_connection_str)

    # Name of the table in MySQL
    table_name = 'OPTION_CHAINS'  

    option_chains_df = pd.read_sql_table(table_name, con=engine)
    
    return option_chains_df

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
    table_name = 'OPTION_CHAINS_2SD'  

    # Will create the following columns
    # contractSymbol,lastTradeDate,strike,lastPrice,bid,ask,change,percentChange,volume,openInterest,impliedVolatility,inTheMoney,contractSize,currency,currentPrice,annualized_volatility,expiry_date,days_to_expire,lower_68,upper_68

    # Use to_sql() to write the DataFrame to the database
    # if_exists options: 'fail', 'replace', 'append'
    # index=False prevents writing the DataFrame index as a column in the table
    # Used 'replace' to initially create the table and column from the DataFrame
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

# Main

# Get option chain
option_chains_df = read_from_mysql()

# Get 2 SD
filtered_option_chains_2sd_df = add_2_sd(option_chains_df)

# Save to MySQL
save_to_mysql(filtered_option_chains_2sd_df)

# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

# Write to log file when run as cron job
# Write to console when run from terminal
print("Option Chain Successfully Rebuilt", current_date)


