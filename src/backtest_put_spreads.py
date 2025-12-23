# Backtest Weekly Put Credit Spreads for selected stock
# Table is appended every Friday afternoon with option chains
# expiry_date an is the following Friday
# days_to_expiry is always 7
# current_price is the stock price at the time the table is appended
#
# Get the stock rows for Puts
# Get the short (sell put bid) for ATM
# Get the long (buy put ask) for the previous strike which should be $2.5 less

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------------
SYMBOL = 'AMZN'
PATTERN_CHARACTER_LENGTH = '10' # 11 for 5 character symbols like GOOGL
CONTRACT_MULTIPLIER = 100
TRADE_DATE_START = None
TRADE_DATE_END = None
# ATM Strikes with 2.5 spread
# This fails for META, missing rows due to 5 strikes
#ATM_SPREAD = '1 Strike below ATM'
#SHORT_STRIKE_ATM_DISTANCE_START = 0.0
#SHORT_STRIKE_ATM_DISTANCE_END = 2.5
#LONG_STRIKE_ATM_DISTANCE_START = 2.5
#LONG_STRIKE_ATM_DISTANCE_END = 5.0

# ATM Strikes with 5 spread
# This picks up 2.5 for some stocks like AAPL and 5 for META
ATM_SPREAD = '1 Strike below ATM'
SHORT_STRIKE_ATM_DISTANCE_START = 0.0
SHORT_STRIKE_ATM_DISTANCE_END = 5.0
LONG_STRIKE_ATM_DISTANCE_START = 5.0
LONG_STRIKE_ATM_DISTANCE_END = 10.0

# 2 strikes below ATM Strikes with 5 spread
#ATM_SPREAD = '2 Strikes below ATM'
#SHORT_STRIKE_ATM_DISTANCE_START = 10.0
#SHORT_STRIKE_ATM_DISTANCE_END = 15.0
#LONG_STRIKE_ATM_DISTANCE_START = 15.0
#LONG_STRIKE_ATM_DISTANCE_END = 20.0

# ---------------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------------
def load_data():
    # Connect to database
    db_connection = 'mysql+pymysql://root:Marathon#262@localhost:3306/OPTIONS'
    engine = create_engine(db_connection)

    # Get all rows from table
    sql_query = "SELECT * FROM OPTION_CHAINS"
    df = pd.read_sql(sql_query, con=engine)
    
    return df

# ---------------------------------------------------------------------------------
# Filter Data
#
# Get the stock rows for Puts
# ---------------------------------------------------------------------------------
def filter_data(df):

    #symbol = 'AAPL'
    # 10 for 4 character symbols
    # 11 for 5 character symbols
    #pattern_character_length = '10'
    
    # Get just AAPL and drop any NaN
    filtered_stock_df =  df[df['contractSymbol'].str.startswith(SYMBOL, na=False)]

    # Get just Puts
    substring_to_find = 'P'

    # Regex explanation:
    # ^      - Matches the start of the string
    # .{10}   - Matches any ten characters
    # word   - Matches the literal substring "word"
    pattern = '^.{' + PATTERN_CHARACTER_LENGTH + '}' + substring_to_find
    filtered_puts_df =  filtered_stock_df[filtered_stock_df['contractSymbol'].str.contains(pattern, na=False)]

    # Get just 7 days expiration
    filtered_7_days_df = filtered_puts_df[filtered_puts_df['days_to_expiry'] == 7]

    return filtered_7_days_df

# ---------------------------------------------------------------------------------
# Get Spread Strikes
#
# Get the short (sell put bid) for ATM current_price previous strike closest to current_price
# Get the long (buy put ask) for the previous strike which should be $2.5 less
#
# Return DataFrame with 2 row pairs, 1 for long and 1 for short
# ---------------------------------------------------------------------------------
def get_spread_strikes(df):

    results = []
    
    for index, row in df.iterrows():
        
        strike = row['strike']
        current_price = row['currentPrice']
       
        # Short strike
        if abs(current_price - strike) >= SHORT_STRIKE_ATM_DISTANCE_START and abs(current_price - strike) <= SHORT_STRIKE_ATM_DISTANCE_END:
            if (strike <= current_price):
                closest_strike = strike
                results.append(row.to_dict())
        else:
            # Long strike needs to be lower
            if abs(current_price - strike) >= LONG_STRIKE_ATM_DISTANCE_START and abs(current_price - strike) <= LONG_STRIKE_ATM_DISTANCE_END:
                if (strike <= current_price):
                    closest_strike = strike
                    results.append(row.to_dict())                           

    spreads_df = pd.DataFrame(results)
    #print(spreads_df)

    return spreads_df

# ---------------------------------------------------------------------------------
# Calculate Premiums
#
# Loop through results_df getting the symbol, date, strike, bid, ask and price
# Need the first 2 rows to calculate premium
#
# Return 1 row with combined data from the 2 row pair
# ---------------------------------------------------------------------------------
def calculate_premiums(df):

    # Get first and last trade dates for report
    global TRADE_DATE_START
    global TRADE_DATE_END
    

    prev_row_1 = None
    prev_row_2 = None
    premiums_results = []

    # Recast lastTradeDate to just YYYY-MM-DD for comparison
    # Format as Year-Month-Day string
    df['lastTradeDate'] = df['lastTradeDate'].dt.strftime('%Y-%m-%d') 

    TRADE_DATE_START = df['lastTradeDate'].iloc[0]
    TRADE_DATE_END = df['lastTradeDate'].iloc[-1]   

    for index, row in df.iterrows():

        if prev_row_1 is not None:
            if (row['lastTradeDate'] != prev_row_1['lastTradeDate']):

                # Calculate Premium
                short_bid = prev_row_1['bid']
                long_ask = prev_row_2['ask']
                premium = (short_bid - long_ask) * CONTRACT_MULTIPLIER
            
                # Calculate Cost 
                # If currentPrice >= expirationPrice, $0.00
                cost = 0.00
                net = 0.00
                current_price = prev_row_1['currentPrice']
                expirationPrice = row['currentPrice']
                short_strike = prev_row_1['strike']
                long_strike = prev_row_2['strike']
                if (current_price >= expirationPrice):
                    cost = 0.00
                else:
                    # Calculate difference between shortStrike and LongStrike
                    cost =  (short_strike - long_strike) * CONTRACT_MULTIPLIER  

                net = premium - cost
                
                new_row = {'contractSymbol': prev_row_1['contractSymbol'], 'lastTradeDate': prev_row_1['lastTradeDate'], 
                        'currentPrice': prev_row_1['currentPrice'], 'expirationPrice': row['currentPrice'],
                        'shortStrike': prev_row_1['strike'], 'shortBid': prev_row_1['bid'], 'shortAsk': prev_row_1['ask'],
                        'longStrike': prev_row_2['strike'], 'longBid': prev_row_2['bid'], 'longAsk': prev_row_2['ask'],
                        'premium': premium, 'cost': cost, 'net': net}
                
                premiums_results.append(new_row)
            
            # Update prev_row tracker at the end of the loop
            prev_row_2 = prev_row_1

        prev_row_1 = row   # Current row is now previous row 1 
        
    
    #print(current_row)
    premiums_df = pd.DataFrame(premiums_results)
    
    return premiums_df

# -------------------------
# Main
# -------------------------
def main():
    #global TRADE_DATE_START
    #global TRADE_DATE_END

    

    load_df = load_data()
    filtered_df = filter_data(load_df)
    spread_df = get_spread_strikes(filtered_df)
    premiums_df = calculate_premiums(spread_df)

    premiums_df.to_csv('backtest_results_df.csv')  

    # -------------------------
    # RESULTS
    # -------------------------

    print(ATM_SPREAD + " PUT Credit Bear Spread Strategy for " + SYMBOL + " from Option Chain History from " + TRADE_DATE_START + " to " + TRADE_DATE_END )
    print("Total Trades:", len(premiums_df))
    print("Total PnL:", premiums_df["net"].sum())

    print("")
    print("Option Chain")
    print(premiums_df)

    plt.figure(figsize=(10, 6))
    plt.plot(premiums_df['lastTradeDate'], premiums_df['net'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Net')
    plt.title('Weekly Net')
    plt.xticks(rotation=45) # Rotate date labels for readability
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
