# Importing necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf

# Function to get historical stock price data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    return data

# Ticker symbol of the stock you want to predict
ticker = "AAPL"
# Date range for historical data (format: "YYYY-MM-DD")
start_date = "2009-09-14"
end_date = "2019-09-12"

# Getting historical stock data
stock_data = get_stock_data(ticker, start_date, end_date)

# Save to CSV
stock_data.to_csv(f"{ticker}_financial.csv")
#print(stock_data)

#df = pd.read_csv(f"{ticker}_financial.csv")
#print(df)

df1 = stock_data.reset_index()
# If the new column is not named 'index' or a custom name, rename it
# (This step is often not needed if the original index had a name)
if 'index' in df1.columns:
    df1.rename(columns={'index': 'Date'}, inplace=True)


print(df1.columns)
print(df1)
df1.to_csv(f"{ticker}_financial.csv")

# Resetting the index to a default integer index (optional, for demonstration)
#data_reset = stock_data.reset_index()

# Setting the 'Date' column as the new index
#data_with_date_index = data_reset.set_index('Date')

# Display the head of the DataFrame to verify
#print(data_with_date_index.head())

#print(data_with_date_index.columns)

##print(data_with_date_index)



# Download data for a specific ticker (e.g., Apple - AAPL)
#ticker = "AAPL"
#data = yf.download(ticker, start="2024-01-01", end="2025-01-01", auto_adjust=False)

# The 'data' DataFrame will contain the requested columns including 'Adj Close'
#print(data.head())