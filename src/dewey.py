import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
 
# Function to get short interest from Yahoo Finance
def get_short_interest(ticker):
    stock = yf.Ticker(ticker)
    short_data = stock.get_info()  # Pull stock information
    short_interest = short_data.get('sharesShort', 0)
    return short_interest
 
# Scrape insider buy transactions from openinsider.com
def get_insider_buys(ticker):
    url = f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=0&td=365&f=0&v=0&sortcol=0&cnt=100&page=1"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
   
    # Parse the table for insider transactions
    table = soup.find('table', {'class': 'tinytable'})
    rows = table.find_all('th')[1:]  # Exclude header
   
    buys = []
    for row in rows:
        columns = row.find_all('td')
        if 'Buy' in columns[6].text:  # Filter for 'Buy' transactions
            buys.append({
                'Date': columns[1].text,
                'Insider Name': columns[2].text,
                'Shares Traded': columns[8].text,
                'Price': columns[9].text,
            })
    return buys
 
# Function to find heavily shorted stocks with insider buys
def correlate_short_interest_with_insider_buys(tickers):
    results = []
    for ticker in tickers:
        short_interest = get_short_interest(ticker)
        insider_buys = get_insider_buys(ticker)
       
        if short_interest > 1000000 and len(insider_buys) > 0:  # Example filter criteria
            results.append({
                'Ticker': ticker,
                'Short Interest': short_interest,
                'Insider Buys': insider_buys
            })
    return pd.DataFrame(results)
 
# Example tickers to analyze
tickers = ['GME', 'AMC', 'TSLA']
 
# Correlating heavily shorted stocks with insider buys
results_df = correlate_short_interest_with_insider_buys(tickers)
 
# Display results
print(results_df)