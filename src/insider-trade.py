import yfinance as yf
import finnhub
import pandas as pd
import requests

# First, get all stock symbols
# There are 11,683 symbols

# Alpha Vantage API Key (Sign up for a free API key)
api_key = 'D9NC3YOL0KEJ34MS'

# Function to get all stock symbols
def get_all_stock_symbols():
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'
    response = requests.get(url)
    data = response.text
      
    # Save data to a file
    with open('stock_symbols.csv', 'w') as f:
        f.write(data)

    # Create DataFrame
    df_symbols = pd.read_csv('stock_symbols.csv')
    
    # Remove unnecessary columns: name,exchange,assetType,ipoDate,delistingDate,status
    # Only keep symbol column
    df_symbols.drop('name', axis=1, inplace=True) 
    df_symbols.drop('exchange', axis=1, inplace=True)
    df_symbols.drop('assetType', axis=1, inplace=True)
    df_symbols.drop('ipoDate', axis=1, inplace=True)
    df_symbols.drop('delistingDate', axis=1, inplace=True)
    df_symbols.drop('status', axis=1, inplace=True)

    #print(df_symbols.shape) # 11,683 symbols

    # Convert to comma seperated string
    list = df_symbols.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(element.split()) for element in list]
    
    # Return
    return vals 

# Call the function
all_symbols = get_all_stock_symbols()
#print(all_symbols)




# Second, use all_symbols to populate tickers below...

# Set up your API key for Finnhub
finnhub_client = finnhub.Client(api_key='crq56thr01qutsn3f85gcrq56thr01qutsn3f860')

# Function to get short interest data
def get_short_interest(ticker):
    stock = yf.Ticker(ticker)
    short_interest = stock.info.get('shortPercentOfFloat', None)
    return short_interest

# Function to get insider buying data
def get_insider_buying(ticker):
    insider_transactions = finnhub_client.stock_insider_transactions(ticker, "2024-01-01", "2024-08-31")
    buying = [t for t in insider_transactions['data'] if t['change'] > 0]
    total_buying = sum(t['change'] for t in buying)
    return total_buying

# Function to screen stocks based on short interest and insider buying
def screen_stocks(tickers, short_interest_threshold=0.1, insider_buying_threshold=1000):
    shortlisted_stocks = []
    
    for ticker in tickers:
        short_interest = get_short_interest(ticker)
        insider_buying = get_insider_buying(ticker)
        
        if short_interest and short_interest > short_interest_threshold and insider_buying > insider_buying_threshold:
            shortlisted_stocks.append({
                'Ticker': ticker,
                'Short Interest (%)': short_interest * 100,
                'Insider Buying': insider_buying
            })
    
    return pd.DataFrame(shortlisted_stocks)

# List of stocks to screen (can be customized)
#tickers = ['AAPL', 'TSLA', 'GME', 'AMC']

# all_symbols
tickers = all_symbols

# Screen stocks
# Error
# finnhub.exceptions.FinnhubAPIException: FinnhubAPIException(status_code: 429)
# If your limit is exceeded, you will receive a response with status code 429. On top of all plan's limit, there is a 30 API calls/ second limit.
result = screen_stocks(tickers)
print(result)
