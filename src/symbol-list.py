import requests
import pandas as pd

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
print(all_symbols)

