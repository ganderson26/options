import yaml

from pyschwab.auth import Authorizer
from pyschwab.trading import TradingApi
from pyschwab.types import Symbol
from pyschwab.market import MarketApi

import os
from dotenv import load_dotenv
load_dotenv()

# Load configuration
with open("config/pyschwab.yaml", 'r') as file:
    app_config = yaml.safe_load(file)

# Authorization
# On the first run, this will open a browser for authorization.
# Follow the instructions. Subsequent runs will auto-refresh the access token.
# When refresh token expires, it will open browser for authorization again.
authorizer = Authorizer(app_config['auth'])

access_token = authorizer.get_access_token()
print(access_token)

# Example usage of market APIs
market_api = MarketApi(access_token, app_config['market'])

# Get quotes
symbols = ['TSLA', 'NVDA']
quotes = market_api.get_quotes(symbols)
for symbol in symbols:
    print("quote for ", symbol, ":", quotes[symbol])

# Get option chains
option_chain = market_api.get_option_chains('TSLA')
print(option_chain)

# Get price history 
history = market_api.get_price_history('TSLA')
print(history)