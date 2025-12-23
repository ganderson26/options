from schwab.auth import easy_client
from schwab.client import Client
from http import HTTPStatus
import datetime
import pandas as pd

# Follow the instructions on the screen to authenticate your client.
# Login with your Schwab account credentials and NOT your developer account.
client = easy_client(
        api_key='lOT4mnEUD2efP7iOu7bFCcYRvy0EZtDc',
        app_secret='4nPL5kqWxAhwfwxF',
        callback_url='https://127.0.0.1:8182',
        token_path='/tmp/token.json')

#resp = client.get_price_history_every_day('AAPL')
#assert resp.status_code == HTTPStatus.OK

# Get Option Chain.
symbol = 'PLTR'
start_date = datetime.datetime(2025, 4, 4)
end_date = datetime.datetime.now()
contract = Client.Options.ContractType.ALL
#print(contract)
#print(start_date)
#print(end_date)
resp = client.get_option_chain(symbol, contract_type=contract, strike_count=None, 
                               include_underlying_quote=None, strategy=None, interval=None, 
                               strike=None, strike_range=None, from_date=None, to_date=None, 
                               volatility=None, underlying_price=None, interest_rate=None, 
                               days_to_expiration=None, exp_month=None, option_type=None, entitlement=None)
#assert resp.status_code == HTTPStatus.OK
df_history = pd.DataFrame(resp.json())

print(df_history)

# Save to csv
# Get the current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
# Create the filename with the current date
filename = f"schwab-option-chains-{current_date}.csv"

df_history.to_csv(filename, index=False, encoding='utf-8')
