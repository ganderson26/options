import yfinance as yf
import pandas as pd

ticker = 'MSTR'
expirationDate = '2024-12-20'
strike = 300

options = pd.DataFrame()

tk = yf.Ticker(ticker)  
option_chain = tk.option_chain(expirationDate)
calls_df = pd.DataFrame(option_chain.calls)
puts_df = pd.DataFrame(option_chain.puts)

options = pd.concat([calls_df, puts_df], ignore_index=True, axis=0)
options['expirationDate'] = expirationDate
options['symbol'] = ticker

strike_df = options[options['strike'] == strike]

print(strike_df)


