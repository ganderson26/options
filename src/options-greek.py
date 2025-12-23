from yahoo_fin import options
from yahoo_fin import stock_info as si
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import *

stock = 'AAPL'
strike = 300
Day = 1
month = 9
year = 2025
today = date.today()
future = date(year,month,Day)
expiry = future
str(future - today)
pd.set_option("display.max_rows", None, "display.max_columns", None)
#options.get_options_chain(stock)
chain = options.get_options_chain(stock, expiry)

print(stock)
print(strike)
print(today)
print(future)


s = chain["puts"]['Implied Volatility']
strikes = chain["puts"]['Strike']
volume = chain["puts"]['Volume']
bids = chain["puts"]['Bid']

s_strikes = pd.DataFrame({'Strike': strikes, 'IV': s, 'Volume': volume, 'Bid': bids})  

filtered_s_strings = s_strikes[(s_strikes.Strike == 300)]
print(filtered_s_strings)



r = .025
S = si.get_live_price(stock)
K = chain["puts"].Strike
t = float((future - today).days)
T = t/365




sigma = filtered_s_strings["IV"].apply(lambda x: float(x[:-1]) / 100)

def delta_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    r_delta_calc = norm.cdf(d1, 0, 1) 
    #print(type(r_delta_calc))
    #for i in range(len(r_delta_calc)):
    #    print(i, '  ', r_delta_calc[i]) 

    r_delta_calc = r_delta_calc[~np.isnan(r_delta_calc)]

    for i in range(len(r_delta_calc)):
        r_delta_calc[i] = 1.00 - r_delta_calc[i]

    return r_delta_calc


delta = delta_calc(r, S, K, T, sigma)
#print("Option Delta is: ", (delta_calc(r, S, K, T, sigma)))

filtered_s_strings['Delta'] = delta

filtered_s_strings['ROR'] = (filtered_s_strings['Bid'] / filtered_s_strings['Strike']) * 100

print(filtered_s_strings)
