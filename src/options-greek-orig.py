from yahoo_fin import options
from yahoo_fin import stock_info as si
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import *

stock = 'MSTR'
Day = 20
month = 12
year = 2024
today = date.today()
future = date(year,month,Day)
expiry = future
str(future - today)
pd.set_option("display.max_rows", None, "display.max_columns", None)
options.get_options_chain(stock)
chain = options.get_options_chain(stock, expiry)
#chain_df = pd.DataFrame.from_dict(chain)

#print(type(chain)) #dict
strike = 300

print(stock)
print(strike)
print(today)
print(future)

# Loop thur chain for specific Strike
#for x in chain.items():
    #if x['Strike'] == strike:
    #    print('hit')
        

s = chain["puts"]['Implied Volatility']
print(type(s))
print(len(s))
strikes = chain["puts"]['Strike']
print(type(strikes))
print(len(strikes))




for i in range(len(s)):
    print(i, ' ', s[i])

for i in range(len(strikes)):
    print(i, ' ', strikes[i])   

#s_strikes = pd.DataFrame(strikes)
#print(s_strikes)

#for i in range(len(s)):
    #s_strikes['IV'] = s[i] 

s_strikes = pd.DataFrame({'Strike': strikes, 'IV': s})   

print(s_strikes)    
 
         

#print(s_strikes.info())
#print(s_strikes['Strike'])

filtered_s_strings = s_strikes[(s_strikes.Strike == 300)]
print(filtered_s_strings)
#s_strikes = pd.concat([s, strikes])
#s_strikes = float(s) + float(strikes)
#print(s_strikes)
#filtered_s = chain["calls"].Strike == strike

#s = chain["calls"]['Implied Volatility']


#print(chain["calls"].Strike)

r = .025
S = si.get_live_price(stock)
K = chain["puts"].Strike
t = float((future - today).days)
T = t/365


sigma = chain["puts"]["Implied Volatility"].apply(lambda x: float(x[:-1]) / 100)

#sigma = filtered_s_strings["IV"].apply(lambda x: float(x[:-1]) / 100)

def delta_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    r_delta_calc = norm.cdf(d1, 0, 1) 
    #print(type(r_delta_calc))
    #for i in range(len(r_delta_calc)):
    #    print(i, '  ', r_delta_calc[i]) 

    #r_delta_calc = r_delta_calc[~np.isnan(r_delta_calc)]

    return r_delta_calc

print("Option Delta is: ", (delta_calc(r, S, K, T, sigma)))
