from flask import Flask

from yahoo_fin import options
from yahoo_fin import stock_info as si
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import *

stock = 'MSTR'
strike = 300
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

s = chain["puts"]['Implied Volatility']
strikes = chain["puts"]['Strike']
volume = chain["puts"]['Volume']
bids = chain["puts"]['Bid']

s_strikes = pd.DataFrame({'Strike': strikes, 'IV': s, 'Volume': volume, 'Bid': bids})  

filtered_s_strings = s_strikes[(s_strikes.Strike == 300)]

r = .025
S = si.get_live_price(stock)
K = chain["puts"].Strike
t = float((future - today).days)
T = t/365

sigma = filtered_s_strings["IV"].apply(lambda x: float(x[:-1]) / 100)

def delta_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    r_delta_calc = norm.cdf(d1, 0, 1) 

    r_delta_calc = r_delta_calc[~np.isnan(r_delta_calc)]

    for i in range(len(r_delta_calc)):
        r_delta_calc[i] = 1.00 - r_delta_calc[i]

    return r_delta_calc




app = Flask(__name__)

@app.route("/")
def hello_world():
    delta = delta_calc(r, S, K, T, sigma)
    filtered_s_strings['Delta'] = delta
    filtered_s_strings['ROR'] = (filtered_s_strings['Bid'] / filtered_s_strings['Strike']) * 100

    return "<p>" + stock + " " + expiry + " " + strike + " PUT " + filtered_s_strings + "</p>"
