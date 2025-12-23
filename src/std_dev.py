import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- Input Parameters ---
S = 93.78  # Current stock price of PLTR (update this with latest)
K = 93.78  # Strike price
T_date = datetime(2025, 4, 25)
today = datetime.today()
T = (T_date - today).days / 365  # Time to expiration in years

r = 0.045  # Risk-free interest rate (e.g., 10-year Treasury yield ~4.5%)
sigma = 100.30  # Implied volatility (update with actual PLTR IV, e.g., 55%)

# --- Calculate probability of ending above strike (ITM for call option) ---
d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
prob_ITM_call = 1 - norm.cdf(d2)

# --- Output ---
print(f"Probability PLTR will be in the money (above ${K}) on {T_date.date()}: {prob_ITM_call:.2%}")
