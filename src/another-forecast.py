import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm

def black_scholes_put(S, K, T, r, sigma):
    """Calculate the Black-Scholes price of a European put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_best_strike_to_sell_put(ticker, days_to_expiration, risk_free_rate=0.03):
    """Calculate the best strike price to sell a put option."""

    # Fetch historical data
    data = yf.download(ticker, period="1y")

    # Calculate historical volatility
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    sigma = np.std(data['LogReturn'].dropna()) * np.sqrt(252)  # Annualized volatility

    # Get the current stock price
    current_price = data['Close'].iloc[-1]

    # Define a range of strike prices (e.g., 80% to 120% of the current price)
    strike_prices = np.linspace(0.8 * current_price, 1.2 * current_price, 20)

    # Time to expiration in years
    T = days_to_expiration / 365

    # Prepare results
    results = []

    for K in strike_prices:
        put_price = black_scholes_put(current_price, K, T, risk_free_rate, sigma)
        potential_profit = put_price
        potential_loss = max(0, K - current_price) - put_price
        risk_reward_ratio = potential_profit / (abs(potential_loss) + 1e-10)
        results.append({
            "Strike Price": K,
            "Option Premium": put_price,
            "Potential Profit": potential_profit,
            "Potential Loss": potential_loss,
            "Risk-Reward Ratio": risk_reward_ratio
        })

    results_df = pd.DataFrame(results)

    print(results_df)

    # Find the strike price with the best risk-reward ratio
    best_strike = results_df.loc[results_df['Risk-Reward Ratio'].idxmax()]

    print("Best Strike Price to Sell a Put")
    print("--------------------------------")
    print(f"Ticker: {ticker}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Volatility (annualized): {sigma:.2%}")
    print(f"Days to Expiration: {days_to_expiration}")
    print("\nBest Strike Price:")
    print(best_strike)

    return results_df, best_strike

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"  # Replace with your stock ticker
    days_to_expiration = 30  # Replace with desired days to expiration

    results, best_strike = calculate_best_strike_to_sell_put(ticker, days_to_expiration)

    # Save the results to a CSV file
    results.to_csv("best_strike_to_sell_put.csv", index=False)
    print("\nResults saved to 'best_strike_to_sell_put.csv'")
