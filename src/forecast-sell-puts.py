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

def forecast_sell_put_strategy_over_year(ticker, risk_free_rate=0.03):
    """Forecast returns for a sell put strategy over a 1-year period with varying strikes."""

    # Fetch historical data
    data = yf.download(ticker, period="1y")

    # Calculate historical volatility
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    sigma = np.std(data['LogReturn'].dropna()) * np.sqrt(252)  # Annualized volatility

    # Get the current stock price
    current_price = data['Close'].iloc[-1]

    # Define a range of strike prices (e.g., 80% to 120% of the current price)
    strike_prices = np.linspace(0.8 * current_price, 1.2 * current_price, 10)

    # Time to expiration intervals (e.g., 30, 60, 90 days)
    expirations = [30, 60, 90, 180, 365]

    # Prepare a results DataFrame
    results = []

    for days_to_expiration in expirations:
        T = days_to_expiration / 365
        for K in strike_prices:
            put_price = black_scholes_put(current_price, K, T, risk_free_rate, sigma)
            potential_profit = put_price
            potential_loss = max(0, K - current_price) - put_price

            risk_reward_ratio = potential_profit / (abs(potential_loss) + 1e-10)

            results.append({
                "Days to Expiration": days_to_expiration,
                "Strike Price": K,
                "Option Premium": put_price,
                "Potential Profit": potential_profit,
                "Potential Loss": potential_loss,
                "Risk Reward Ratio": risk_reward_ratio
            })

    results_df = pd.DataFrame(results)

    

    # Find the optimal strike price for each expiration
    #optimal_strikes = results_df.groupby("Days to Expiration").apply(
    #    lambda x: x.loc[x['Potential Profit'] / (x['Potential Loss'] + 1e-10).abs().idxmax()]
    #)


    print(results_df)

    optimal_strikes = results_df.groupby("Days to Expiration")['Risk Reward Ratio'].idxmax()
    

    print("Forecast for Sell Put Strategy Over 1 Year")
    print("------------------------------------------")
    print(f"Ticker: {ticker}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Volatility (annualized): {sigma:.2%}")
    print("\nOptimal Strike Prices:")
    print(results_df.loc[optimal_strikes])

    return results_df, optimal_strikes

# Example usage
if __name__ == "__main__":
    ticker = "MSTR"  # Replace with your stock ticker

    results, optimal_strikes = forecast_sell_put_strategy_over_year(ticker)

    # Save the results to a CSV file
    results.to_csv("sell_put_strategy_forecast.csv", index=False)
    print("\nResults saved to 'sell_put_strategy_forecast.csv'")
