import yfinance as yf
import pandas as pd
import numpy as np

def backtest_wheel_strategy(ticker, start_date, end_date, cash=10000, put_strike_offset=0.95, call_strike_offset=1.05):
    # Fetch historical stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    
    balance = cash
    shares_held = 0
    option_premium_income = 0
    trade_log = []
    
    for i in range(1, len(stock_data)):
        price = stock_data['Adj Close'].iloc[i]
        prev_price = stock_data['Adj Close'].iloc[i-1]
        
        # Sell cash-secured puts if no shares are held
        if shares_held == 0:
            put_strike = price * put_strike_offset
            premium = price * 0.02  # Assume 2% premium income
            option_premium_income += premium
            trade_log.append(f"Sold Put at {put_strike:.2f}, Collected Premium {premium:.2f}")
            
            # If assigned (stock price drops below put strike), buy 100 shares
            if price < put_strike:
                shares_held = 100
                balance -= price * 100
                trade_log.append(f"Assigned at {price:.2f}, Bought 100 shares")
        
        # Sell covered calls if shares are held
        elif shares_held > 0:
            call_strike = price * call_strike_offset
            premium = price * 0.02  # Assume 2% premium income
            option_premium_income += premium
            trade_log.append(f"Sold Call at {call_strike:.2f}, Collected Premium {premium:.2f}")
            
            # If stock price rises above call strike, shares are called away
            if price > call_strike:
                shares_held = 0
                balance += price * 100
                trade_log.append(f"Called away at {price:.2f}, Sold 100 shares")
    
    final_balance = balance + shares_held * stock_data['Adj Close'].iloc[-1] + option_premium_income
    return final_balance, trade_log

# Example usage
final_balance, trade_log = backtest_wheel_strategy("AAPL", "2022-01-01", "2023-01-01")
print(f"Final Balance: ${final_balance:.2f}")
for trade in trade_log[-10:]:  # Show last 10 trades
    print(trade)
