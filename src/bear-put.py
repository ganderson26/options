# Started with the following ChatGPT prompt:
#write a script that shows best bearish put option with low risk with an expiration date of 10/13/2023

import yfinance as yf

def find_best_bearish_put_option(ticker, expiration_date):
    try:
        # Get options data for the specified expiration date
        put_options = yf.Ticker(ticker).option_chain(expiration_date).puts
        
        # Filter bearish put options (example criteria: Delta < -0.5 and Open Interest > 100)
        bearish_puts = put_options[(put_options['delta'] < -0.5) & (put_options['openInterest'] > 100)]
        
        # Sort options by volume in descending order to find the most liquid ones
        sorted_bearish_puts = bearish_puts.sort_values(by='volume', ascending=False)
        
        # Print the best bearish put option
        if not sorted_bearish_puts.empty:
            print("Best Bearish Put Option:")
            print(sorted_bearish_puts.iloc[0])
        else:
            print("No suitable bearish put options found for the given criteria.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    ticker = "ORCL"  # Replace with the desired stock symbol
    expiration_date = "2023-10-13"  # Replace with the desired expiration date
    find_best_bearish_put_option(ticker, expiration_date)
