import yfinance as yf

def find_low_risk_put_options(ticker, expiration_date):
    try:
        # Get options data for the specified expiration date
        options = yf.Ticker(ticker).options
        put_options = yf.Ticker(ticker).option_chain(expiration_date).puts
        
        # Filter low-risk put options (example criteria: Delta < 0.3 and Open Interest > 100)
        low_risk_puts = put_options[(put_options['delta'] < 0.3) & (put_options['openInterest'] > 100)]
        
        # Print the low-risk put options
        print(low_risk_puts)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    ticker = "ORCL"  # Replace with the desired stock symbol
    expiration_date = "2023-10-13"  # Replace with the desired expiration date
    find_low_risk_put_options(ticker, expiration_date)
