
# auth.openai.com
# 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

# https://platform.openai.com/settings/organization/billing/history
# https://github.com/openai/openai-python

# pip install openai
# Or if issues based on other versions installed and or different python environments
# python -m pip install openai


import os
from openai import OpenAI



# Function to get social sentiment about a stock
def get_stock_sentiment(stock_symbol):

    prompt = f"What is the current social sentiment about the stock {stock_symbol}? Please provide a summary of positive and negative sentiments from social media and news."

    client = OpenAI(
        # This is the default and can be omitted
        api_key="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"
    )
     
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can use gpt-4 if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        #sentiment_summary = response['choices'][0]['message']['content']
        sentiment_summary = response

        return sentiment_summary

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ")
    sentiment = get_stock_sentiment(stock_symbol)

    print(f"Sentiment about {stock_symbol}: {sentiment}")
