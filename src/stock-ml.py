# write a script using machine learning, social media and news that predicts stock price over the next few days

# pip install scikit-learn pandas numpy tweepy vaderSentiment

# Twitter/X Developer Key
# API Key yL6JRjieTYaCHCFMWKus7uMOJ
# API Key Secret fYIzx4gGdAAbON9SKzp87kMK8J3nytETkRmHcqIlxZfFNJ5af9
# Access Token 61670653-2TicmcqkuabebHZiwiCtHvvfKL2CJRyoUSQQkuhMd
# Access Token Secret SjZqdta2ZEJvYW3C2g8xXoXaYyjeRz4wEWuelXlaYCFXz




import pandas as pd
import numpy as np
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Twitter API credentials
consumer_key = 'yL6JRjieTYaCHCFMWKus7uMOJ'
consumer_secret = 'fYIzx4gGdAAbON9SKzp87kMK8J3nytETkRmHcqIlxZfFNJ5af9'
access_token = '61670653-2TicmcqkuabebHZiwiCtHvvfKL2CJRyoUSQQkuhMd'
access_token_secret = 'SjZqdta2ZEJvYW3C2g8xXoXaYyjeRz4wEWuelXlaYCFXz'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to fetch tweets based on a specific hashtag
def fetch_tweets(hashtag, num_tweets=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=hashtag, count=num_tweets, lang='en').items(num_tweets):
        tweets.append(tweet.text)
    return tweets

# Function to perform sentiment analysis on tweets
def analyze_sentiment(tweets):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for tweet in tweets:
        sentiment = analyzer.polarity_scores(tweet)
        sentiments.append(sentiment['compound'])
    return sentiments

# Fetch tweets and analyze sentiment
hashtag = '#AAPL'  # Example hashtag for Apple Inc.
tweets = fetch_tweets(hashtag)
sentiments = analyze_sentiment(tweets)

# Generate random sample data for demonstration (replace with real data)
stock_prices = np.random.randint(100, 200, len(sentiments))  # Replace with actual stock prices

# Create a DataFrame with sentiment scores and stock prices
data = pd.DataFrame({'Sentiment': sentiments, 'StockPrice': stock_prices})

# Split data into features (sentiment) and target (stock price)
X = data[['Sentiment']]
y = data['StockPrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict stock price for a new sentiment score (for demonstration purposes)
new_sentiment = 0.5  # Replace with actual sentiment score
predicted_stock_price = model.predict([[new_sentiment]])
print(f'Predicted Stock Price: {predicted_stock_price[0]}')


