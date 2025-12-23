
# Get your API key
# https://www.reddit.com/prefs/apps
# client_id: kuddmvewYxuYwomjUUoCyg
# secret: 8RwXicv20gfcndBhKJPCNf69DXa0mA

# pip install praw textblob


import praw
from textblob import TextBlob
import re

# Configure Reddit API
reddit = praw.Reddit(
    client_id='kuddmvewYxuYwomjUUoCyg',
    client_secret='8RwXicv20gfcndBhKJPCNf69DXa0mA',
    user_agent='StockSentimentApp by u/Nervous_Present_9497'
)

# Define your search and keywords
search_keyword = "Trump"
subreddits = ["politics", "ask_politics", "neutralpolitics", "politicaldebate", "politicaldiscussion"]
keywords = [search_keyword, "Trump"]

# Sentiment Analysis function
def get_sentiment(text):
    analysis = TextBlob(text)

    return analysis.sentiment.polarity

# Get posts and analyze sentiment
def analyze_sentiment():
    total_sentiment = 0
    count = 0
    
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Fetch recent posts from each subreddit
        for submission in subreddit.new(limit=1000):
            # Check if any of the keywords are in the post title or body
            if any(keyword.lower() in submission.title.lower() or keyword.lower() in submission.selftext.lower() for keyword in keywords):
                # Combine title and selftext for analysis
                content = submission.title + " " + submission.selftext
                content = re.sub(r'\W+', ' ', content)  # Clean up text

                # Calculate sentiment
                sentiment = get_sentiment(content)
                total_sentiment += sentiment
                count += 1
    
    if count == 0:
        return "No relevant posts found."
    else:
        avg_sentiment = total_sentiment / count
        sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
        return f"Average Sentiment: {avg_sentiment:.2f} ({sentiment_label})"

# Run the analysis
print(f"Sentiment for {search_keyword}:")
print(analyze_sentiment())
