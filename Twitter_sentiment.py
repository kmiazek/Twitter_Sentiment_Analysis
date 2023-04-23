# Import necessary libraries
import configparser
import tweepy
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# Read configurations & access
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Load pre-trained RoBERTa model and tokenizer for sentiment analysis
roberta = 'cardiffnlp/twitter-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Twitter API authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Search tweets with a specific hashtag
keywords = '#python'
limit = 300

tweets = tweepy.Cursor(api.search_tweets, q=keywords, start_time="2022-10-01",
                       count=300, tweet_mode='extended').items(limit)

# Define columns and labels for the DataFrame
columns = ['Time', 'User', 'Retweets', 'Tweet']
labels = ['Negative', 'Neutral', 'Positive']

# Collect tweet data
data = []
for tweet in tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.retweet_count, tweet.full_text])

# Create DataFrame and remove duplicate tweets
df = pd.DataFrame(data, columns=columns)
df.drop_duplicates(subset=['Tweet'], keep='first')

# Function to clean up tweet text
def clean_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    cleaned_tweet = ' '.join(tweet_words)
    return cleaned_tweet

# Function to perform sentiment analysis on a tweet
def analyze_sentiment(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    processed_tweet = ' '.join(tweet_words)
    encoded_tweet = tokenizer(processed_tweet, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment_scores = dict(zip(labels, scores))
    return sentiment_scores

# Apply cleaning and sentiment analysis functions to the DataFrame
df['CleanedTweet'] = df.apply(lambda row: clean_tweet(row['Tweet']), axis=1)
df['Sentiment'] = df.apply(lambda row: analyze_sentiment(row['Tweet']), axis=1)

# Print the DataFrame
print(df)

# (Optional) export DataFrame to csv
#df.to_csv('tweets.csv', encoding='utf-8')