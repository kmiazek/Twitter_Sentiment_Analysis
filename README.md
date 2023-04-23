# Twitter_Sentiment_Analysis

<b>IMPORTANT: This code has ben written before Twitter ownership changes. I am uploading this for reference and to archive it. Due to fast changes on the platform, this code might not work as initially intended. </b>

This script performs sentiment analysis on tweets containing a specific hashtag. It searches for the hashtag, retrieves the tweets, cleans the tweet text, performs sentiment analysis using a pre-trained RoBERTa model, and outputs the results in a DataFrame.

# Requirements
Python 3.6 or higher,
pandas,
tweepy,
transformers,
numpy,
configparser.

# Configuration
To use this script, you'll need to create a config.ini file with your Twitter API credentials. The file should have the following structure:

api_key = YOUR_API_KEY

api_key_secret = YOUR_API_KEY_SECRET

access_token = YOUR_ACCESS_TOKEN

access_token_secret = YOUR_ACCESS_TOKEN_SECRET


Replace YOUR_API_KEY, YOUR_API_KEY_SECRET, YOUR_ACCESS_TOKEN, and YOUR_ACCESS_TOKEN_SECRET with your own Twitter API credentials.

# Usage
1. Customize the keywords variable with the hashtag you want to analyze.
2. Modify the limit variable to set the maximum number of tweets to retrieve.
3. Run the script using python your_script_name.py.

The script will output a DataFrame containing the following columns:

* Time:: The timestamp of the tweet.

* User: The username of the tweet author.

* Retweets: The number of retweets the tweet has received.

* Tweet: The original tweet text.

* CleanedTweet: The cleaned version of the tweet, with usernames replaced by '@user' and URLs replaced by 'http'.

* Sentiment: A dictionary containing sentiment scores for negative, neutral, and positive sentiments.

# Functions
clean_tweet(tweet): Cleans the input tweet text by replacing usernames with '@user' and URLs with 'http'.

analyze_sentiment(tweet): Performs sentiment analysis on the input tweet using a pre-trained RoBERTa model and returns a dictionary of sentiment scores.
