# we messed up so we're adding sentiment manually to a tweet subset
from textblob import TextBlob
import sqlite3
import pandas as pd

conn = sqlite3.connect('stock_data.db')

data = pd.read_sql("SELECT id, time, text, quote_count, reply_count, retweet_count, favorite_count FROM TWEETS", conn)
pol_col = []
sub_col = []

for tweet_text in data["text"]: 
    blob = TextBlob(tweet_text)
    polarity, subjectivity = blob.sentiment
    pol_col.append(polarity)
    sub_col.append(subjectivity)

data["polarity"] = pol_col
data["subjectivity"] = sub_col

data.to_sql('Tweets', conn, if_exists='replace', index=False)

conn.close()