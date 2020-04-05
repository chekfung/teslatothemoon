import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from dateutil import parser
from pandas import Series, DataFrame

# Returns a tuple of mean scores used to predict the given date/hour and scores starting at inputted date/hour
def get_score(scores, date, hour): 
    idx = 0
    dates = scores.index
    curr = dates[0]
    while (curr[0].date() < date) or (curr[0].date() == date and curr[1] < hour): 
        idx += 1
        curr = dates[idx]
    return scores[:idx].mean(), scores[idx:]

#Score for Hour = Sum(polarity * [likes + retweets + favorites + comments] / tweets)
conn = sqlite3.connect('data/stock_data.db')

#id, time, text, quote_count, reply_count, retweet_count, favorite_count, polarity, subjectivity
tweets = pd.read_sql("SELECT * FROM Tweets", conn)
#date, Open, High, Low, Close, Adj Close, Volume
month_stock_data = pd.read_sql("SELECT * FROM StockData", conn)

conn = sqlite3.connect('data/rnn_data.db')
#date, Open, High, Low, Close, Adj Close, Volume
rnn_data = pd.read_sql("SELECT * FROM RNNData", conn)

# Convert date/time to timestamp object
times = []
month_dates = []
dates = []
for time in tweets["time"]: 
    times.append(parser.parse(time))
for date in month_stock_data["date"]: 
    month_dates.append(parser.parse(date))
for date in rnn_data["date"]: 
    dates.append(parser.parse(date))
    
tweets["time"] = times
month_stock_data["date"] = month_dates
rnn_data["date"] = dates

# Every 100 tweets corresponds to one hour. Aggregate tweets to per hour
vals = {}
for i in range(0,len(tweets), 100): 
    sample = tweets[i:i+100]
    date = tweets["time"][i].date()
    hour = tweets["time"][i].hour
    idx = (date, hour)
    scores = sample["polarity"] * (sample["favorite_count"] + sample["retweet_count"] + sample["reply_count"] + \
                                  sample["quote_count"])
    score = scores.sum()
    vals[idx] = score  
vals = Series(vals)

# Get pertinent stock rows (per hour)
is_zero_mins = month_stock_data["date"].apply(lambda x: x.minute) == 0
is_three_hour = month_stock_data["date"].apply(lambda x: x.hour) == 15
is_fourty_five_mins = month_stock_data["date"].apply(lambda x: x.minute) == 45
hourly_stock_data = month_stock_data[is_zero_mins | (is_three_hour & is_fourty_five_mins)]

# Filter so hours 10-3 use open while 3:45 is the adj close price
prices = {}
for i in range(len(hourly_stock_data)): 
    row = hourly_stock_data.iloc[i]
    if row["date"].minute == 0: 
        prices[(row["date"].date(), row["date"].hour)] = row["Open"]
    else: 
        prices[(row["date"].date(), 16)] = row["Open"]
prices = Series(prices)
# Only get the last price for 2/6
prices = prices[6:]

#------------------------------------------------------------------------------

numpy_data = []
df_data = {}
val_copy = vals.copy()
for i in range(len(prices) - 1): 
    curr = prices.iloc[i]
    next_date = prices.index[i+1][0].date()
    next_hour = prices.index[i+1][1]
    score, val_copy = get_score(val_copy, next_date, next_hour)
    numpy_data.append([curr, score])
    df_data[prices.index[i]] = [curr, score]
numpy_data = np.array(numpy_data)
df_data = DataFrame(df_data).transpose()
df_data.columns = ["Price", "Score"]

return numpy_data, df_data
