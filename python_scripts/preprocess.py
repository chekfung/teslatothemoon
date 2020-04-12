import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from dateutil import parser
from pandas import Series, DataFrame

"""
Returns price and twitter data as a numpy and dataframe, as well as prices for
the vanilla rnn both as a numpy array and series. 

Each row of stock and twitter data is a price and twitter score for tweets up
until the next price data point. Each row of vanilla rnn data is just the price
"""

class Preprocess(): 

    def __init__(self, stock_path='data/stock_data.db', rnn_path='data/rnn_data.db'): 
        self.stock_path = stock_path
        self.rnn_path = rnn_path

    # Returns a tuple of mean scores used to predict the given date/hour and scores starting at inputted date/hour
    def get_score(self, scores, date, hour): 
        idx = 0
        dates = scores.index
        curr = dates[0]
        while (curr[0].date() < date) or (curr[0].date() == date and curr[1] < hour): 
            idx += 1
            curr = dates[idx]
        return scores[:idx].mean(), scores[idx:]

    def get_data(self): 
        #Score for Hour = Sum(polarity * [likes + retweets + favorites + comments] / tweets)
        conn = sqlite3.connect(self.stock_path)

        #id, time, text, quote_count, reply_count, retweet_count, favorite_count, polarity, subjectivity
        tweets = pd.read_sql("SELECT * FROM Tweets", conn)
        #date, Open, High, Low, Close, Adj Close, Volume
        month_stock_data = pd.read_sql("SELECT * FROM StockData", conn)
        conn.close()

        conn = sqlite3.connect(self.rnn_path)
        #date, Open, High, Low, Close, Adj Close, Volume
        rnn_data = pd.read_sql("SELECT * FROM RNNData", conn)
        conn.close()

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
        is_fourty_five_mins = month_stock_data["date"].apply(lambda x: x.minute) == 45
        hourly_stock_data = month_stock_data[is_fourty_five_mins]

        # Convert to dataframe with 2d index of date and hour
        data = {}
        for i in range(len(hourly_stock_data)): 
            row = hourly_stock_data.iloc[i]
            data[(row["date"].date(), row["date"].hour)] = row
        data = DataFrame(data).transpose()
        # Only get the last price for 2/6
        data = data[6:]

        #------------------------------------------------------------------------------
        numpy_data = []
        df_data = {}
        val_copy = vals.copy()
        for i in range(len(data) - 1): 
            curr = list(data.iloc[i])
            next_date = data.index[i+1][0].date()
            next_hour = data.index[i+1][1]
            score, val_copy = self.get_score(val_copy, next_date, next_hour)
            curr.append(score)
            numpy_data.append(curr)
            df_data[data.index[i]] = curr
        numpy_data = np.array(numpy_data)
        df_data = DataFrame(df_data).transpose()
        df_data.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]

        # Get Vanilla rnn data
        numpy_vanilla_rnn_data = np.array(rnn_data)
        df_vanilla_rnn_data = DataFrame(rnn_data)
        df_vanilla_rnn_data.index = rnn_data["date"]

        return numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data