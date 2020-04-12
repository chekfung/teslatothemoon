import numpy as np
# import csv
# import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import sqlite3
import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from preprocess import Preprocess

STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"

x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()


print(df_data)

#conn = sqlite3.connect(DATABASE_PATH)
#
## schema: {id, time, text, quote_count, reply_count, retweet_count, 
## favorite_count, polarity, subjectivity}
#tweet_df = pd.read_sql_query("SELECT * FROM Tweets;", conn)
#
## schema: {date, Open, High, Low, Close, Adj Close, Volume}
#stock_df = pd.read_sql_query("SELECT * FROM StockData;", conn)
#
#conn.close()
#
## Add data about the tweet period to tweet_df
#tweet_df["tweet_period"] = [x[:13] for x in tweet_df["time"]]
#
## Add data about the stock period to stock_df
#stock_df["stock_period"] = [x[:13] for x in stock_df["date"]]

# stock_dict is a dictionary with keys as stock period and values as averaged
# stock close value
#stock_dict = {}
#
#for x in np.unique(stock_df["stock_period"]):
#    sum = 0
#    count = 0
#    for ind in stock_df.index:
#        if stock_df["stock_period"][ind]==x:
#            count = count + 1
#            sum = sum + stock_df["Close"][ind]
#    stock_dict[x] = sum/count
#            
#
#tweet_df["stock_price"] = [stock_dict.get(x) for x in tweet_df["tweet_period"]]
#
##for ind in tweet_df.index:
##    if not(math.isnan(tweet_df["stock_price"][ind])):
##        print(tweet_df["stock_price"][ind])
## period_dict is a dictionary listing how many tweets occured in a given time
## period
#period_dict = {}
#
#period_list = list(tweet_df["tweet_period"])
#
#for x in np.unique(tweet_df["tweet_period"]):
#    period_dict[x] = period_list.count(x)
#
#
## Make a new column in tweet_df that calculates our sentiment score
## Sentiment (polarity) (score) * [likes + retweets + quotes + comments] 
## / number of tweets in that time period
#tweet_df["sentiment_score"] = [(tweet_df["polarity"][ind]*(tweet_df["favorite_count"][ind] 
#+ tweet_df["retweet_count"][ind] + tweet_df["reply_count"][ind] + tweet_df["quote_count"][ind])) 
#    / period_dict[tweet_df["tweet_period"][ind]] for ind in tweet_df.index]
#
#final_df = tweet_df[["sentiment_score", "stock_price"]]
#
#final_df = final_df.dropna()
#final_df = final_df.loc[final_df["sentiment_score"]>0.1]
#X = final_df["sentiment_score"] 
#X = X.values.reshape(-1, 1)
#y = final_df["stock_price"]
#
## Build Regression Models
#lin = LinearRegression()
#lin.fit(X, y)
#
#plt.scatter(X, y, color = 'blue') 
#  
#plt.plot(X, lin.predict(X), color = 'red') 
#plt.title('Linear Regression of Stock Price vs. Twitter Sentiment') 
#plt.xlabel('Twitter Sentiment') 
#plt.ylabel('Stock Price') 
#  
#plt.show()
#
#poly = PolynomialFeatures(degree = 2) 
#X_poly = poly.fit_transform(X) 
#  
#poly.fit(X_poly, y) 
#lin2 = LinearRegression() 
#lin2.fit(X_poly, y) 
#
#plt.scatter(X, y, color = 'blue') 
#  
#plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
#plt.title('Polynomial Regression') 
#plt.xlabel('Twitter Sentiment') 
#plt.ylabel('Stock Price') 
#  
#plt.show()

