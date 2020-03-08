import tweepy
import yfinance as yf
import pandas as pd
from pandas import Series
import sqlite3

# Get Tesla stock data as a DataFrame from start date to end date
# DataFrame index is the date, Columns include Open, High, Low, Close, Adj Close
# and Volume
data = yf.download("TSLA", start="2020-02-01", end="2020-02-29", interval="15m")

# Create connection to database
conn = sqlite3.connect('stock_data.db')
c = conn.cursor()

# Dictionary that defines column names and their types
data_types = {
    "date":"DATE", 
    "open":"FLOAT",
    "high":"FLOAT",
    "low":"FLOAT",
    "close":"FLOAT",
    "adjClose":"FLOAT", 
    "volume":"INT"
}

# Converts the DataFrame to a sql database, replacing it if it already exists
data.to_sql('StockData', conn, if_exists='replace', index=True, \
    index_label="date", dtype=data_types)

# Writes the DataFrame to a csv file for the data sample deliverable
data[0:50].to_csv(path_or_buf='one_month_stock_data.csv')

rnn_data = yf.download("TSLA", start="2019-10-01", end="2020-01-31", interval="1h")

# Create connection to database
conn = sqlite3.connect('rnn_data.db')
c = conn.cursor()

# Writes the RNN training data DataFrame to a sqlite3 database
rnn_data.to_sql('RNNData', conn, if_exists='replace', index=True, index_label="date", dtype=data_types)

# Writes the RNN training data DataFrame to a csv file for the sample deliverable
rnn_data[0:50].to_csv(path_or_buf='rnn_stock_data.csv')