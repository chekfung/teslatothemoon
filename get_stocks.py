import tweepy
import yfinance as yf
import pandas as pd
from pandas import Series
import sqlite3

# Get Tesla stock data as a DataFrame from start date to end date
# DataFrame index is the date, Columns include Open, High, Low, Close, Adj Close
# and Volume
data = yf.download("TSLA", start="2020-02-01", end="2020-02-29")
data.index = Series.dt.strftime(data.index, '%Y-%m-%d')

# Create connection to database
conn = sqlite3.connect('stock_data.db')
c = conn.cursor()

# Dictionary that defines column names and their types
data_types = {
    "date":"TEXT", 
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