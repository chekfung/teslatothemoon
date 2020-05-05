import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import Preprocess

# Playing around with seaborn
import seaborn as sns

# sns.set_style("ticks")
# sns.set_style("darkgrid")

# # # Get the data
# # STOCK_DATABASE_PATH = "../data/stock_data.db"
# # RNN_DATABASE_PATH = "../data/rnn_data.db"

# # x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
# # numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

# # # Get data in nice format with correct lagging 

# x, y, z = np.random.rand(3, 100)

# #cmap = sns.cubehelix_palette(as_cmap=True)
# cmap = sns.diverging_palette(10, 133, sep=3, as_cmap=True, center="dark")
# #cmap = sns.light_palette("green", as_cmap=True)

# f, ax = plt.subplots()
# points = ax.scatter(x, y, c=z, s=50, cmap=cmap, marker="8")
# f.colorbar(points, label="meow")
# plt.show()

def get_data(test_prob=0.2):

    # Parsing actual data that we will use
    STOCK_DATABASE_PATH = "../data/stock_data.db"
    RNN_DATABASE_PATH = "../data/rnn_data.db"
    x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
    numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

    X = df_data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

    # Shift to get the previous data as next time step X
    X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].shift(-1)
    y =  df_data["Close"]

    # Remove last step that now has a NaN in shifted values
    X = X[:-1]
    y = y[:-1]

    # Manually make test_size last 20 percent
    dataset_size = len(X)
    train_prob = 1 - test_prob
    split_point = int(np.round(dataset_size * train_prob))

    train_data = X.iloc[:split_point,:]
    test_data = X.iloc[split_point:, :]
    train_prices = y[:split_point]
    test_prices = y[split_point:]

    # Without the twitter data X_train and X_test
    #X_train_no_twit = X_train["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    #X_test_no_twit = X_test["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    return train_data, test_data, train_prices, test_prices

get_data()