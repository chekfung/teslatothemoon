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

sns.set_style("ticks")
sns.set_style("darkgrid")

# # Get the data
# STOCK_DATABASE_PATH = "../data/stock_data.db"
# RNN_DATABASE_PATH = "../data/rnn_data.db"

# x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
# numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

# # Get data in nice format with correct lagging 

x, y, z = np.random.rand(3, 100)

#cmap = sns.cubehelix_palette(as_cmap=True)
cmap = sns.diverging_palette(10, 133, sep=3, as_cmap=True, center="dark")
#cmap = sns.light_palette("green", as_cmap=True)

f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=50, cmap=cmap, marker="8")
f.colorbar(points, label="meow")
plt.show()