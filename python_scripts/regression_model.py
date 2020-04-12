import numpy as np
# import csv
# import random
import matplotlib.pyplot as plt
import math
import pandas as pd
from statsmodels.tools import eval_measures
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from preprocess import Preprocess

STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"

x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

# Schema for df_data: {Date, Open, High, Low, Close, Adj Close, Volume, 
# Twitter Score}

X = df_data["Twitter Score"]
X = X.values.reshape(-1, 1)
y = df_data["Close"]

# Build Regression Models
lin = LinearRegression()
lin.fit(X, y)

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression of Stock Price vs. Twitter Sentiment') 
plt.xlabel('Twitter Sentiment') 
plt.ylabel('Stock Price') 
  
plt.show()

poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Twitter Sentiment') 
plt.ylabel('Stock Price') 
  
plt.show()

