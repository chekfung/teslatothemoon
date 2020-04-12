import numpy as np
# import csv
# import random
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import Preprocess

STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"

x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

# Schema for df_data: {Date, Open, High, Low, Close, Adj Close, Volume, 
# Twitter Score}

# =========================================================================== #
# This is an attempt to get rid of outliers and then view what it looks like 


# Compute bounds with z score abs(3) to get rid of outliers
a = df_data['Twitter Score'].to_numpy()
std = np.std(a)
mean = np.mean(a)
print('Mean:', mean, 'Standard Deviation', std)
bottom_bound = (-3 * std) + mean
top_bound = (3 * std) + mean
print('Bottom Bound:', bottom_bound, 'Top Bound:', top_bound)

# Actually get rid of outliers f
new_df = df_data[['Twitter Score', 'Close', 'Open']]
new_df = new_df[new_df['Twitter Score'] < top_bound]
new_df = new_df[new_df['Twitter Score'] > bottom_bound]
print('Found',df_data["Twitter Score"].size - new_df["Twitter Score"].size, 'outliers')

# Set X,y to be twitter Score
X = new_df["Twitter Score"]
X = X.values.reshape(-1,1)
y =  new_df["Close"] - new_df["Open"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Quick plot to show twitter data next to stock data
fig = plt.figure(0)
host = fig.add_subplot(111)

par1 = host.twinx()
host.set_xlabel('Date')
host.set_ylabel('Twitter Sentiment Scores')
par1.set_ylabel('Tesla Share Price (USD)')
host.set_title('Tesla Share Price vs. Twitter Sentiment Analysis Scores')

p1, = host.plot(df_data['Date'], df_data['Twitter Score'], '-r', label='Twitter Sentiment Scores')
p2, = par1.plot(df_data['Date'], df_data['Close'], label="Tesla Share Price (USD)")
host.legend(handles=[p1,p2], loc='best')
fig.autofmt_xdate()
plt.show()

# ========================================================================== #

# Build Regression Models
lin = LinearRegression()
lin.fit(X_train, y_train)

print('Training MSE:', mean_squared_error(y_train, lin.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lin.predict(X_test)))
print('Training R-squared:', r2_score(y_train, lin.predict(X_train)))
print('Testing R-squared:', r2_score(y_test, lin.predict(X_test)))
print('\n')

plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
plt.show()

# ============================================================================ #
# Polynomial Regression with degree 2

poly = PolynomialFeatures(degree = 2) 
X_poly_train = poly.fit_transform(X_train) 
X_poly_test = poly.fit_transform(X_test)
  
poly.fit(X_poly_train, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly_train, y_train) 

print('Polynomial Regression Degree 2 Training MSE:', mean_squared_error(y_train, lin2.predict(X_poly_train)))
print('Polynomial Regression Degree 2 Testing MSE:', mean_squared_error(y_test, lin2.predict(X_poly_test)))
print('Polynomial Regression Degree 2 Training R-squared:', r2_score(y_train, lin2.predict(X_poly_train)))
print('Polynomial Regression Degree 2 Testing R-squared:', r2_score(y_test, lin2.predict(X_poly_test)))
print('\n')

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
plt.show()

# ============================================================================ #
# Polynomial regression with degree 3
poly3 = PolynomialFeatures(degree = 3) 
X_poly3_train = poly3.fit_transform(X_train) 
X_poly3_test = poly3.fit_transform(X_test)
  
poly3.fit(X_poly3_train, y_train) 
lin3 = LinearRegression() 
lin3.fit(X_poly3_train, y_train) 

print('Polynomial Regression Degree 3 Training MSE:', mean_squared_error(y_train, lin3.predict(X_poly3_train)))
print('Polynomial Regression Degree 3 Testing MSE:', mean_squared_error(y_test, lin3.predict(X_poly3_test)))
print('Polynomial Regression Degree 3 Training R-squared:', r2_score(y_train, lin3.predict(X_poly3_train)))
print('Polynomial Regression Degree 3 Testing R-squared:', r2_score(y_test, lin3.predict(X_poly3_test)))
print('\n')

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin3.predict(poly3.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression (3rd degree) of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
plt.show()

# ============================================================================ #
# SVR model to see how it does
clf = SVR()
clf.fit(X_train, y_train)

print('Support Vector Regression Training MSE:', mean_squared_error(y_train, clf.predict(X_train)))
print('Support Vector Regression Testing MSE:', mean_squared_error(y_test, clf.predict(X_test)))
print('Support Vector Regression Training R-squared:', r2_score(y_train, clf.predict(X_train)))
print('Support Vector Regression Testing R-squared:', r2_score(y_test, clf.predict(X_test)))
print('\n')

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, clf.predict(X), color = 'red') 
plt.title('Support Vector Regression of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 

plt.show()



