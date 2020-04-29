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


# Trying stats tools to see if they give other information
import statsmodels.api as sm 
from statsmodels.tools import eval_measures


STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"

x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()


# Lets try and make a histogram of the twitter sentiment
twitter_sentiment = df_data["Twitter Score"]
twitter_sentiment.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Twitter Sentiment Histogram')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.show()


# ============================================================================ #
# Graphing the raw sentiment vs. the stock price
#avg_stock = (df_data["High"] + df_data["Low"]) / 2
close = df_data["Close"]
sentiment = df_data["Twitter Score"]

plt.scatter(sentiment, close, color = 'green') 
plt.title("Raw Plot of Sentiment Vs. Close Price for TSLA")
plt.xlabel("Sentiment")
plt.ylabel("Close Price (USD)")
plt.show()


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
new_df = df_data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

print(new_df[new_df['Twitter Score'] > top_bound])
print(new_df[new_df['Twitter Score'] < bottom_bound])

# new_df = new_df[new_df['Twitter Score'] < top_bound]
# new_df = new_df[new_df['Twitter Score'] > bottom_bound]
# print('Found',df_data["Twitter Score"].size - new_df["Twitter Score"].size, 'outliers')

# Set X,y to be twitter Score
X = new_df[["Open", "Close", "Adj Close"]]

# Shift to get the previous data as next time step X
X[["Open",  "Close", "Adj Close"]] = X[["Open", "Close", "Adj Close"]].shift(-1)
X = X[:-1] 
X = X.values.reshape(-1,3)
y =  new_df["Close"] #- new_df["Open"]
y = y[:-1] 


# Manually make test_size last 20 percent
dataset_size = len(X)
split_point = int(np.round(dataset_size * 0.8))
X_train = X[:split_point,:]
X_test = X[split_point:, :]
y_train = y[:split_point]
y_test = y[split_point:]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)



hours = np.arange(len(df_data['Date']))

test_hours = hours[:-1]
test_hours = test_hours[split_point:]


#Quick plot to show twitter data next to stock data
fig = plt.figure(0)
host = fig.add_subplot(111)

par1 = host.twinx()
host.set_xlabel('Date')
host.set_ylabel('Twitter Sentiment Scores')
par1.set_ylabel('Tesla Share Price (USD)')
host.set_title('Tesla Share Price vs. Twitter Sentiment Analysis Scores')

p1, = host.plot(hours, df_data['Twitter Score'], '-r', label='Twitter Sentiment Scores')
p2, = par1.plot(hours, df_data['Close'], label="Tesla Share Price (USD)")
host.legend(handles=[p1,p2], loc='best')
fig.autofmt_xdate()
plt.show()




# ========================================================================== #

# Build Regression Models
lin = LinearRegression()
lin.fit(X_train, y_train)
print(lin.coef_)

print('Training MSE:', mean_squared_error(y_train, lin.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lin.predict(X_test)))
print('Training R-squared:', r2_score(y_train, lin.predict(X_train)))
print('Testing R-squared:', r2_score(y_test, lin.predict(X_test)))
print('\n')

plt.scatter(test_hours, y_test, color = 'blue') 

plt.plot(test_hours, lin.predict(X_test), color = 'red') 
plt.title('Linear Regression of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
plt.show()

# ============================================================================ #
#Testing to see whether or not the statsmodel produces anything different 

# x_numpy = numpy_data[:, [1,7]].astype(float)
# y_numpy = numpy_data[:, 4].astype(float)

# nX_train, nX_test, ny_train, ny_test = train_test_split(x_numpy, y_numpy, test_size=0.2)

# Manually make test_size last 20 percent

nX_train = X_train.astype(float)
nX_test = X_test.astype(float)
ny_train = y_train.astype(float)
ny_test = y_test.astype(float)


# Try to see if statsmodels has anything different it can provide us
modified_x = sm.add_constant(nX_train)
model = sm.OLS(ny_train, modified_x)
results = model.fit() 

print(results.summary())

print('R squared value:', results.rsquared)

# Predicted y for both test and train
predicted_y_train = results.predict(modified_x)
modified_x_test = sm.add_constant(nX_test)
predicted_y_test = results.predict(modified_x_test)

# Get training and test MSE
train_mse = eval_measures.mse(ny_train, predicted_y_train)
test_mse = eval_measures.mse(ny_test, predicted_y_test)
print("Train MSE:", str(train_mse))
print("Test MSE:", str(test_mse))


# plt.scatter(X[:,0], y, color = 'blue') 

# plt.plot(X[:,0], lin.predict(X), color = 'red') 
# plt.title('Linear Regression of Stock Price vs. Twitter Sentiment') 
# plt.legend(['Predicted Model','Raw data'], loc='best')
# plt.xlabel('Twitter Sentiment') 
# plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
# plt.show()

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

plt.scatter(test_hours, y_test, color = 'blue') 
  
plt.plot(test_hours, lin2.predict(poly.fit_transform(X_test)), color = 'red') 
plt.title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
plt.show()

# ============================================================================ #
# # Polynomial regression with degree 3
# poly3 = PolynomialFeatures(degree = 3) 
# X_poly3_train = poly3.fit_transform(X_train) 
# X_poly3_test = poly3.fit_transform(X_test)
  
# poly3.fit(X_poly3_train, y_train) 
# lin3 = LinearRegression() 
# lin3.fit(X_poly3_train, y_train) 

# print('Polynomial Regression Degree 3 Training MSE:', mean_squared_error(y_train, lin3.predict(X_poly3_train)))
# print('Polynomial Regression Degree 3 Testing MSE:', mean_squared_error(y_test, lin3.predict(X_poly3_test)))
# print('Polynomial Regression Degree 3 Training R-squared:', r2_score(y_train, lin3.predict(X_poly3_train)))
# print('Polynomial Regression Degree 3 Testing R-squared:', r2_score(y_test, lin3.predict(X_poly3_test)))
# print('\n')

# plt.scatter(X[:,0], y, color = 'blue') 
  
# plt.plot(X[:,0], lin3.predict(poly3.fit_transform(X)), color = 'red') 
# plt.title('Polynomial Regression (3rd degree) of Stock Price vs. Twitter Sentiment') 
# plt.legend(['Predicted Model','Raw data'], loc='best')
# plt.xlabel('Twitter Sentiment') 
# plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 
  
# plt.show()

# ============================================================================ #
# SVR model to see how it does
clf = SVR(kernel='sigmoid', degree=6)
clf.fit(X_train, y_train)

print('Support Vector Regression Training MSE:', mean_squared_error(y_train, clf.predict(X_train)))
print('Support Vector Regression Testing MSE:', mean_squared_error(y_test, clf.predict(X_test)))
print('Support Vector Regression Training R-squared:', r2_score(y_train, clf.predict(X_train)))
print('Support Vector Regression Testing R-squared:', r2_score(y_test, clf.predict(X_test)))
print('\n')

plt.scatter(test_hours, y_test, color = 'blue') 
  
plt.plot(test_hours, clf.predict(X_test), color = 'red') 
plt.title('Support Vector Regression of Stock Price vs. Twitter Sentiment') 
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment') 
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)') 

plt.show()



