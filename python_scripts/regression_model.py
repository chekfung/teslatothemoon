import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import math
import pandas as pd
from preprocess import Preprocess
import os 
import sys
import csv
# Regression Model Imports
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tools import eval_measures

# Preprocess the data
INCLUDE_ALL_VARIABLES = False

STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"
x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()
sns.set_style("whitegrid")


# Lets try and make a histogram of the twitter sentiment
twitter_sentiment = df_data["Twitter Score"]
twitter_sentiment.plot.hist(grid=True, bins=10, rwidth=0.9,
                   color='#607c8e')
plt.title('Twitter Sentiment Histogram')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Seaborn Histogram
# Histogram of Sentiments
sns.distplot(np.array(df_data['Twitter Score']), hist_kws=dict(alpha=1), kde=False, color="#eda70e", axlabel="Twitter Sentiment Scores", bins=10).set_title("Distribution of Twitter Sentiment Scores")
plt.ylabel("Count")
plt.savefig("../images/sentiment_histogram.png", dpi=300)
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

# =========================================================================== #
# This is an attempt to get rid of outliers and then view what it looks like

# Compute bounds with z score abs(3) 
a = df_data['Twitter Score'].to_numpy()
std = np.std(a)
mean = np.mean(a)
print('Mean:', mean, 'Standard Deviation', std)
bottom_bound = (-3 * std) + mean
top_bound = (3 * std) + mean
print('Bottom Bound:', bottom_bound, 'Top Bound:', top_bound)

new_df = df_data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

print("Outliers that we found:")
print(new_df[new_df['Twitter Score'] > top_bound])
print(new_df[new_df['Twitter Score'] < bottom_bound])

# If we get rid of high, low, volume
if not INCLUDE_ALL_VARIABLES:
    # Set X,y to be twitter Score
    X = new_df[["Open", "Close", "Adj Close", "Twitter Score"]]

    # Shift to get the previous data as next time step X
    X[["Open", "Close", "Adj Close"]] = X[["Open", "Close", "Adj Close"]].shift(-1)
    y =  new_df["Close"]

    # Remove last step that now has a NaN in shifted values
    X = X[:-1]
    y = y[:-1]

    # Manually make test_size last 20 percent
    dataset_size = len(X)
    split_point = int(np.round(dataset_size * 0.8))

    X_train_p = X.iloc[:split_point,:]
    X_test_p = X.iloc[split_point:, :]
    y_train = y[:split_point]
    y_test = y[split_point:]

    X_train = X_train_p[["Open", "Close", "Adj Close", "Twitter Score"]]
    X_test = X_test_p[["Open", "Close", "Adj Close", "Twitter Score"]]

    # Without the twitter data X_train and X_test
    X_train_no_twit = X_train[["Open", "Close", "Adj Close"]]
    X_test_no_twit = X_test[["Open", "Close", "Adj Close"]]

# If want to do heatmap, etc.
else:
    # Set X,y to be twitter Score
    X = new_df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

    # Shift to get the previous data as next time step X
    X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].shift(-1)
    y =  new_df["Close"]

    # Remove last step that now has a NaN in shifted values
    X = X[:-1]
    y = y[:-1]

    # Manually make test_size last 20 percent
    dataset_size = len(X)
    split_point = int(np.round(dataset_size * 0.8))

    X_train_p = X.iloc[:split_point,:]
    X_test_p = X.iloc[split_point:, :]
    y_train = y[:split_point]
    y_test = y[split_point:]

    X_train = X_train_p[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]
    X_test = X_test_p[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

    # Without the twitter data X_train and X_test
    X_train_no_twit = X_train[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    X_test_no_twit = X_test[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# Label test hours instead of using date time objects
hours = np.arange(len(df_data['Date']))
hours_clipped = hours[:-1]
test_hours = hours_clipped[split_point:]

# ========================================================================== #

#Quick plot to show twitter data next to stock data
with sns.axes_style("white"):
    fig = plt.figure(0)
    host = fig.add_subplot(111)
    
    par1 = host.twinx()
    host.set_xlabel('Date')
    host.set_ylabel('Twitter Sentiment Scores')
    par1.set_ylabel('Tesla Share Price (USD)')
    host.set_title('Tesla Share Price vs. Twitter Sentiment Analysis Scores')

    p1, = host.plot(hours, df_data['Twitter Score'], '-r', color="#cc0000", label='Twitter Sentiment Scores')
    p2, = par1.plot(hours, df_data['Close'], label="Tesla Share Price (USD)", color="#641499")
    host.legend(handles=[p1,p2], loc='best')
    fig.autofmt_xdate()
    # plt.savefig('../images/price_vs_sentiment.png', dpi=300)
    plt.show()

# ========================================================================== #

# Build Regression Models
lin = LinearRegression()
lin.fit(X_train, y_train)
print(lin.coef_)

print('Linear Training MSE:', mean_squared_error(y_train, lin.predict(X_train)))
print('Linear Testing MSE:', mean_squared_error(y_test, lin.predict(X_test)))
print('Linear Training R-squared:', r2_score(y_train, lin.predict(X_train)))
print('Linear Testing R-squared:', r2_score(y_test, lin.predict(X_test)))
print('\n')

# Without Twitter Data Linear Regression
lin_no_twitter = LinearRegression()
lin_no_twitter.fit(X_train_no_twit, y_train)

print('Linear No Twitter Training MSE:', mean_squared_error(y_train, lin_no_twitter.predict(X_train_no_twit)))
print('Linear No Twitter Testing MSE:', mean_squared_error(y_test, lin_no_twitter.predict(X_test_no_twit)))
print('Linear No Twitter Training R-squared:', r2_score(y_train, lin_no_twitter.predict(X_train_no_twit)))
print('Linear No Twitter Testing R-squared:', r2_score(y_test, lin_no_twitter.predict(X_test_no_twit)))
print('\n')

# Seaborn
ax = sns.scatterplot(x=test_hours, y=y_test, color="#558cf2")#f59505
sns.lineplot(x=test_hours, y=lin.predict(X_test), color="#cc0000",
             ax=ax).set_title("Linear Regression of Stock Price vs. Twitter Sentiment")
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Days Since Start')
plt.ylabel('TSLA Share Close Price - Open Price (USD)')
plt.savefig('../images/linear_regression.png', dpi=300)
plt.show()

# # Save Linear Regression Values in CSV
# filename = "linear_regression.csv"
# path = os.path.join(os.path.dirname(sys.path[0]), "csv", filename)
# predicted = lin.predict(X_test)
# predicted_no_twitter = lin_no_twitter.predict(X_test_no_twit)
# truth = y_test

# with open(path, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Test Hour", "Predicted Price", "No Twitter Predicted Price", "Actual Price"])

#     for i in range(test_hours.shape[0]):
#         writer.writerow([test_hours[i], predicted[i], predicted_no_twitter[i], truth.iloc[i]])


# ============================================================================ #
#Testing to see whether or not the statsmodel produces anything different
# Manually make test_size last 20 percent

nX_train = X_train_p.astype(float)
nX_test = X_test_p.astype(float)
ny_train = y_train.astype(float)
ny_test = y_test.astype(float)

# Try to see if statsmodels has anything different it can provide us
modified_x = sm.add_constant(nX_train)
model = sm.OLS(ny_train, modified_x)
results = model.fit()
print(results.summary())

# INFLUENCE PLOT
#fig, ax = plt.subplots(figsize=(12,8))
#fig = sm.graphics.influence_plot(results, ax=ax, criterion="cooks")
#plt.show()

# Partial Regression Plot
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(results, fig=fig)
plt.show()

#print('R squared value:', results.rsquared)

# Predicted y for both test and train
predicted_y_train = results.predict(modified_x)
modified_x_test = sm.add_constant(nX_test)
predicted_y_test = results.predict(modified_x_test)

# Get training and test MSE
train_mse = eval_measures.mse(ny_train, predicted_y_train)
test_mse = eval_measures.mse(ny_test, predicted_y_test)
#print("Train MSE:", str(train_mse))
#print("Test MSE:", str(test_mse))

# ============================================================================ #
# Polynomial Regression with degree 2

poly = PolynomialFeatures(degree = 3)
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

# Without Twitter Data Linear Regression
X_no_twitter_train = poly.fit_transform(X_train_no_twit)
X_no_twitter_test = poly.fit_transform(X_test_no_twit)
poly2_no_twitter = LinearRegression()
poly2_no_twitter.fit(X_no_twitter_train, y_train)

print('Polynomial No Twitter Training MSE:', mean_squared_error(y_train, poly2_no_twitter.predict(X_no_twitter_train)))
print('Polynomial No Twitter Testing MSE:', mean_squared_error(y_test, poly2_no_twitter.predict(X_no_twitter_test)))
print('Polynomial No Twitter Training R-squared:', r2_score(y_train, poly2_no_twitter.predict(X_no_twitter_train)))
print('Polynomial No Twitter Testing R-squared:', r2_score(y_test, poly2_no_twitter.predict(X_no_twitter_test)))
print('\n')

# plt.scatter(test_hours, y_test, color = 'blue')
# plt.plot(test_hours, lin2.predict(poly.fit_transform(X_test)), color = 'red')
# plt.title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment')
# plt.legend(['Predicted Model','Raw data'], loc='best')
# plt.xlabel('Twitter Sentiment')
# plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
# plt.show()

# Seaborn
ax = sns.scatterplot(x=test_hours, y=y_test, color="#558cf2")
sns.lineplot(x=test_hours, y=lin2.predict(poly.fit_transform(X_test)),
             ax=ax, color="#cc0000").set_title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment')
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Days Since Start')
plt.ylabel('TSLA Share Close Price - Open Price (USD)')
plt.savefig('../images/polynomial_regression.png', dpi=300)
plt.show()

# # Save Polynomial Regression Values in CSV
# filename = "polynomial_regression.csv"
# path = os.path.join(os.path.dirname(sys.path[0]), "csv", filename)
# predicted = lin2.predict(X_poly_test)
# x_test_poly_no_twit = poly.fit_transform(X_test_no_twit)
# predict_no_twit = poly2_no_twitter.predict(x_test_poly_no_twit)
# truth = y_test

# with open(path, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Test Hour", "Predicted Price", "No Twitter Predicted Price", "Actual Price"])

#     for i in range(test_hours.shape[0]):
#         writer.writerow([test_hours[i], predicted[i], predict_no_twit[i], truth.iloc[i]])

# =========================================================================== #
# Combined linear and polynomial plot
cmap = sns.dark_palette("muted purple", input="xkcd", as_cmap=True)
sentiment = X_test["Twitter Score"].astype(int).to_numpy()

f, ax = plt.subplots()
points = ax.scatter(test_hours, y_test, c=sentiment, s=30, cmap=cmap, marker="8")
f.colorbar(points, label="Twitter Sentiment Score")
#ax = sns.scatterplot(x=test_hours, y=y_test.to_numpy(), c=sentiment, cmap=cmap)
sns.lineplot(x=test_hours, y=lin.predict(X_test),
             ax=ax)
sns.lineplot(x=test_hours, y=lin2.predict(poly.fit_transform(X_test)),
             ax=ax)
plt.legend(['Polynomial Degree 2, r2=0.74', "Linear, r2=0.66", "Raw Data"], loc='best')
plt.title("Multiple Regression of Tesla Stock Price Vs. Twitter Sentiment Score")
plt.xlabel('Days Since Start')
plt.ylabel('TSLA Share Close Price - Open Price (USD)')
plt.savefig('../images/both_regression.png', dpi=300)
plt.show()

# =========================================================================== #
# Seaborn Heatmap
p_values = results.pvalues
data = np.asarray(p_values.to_numpy()[1:8]).reshape(7,1)
color_map = cm.get_cmap('Reds', 256)
sns.heatmap(data, vmax=0.6, annot=True, yticklabels=p_values.index.to_numpy()[1:8], 
            cmap=color_map, xticklabels=["p-value"]).set_title("P-Values in Linear Multiple Regression Model with All Variables")
plt.ylabel("Independent Variable")
# plt.savefig('../images/heatmap.png', dpi=300)
plt.show()

# =========================================================================== #


