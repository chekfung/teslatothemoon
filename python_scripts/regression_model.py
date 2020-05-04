import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from preprocess import Preprocess

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
STOCK_DATABASE_PATH = "../data/stock_data.db"
RNN_DATABASE_PATH = "../data/rnn_data.db"
x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()
sns.set_style("ticks")
sns.set_style("darkgrid")

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
sns.distplot(np.array(df_data['Twitter Score']), kde=False, color="#D50E1D", axlabel="Twitter Sentiment Scores", bins=10).set_title("Distribution of Twitter Sentiment Scores")
plt.savefig("../images/sentiment_histogram.png")
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

# Compute bounds with z score abs(3) to get rid of outliers
a = df_data['Twitter Score'].to_numpy()
std = np.std(a)
mean = np.mean(a)
print('Mean:', mean, 'Standard Deviation', std)
bottom_bound = (-3 * std) + mean
top_bound = (3 * std) + mean
print('Bottom Bound:', bottom_bound, 'Top Bound:', top_bound)

# Actually get rid of outliers f
new_df = df_data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

print("Outliers that we found:")
print(new_df[new_df['Twitter Score'] > top_bound])
print(new_df[new_df['Twitter Score'] < bottom_bound])

# Set X,y to be twitter Score
X = new_df[["Open", "Close", "Adj Close", "Twitter Score"]]

# Shift to get the previous data as next time step X
X[["Open",  "Close", "Adj Close"]] = X[["Open", "Close", "Adj Close"]].shift(-1)
y =  new_df["Close"]

# Remove last step that now has a NaN in shifted values
X = X[:-1]
y = y[:-1]

# Manually make test_size last 20 percent
dataset_size = len(X)
split_point = int(np.round(dataset_size * 0.8))
print("Split Point: ", split_point)
X_train = X.iloc[:split_point,:]
X_test = X.iloc[split_point:, :]
y_train = y[:split_point]
y_test = y[split_point:]

# Label test hours instead of using date time objects
hours = np.arange(len(df_data['Date']))
hours_clipped = hours[:-1]
test_hours = hours_clipped[split_point:]

# ========================================================================== #

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

# plt.scatter(test_hours, y_test, color = 'blue')
# print(y_test)
# plt.plot(test_hours, lin.predict(X_test), color = 'red')
# plt.title('Linear Regression of Stock Price vs. Twitter Sentiment')
# plt.legend(['Predicted Model','Raw data'], loc='best')
# plt.xlabel('Twitter Sentiment')
# plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
# plt.show()

# Seaborn
ax = sns.scatterplot(x=test_hours, y=y_test)
sns.lineplot(x=test_hours, y=lin.predict(X_test),
             ax=ax).set_title("Linear Regression of Stock Price vs. Twitter Sentiment")
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment')
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
plt.show()

# ============================================================================ #
#Testing to see whether or not the statsmodel produces anything different
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

# plt.scatter(test_hours, y_test, color = 'blue')
# plt.plot(test_hours, lin2.predict(poly.fit_transform(X_test)), color = 'red')
# plt.title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment')
# plt.legend(['Predicted Model','Raw data'], loc='best')
# plt.xlabel('Twitter Sentiment')
# plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
# plt.show()

# Seaborn
ax = sns.scatterplot(x=test_hours, y=y_test)
sns.lineplot(x=test_hours, y=lin2.predict(poly.fit_transform(X_test)),
             ax=ax).set_title('Polynomial Regression (2nd degree) of Stock Price vs. Twitter Sentiment')
plt.legend(['Predicted Model','Raw data'], loc='best')
plt.xlabel('Twitter Sentiment')
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
plt.show()

# =========================================================================== #
# Combined linear and polynomial plot
cmap = sns.dark_palette("muted purple", input="xkcd", as_cmap=True)
sentiment = X_test["Twitter Score"].astype(int).to_numpy()

f, ax = plt.subplots()
points = ax.scatter(test_hours, y_test, c=sentiment, s=50, cmap=cmap, marker="8")
f.colorbar(points, label="Twitter Sentiment Score")
#ax = sns.scatterplot(x=test_hours, y=y_test.to_numpy(), c=sentiment, cmap=cmap)
sns.lineplot(x=test_hours, y=lin.predict(X_test),
             ax=ax)
sns.lineplot(x=test_hours, y=lin2.predict(poly.fit_transform(X_test)),
             ax=ax)
plt.legend(['Polynomial Degree 2, r2=0.74', "Linear, r2=0.66", "Raw Data"], loc='best')
plt.title("Multiple Regression of Tesla Stock Price Vs. Twitter Sentiment Score")
plt.xlabel('Twitter Sentiment')
plt.ylabel('TSLA Share Close Price - TSLA Share Open Price (USD)')
plt.show()

# =========================================================================== #
# Seaborn Heatmap
p_values = results.pvalues
data = np.asarray(p_values.to_numpy()).reshape(5,1)

sns.heatmap(data, vmax=0.6, annot=True)
plt.show()
# =========================================================================== #


