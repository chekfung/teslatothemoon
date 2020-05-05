import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_style("darkgrid")

# Declare the path to the database
path = os.path.dirname(sys.path[0])
path = os.path.join(path, "data", "stock_data.db")

# Connect and get the tweets that we want
conn = sqlite3.connect(path)
polarities = pd.read_sql("SELECT polarity FROM Tweets", conn)
conn.close()

# Seaborn Histogram
# Histogram of Sentiments
sns.distplot(np.array(polarities), kde=False, color="#D50E1D", axlabel="Twitter Sentiment Scores", bins=10).set_title("Distribution of Twitter Sentiment Scores")
plt.ylabel('Number of Tweets')
plt.savefig("../images/raw_sentiment_histogram.png")
plt.show()



