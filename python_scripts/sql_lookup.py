import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 

# Declare the path to the database
path = os.path.dirname(sys.path[0])
path = os.path.join(path, "data", "stock_data.db")
time_format = '%Y-%m-%d'

# Connect and get the tweets that we want
conn = sqlite3.connect(path)
SQL = '''
SELECT text, (quote_count + reply_count + retweet_count + favorite_count) as meow from TWEETS
WHERE (time > "{}") AND (time < "{}") 
ORDER BY meow DESC
LIMIT 20;
'''.format(datetime.datetime(2020,2,27), datetime.datetime(2020,2,28))
polarities = pd.read_sql(SQL, conn)
conn.close()

print(polarities.to_numpy())