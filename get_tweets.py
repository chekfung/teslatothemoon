from TwitterAPI import TwitterAPI
from textblob import TextBlob
import datetime
import time
import sqlite3
import sys

# Read keys.txt which contains consumer_key and secret, access_key and secret
try:
    # INPUT File name of keys.txt below
    file = open('keys.txt')
    lines = [next(file).rstrip('\n') for x in range(4)]
    print(lines)

    # Input keys and secrets for Twitter API authentication
    CONSUMER_KEY = lines[0]
    CONSUMER_SECRET = lines[1]
    ACCESS_KEY = lines[2]
    ACCESS_SECRET = lines[3]

    file.close()

except FileNotFoundError:
    # If file error occurs
    print('keys.txt not Found.')
    sys.exit(1)

# Connect to the API 
api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)

# Access the database and create Tweets table if it does not exist
db_name = 'stock_data.db'
conn = sqlite3.connect(db_name)
c = conn.cursor()

create_table_command = '''
CREATE TABLE IF NOT EXISTS Tweets (
  id NOT NULL,
  time DATETIME,
  text varchar(280),
  quote_count INT,
  reply_count INT,
  retweet_count INT,
  favorite_count INT,
  polarity FLOAT,
  subjectivity FLOAT,
  PRIMARY KEY (id)
)
'''

c.execute(create_table_command)
conn.commit()

# Establish Time format and offset.
# For each person, we want to do initial offset + 250 hours
time_format = '%Y-%m-%d %H:%M:%S'
NUMBER_OF_BATCHES = 250
HOUR_OFFSET = 716

# Determine start and end times
start_time = datetime.datetime(2020, 2, 7, 0, 0, 0) + datetime.timedelta(hours=HOUR_OFFSET)
end_time = datetime.datetime(2020, 2, 7, 1, 0, 0) + datetime.timedelta(hours=HOUR_OFFSET)

# Loop through and perform number of requests
for i in range(NUMBER_OF_BATCHES):
    # Sleep for 3 seconds so we do not go over request limit
    time.sleep(3)
    print('Iteration: ', i)

    # Put datetime in format twitter API excepts
    start_search = start_time + datetime.timedelta(hours=i)
    end_search = end_time + datetime.timedelta(hours=i) 
    fromDate = int('{0:4d}{1:2d}{2:2d}{3:2d}{4:2d}'.format(start_search.year, start_search.month, start_search.day, start_search.hour, start_search.minute).replace(' ', '0'))
    toDate = int('{0:4d}{1:2d}{2:2d}{3:2d}{4:2d}'.format(end_search.year, end_search.month, end_search.day, end_search.hour, end_search.minute).replace(' ', '0'))

    # Perform Request
    results = api.request('tweets/search/30day/:dev', {'query':"(tesla OR TSLA) lang:en -coil -nikola", 'maxResults':100, 'fromDate':fromDate, 'toDate':toDate})

    # Go through each of the results in the .json 
    for result in results.get_iterator():
        # Put time in good format again, create sentiment analysis and put in db
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(result['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
        blob = TextBlob(result['text'])
        polarity, subjectivity = blob.sentiment
        c.execute('''INSERT INTO Tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);''', (result['id'], ts, result['text'], result['quote_count'], result['reply_count'], result['retweet_count'], result['favorite_count'], polarity, subjectivity))
    conn.commit()


    