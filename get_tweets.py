from TwitterAPI import TwitterAPI
import datetime
import time
import sqlite3
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_KEY = ''
ACCESS_SECRET = ''

api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)

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
  PRIMARY KEY (id)
)
'''

c.execute(create_table_command)

conn.commit()

time_format = '%Y-%m-%d %H:%M:%S'

number_of_batches = 250

start_time = datetime.datetime(2020, 2, 7, 0, 0, 0) + datetime.timedelta(hours=216)
end_time = datetime.datetime(2020, 2, 7, 1, 0, 0) + datetime.timedelta(hours=216)

for i in range(number_of_batches):
    time.sleep(3)
    start_search = start_time + datetime.timedelta(hours=i)
    end_search = end_time + datetime.timedelta(hours=i) 
    fromDate = int('{0:4d}{1:2d}{2:2d}{3:2d}{4:2d}'.format(start_search.year, start_search.month, start_search.day, start_search.hour, start_search.minute).replace(' ', '0'))
    toDate = int('{0:4d}{1:2d}{2:2d}{3:2d}{4:2d}'.format(end_search.year, end_search.month, end_search.day, end_search.hour, end_search.minute).replace(' ', '0'))
    results = api.request('tweets/search/30day/:dev', {'query':"(tesla OR TSLA) lang:en -coil -nikola", 'maxResults':100, 'fromDate':fromDate, 'toDate':toDate})
    for result in results.get_iterator():
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(result['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
        c.execute('''INSERT INTO Tweets VALUES (?, ?, ?, ?, ?, ?, ?);''', (result['id'], ts, result['text'], result['quote_count'], result['reply_count'], result['retweet_count'], result['favorite_count']))
    conn.commit()


    