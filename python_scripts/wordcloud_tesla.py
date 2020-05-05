import pandas as pd
import numpy as np
import sqlite3
import numpy as np
import sys
import os
import re
from PIL import Image
from os import path

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob

try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

set(stopwords.words('english'))

path = os.path.dirname(sys.path[0])
path = os.path.join(path, "data", "stock_data.db")
conn = sqlite3.connect(path)
tweets = pd.read_sql("SELECT * FROM Tweets", conn)
conn.close()

tweet_str = ""

for tweet in tweets["text"]:
    tweet_str += tweet
    tweet_str += " "



def process_document(text):
	"""
	Processes a text document by converting all words to lower case,
	tokenizing, removing all non-alphabetical characters,
	and stemming each word.

	Args:
		text: A string of the text of a single document.

	Returns:
		A list of processed words from the document.
	"""
	# Convert words to lower case
	text = text.lower()

	# TODO: Tokenize document and remove all non-alphanumeric characters
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	# TODO: Remove stopwords
	stop_words = stopwords.words('english')
	extraneous_words = ["https", "co", "rt", "us", "get", "via", "let", "hey", "retweet", "also", "look", "like", "c", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "000", "x"]
	filtered_tokens = [w for w in tokens if not w in stop_words]
	filtered_tokens_two = [w for w in filtered_tokens if not w in extraneous_words]


	# TODO: Return list of processed words
	return filtered_tokens_two



processed_tweets = process_document(tweet_str)
print("Tweets Processed")

# Create dictionary that keeps track of counts of words in tweets
word_dict = Counter()

for elm in processed_tweets:
    word_dict[elm] += 1
print("Counts Dictionary Finished")

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
	blob = TextBlob(word)
	polarity = blob.sentiment.polarity
	bad_words = ["short", "crash", "28delayslater","sell", "never"]
	good_words = ["long", "buy"]
	# Negative sentiment
	if polarity < 0 or (word in bad_words):
		# Red
		value = int(np.round(np.abs(polarity) * 255))
		if word in bad_words:
			value = 255
		color = "rgb({},0,0)".format(value)
	elif (polarity > 0 or (word in good_words)):
		# Green
		value = int(np.round(np.abs(polarity) * 255))
		if word in bad_words:
			value = 255
		color = "rgb(0,{},0)".format(value)
	else:
		color = "rgb(0,0,0)"

	return color


def makeImage(text):
	path = os.path.join(os.path.dirname(sys.path[0]), "images", "tesla-motors-logo.jpg")
	tesla_mask = np.array(Image.open(path))

	wc = WordCloud(background_color="white", max_words=200, mask=tesla_mask)
	# generate word cloud
	wc.generate_from_frequencies(text)

	# show
	#image_colors = ImageColorGenerator(tesla_mask)
	plt.imshow(wc.recolor(color_func=color_func), interpolation="bilinear")
	plt.axis("off")
	#plt.savefig(os.path.join(os.path.dirname(sys.path[0]), "images", "word_cloud"))
	plt.show()

makeImage(word_dict)

