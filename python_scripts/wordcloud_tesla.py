import pandas as pd
import numpy as np
import sqlite3

import nltk


import numpy as np

import os
import re
from PIL import Image
from os import path
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter

try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

set(stopwords.words('english'))

conn = sqlite3.connect("C:\\Users\\nealm\\OneDrive\\Documents\\GitHub\\teslatothemoon\\data\\stock_data.db")

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
	extraneous_words = ["https", "co", "rt"]
	filtered_tokens = [w for w in tokens if not w in stop_words]
	filtered_tokens_two = [w for w in filtered_tokens if not w in extraneous_words]


	# TODO: Return list of processed words
	return filtered_tokens_two



processed_tweets = process_document(tweet_str)

word_dict = Counter()

for elm in processed_tweets:
    word_dict[elm] += 1


def makeImage(text):
    tesla_mask = np.array(Image.open("C:\\Users\\nealm\\OneDrive\\Documents\\GitHub\\teslatothemoon\\images\\tesla-motors-logo.jpg"))

    wc = WordCloud(background_color="white", max_words=250, mask=tesla_mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    image_colors = ImageColorGenerator(tesla_mask)
    plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()



makeImage(word_dict)

