import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import Preprocess

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
from keras.optimizers import Adam 

def get_data(test_prob=0.2):
    # Parsing actual data that we will use
    STOCK_DATABASE_PATH = "../data/stock_data.db"
    RNN_DATABASE_PATH = "../data/rnn_data.db"
    x = Preprocess(STOCK_DATABASE_PATH, RNN_DATABASE_PATH)
    numpy_data, df_data, numpy_vanilla_rnn_data, df_vanilla_rnn_data = x.get_data()

    X = df_data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Twitter Score"]]

    # Shift to get the previous data as next time step X
    X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = X[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].shift(-1)
    y =  df_data["Close"]

    # Remove last step that now has a NaN in shifted values
    X = X[:-1]
    y = y[:-1]

    # Manually make test_size last 20 percent
    dataset_size = len(X)
    train_prob = 1 - test_prob
    split_point = int(np.round(dataset_size * train_prob))

    train_data = X.iloc[:split_point,:]
    test_data = X.iloc[split_point:, :]
    train_prices = y[:split_point]
    test_prices = y[split_point:]

    # If want to use without the twitter data for BASELINE MODEL
    #X_train_no_twit = X_train["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    #X_test_no_twit = X_test["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Convert out of dataframes for use in numpy
    train_data = train_data.to_numpy().astype(np.float32)
    test_data = test_data.to_numpy().astype(np.float32)
    train_prices = train_prices.to_numpy().astype(np.float32)
    test_prices = test_prices.to_numpy().astype(np.float32)


    return train_data, test_data, train_prices, test_prices

def slidingWindow(array, window_size):
    windowed_array = []
    for i in range(0, np.shape(array)[0]-window_size+1):
        window = array[i:i+window_size]
        windowed_array.append(window)
    return np.array(windowed_array)

def sliding_window_test(arr, train_data, window_size):
    # Grab window size - 1 from the end of the training set 
    start = train_data.shape[0] - (window_size - 1) 
    end_of_training = train_data[start:, :]
    stacked = np.vstack([end_of_training, arr])

    # Do the same thing as sliding window 
    arr = slidingWindow(stacked, window_size)
    return arr
    

def preprocess(train_data, test_data, train_prices, test_prices, window_size):
    # FIXME: This is the previous implementation if you want to run on other data
    #return slidingWindow(train_data,window_size), slidingWindow(test_data, window_size), train_prices[window_size-1:], test_prices[window_size-1:]

    # New one that uses the twitter data
    return slidingWindow(train_data,window_size), sliding_window_test(test_data, train_data, window_size), train_prices[window_size-1:], test_prices
    

if __name__=="__main__":
    BATCH_SIZE = 1
    TEST_PROB = 0.2
    NUM_EPOCHS = 100
    WINDOW_SIZE = 5
    num_cols = 7
    EPOCHS = 10

    # Load in the data
    train_data, test_data, train_prices, test_prices = get_data(TEST_PROB)
    train_data, test_data, train_prices, test_prices = preprocess(train_data, test_data, train_prices, test_prices, WINDOW_SIZE)
    print(np.shape(train_data))
    print(np.shape(test_data))
    print(np.shape(train_prices))
    print(np.shape(test_prices))


    # Create keras RNN model
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, amsgrad=False)

    rnn = Sequential()
    rnn.add(LSTM(20, batch_input_shape=(BATCH_SIZE, WINDOW_SIZE, num_cols), return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(20, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(20, return_sequences=True))
    rnn.add(Flatten())
    # rnn.add(Dense(64))
    # rnn.add(Dropout(0.2))
    # rnn.add(Dense(128))
    # rnn.add(Dropout(0.2))
    # rnn.add(Dense(256))
    # rnn.add(Dropout(0.2))
    # rnn.add(Dense(512))
    # rnn.add(Dropout(0.2))
    rnn.add(Dense(1))

    # Compile keras model
    rnn.compile(optimizer=optimizer, loss='mse')
    rnn.fit(train_data, train_prices, epochs=EPOCHS, validation_data=(test_data, test_prices), batch_size=BATCH_SIZE, shuffle=False, verbose=2)
    rnn.summary()

    # Evaluate how well the rnn did
    predictions = rnn.predict(test_data, batch_size=1)
    print(predictions)
    print(predictions.shape)

    plt.plot(test_prices)
    plt.plot(predictions)
    plt.show()



