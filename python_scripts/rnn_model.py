import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv 
import trading


from preprocess import Preprocess

class Stock_RNN(tf.keras.Model):
    def __init__(self, batch_size):
        super(Stock_RNN, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1=0.5, beta_2=0.9) # Optimizer
        self.batch_size = batch_size # Take one day's worth of data at a time
        self.model = tf.keras.Sequential()
        
        #self.model.add(tf.keras.layers.LSTM(128, return_sequences=True)) # LSTM layer
        #self.model.add(tf.keras.layers.Dropout(0.4))

        #self.model.add(tf.keras.layers.LSTM(64, return_sequences=True)) # LSTM layer
        #self.model.add(tf.keras.layers.Dropout(0.4))

        # self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.Dense(128, use_bias=True, activation='relu'))
        # # self.model.add(tf.keras.layers.Dropout(0.2))
        # # self.model.add(tf.keras.layers.Dense(256, use_bias=True, activation='relu')) # Dense layer
        # # self.model.add(tf.keras.layers.Dropout(0.2))
        # self.model.add(tf.keras.layers.Dense(1, use_bias=True)) # Dense layer
        
        # Convolutional Layers
        self.model.add(tf.keras.layers.Conv2D(100, 2, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Conv2D(200, 2, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Conv2D(300, 2, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size = 2))

        # self.model.add(tf.keras.layers.Conv2D(256, 2, strides=(2,2), padding='same', activation='relu'))
        # self.model.add(tf.keras.layers.Dropout(0.2))

        # self.model.add(tf.keras.layers.Conv2D(512, 2, strides=(2,2), padding='same', activation='relu'))
        # self.model.add(tf.keras.layers.Dropout(0.2))

        # Head
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, use_bias=True, activation='relu')) # Dense layer
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(128, use_bias=True, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(256, use_bias=True, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(512, use_bias=True, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, use_bias=True)) # Dense layer
        
    @tf.function
    def call(self, stock_input):
        return self.model(stock_input)

    def accuracy_function(self, predicted_results, real_prices):
        real_prices = tf.convert_to_tensor(real_prices)
        accuracy = tf.reduce_mean(tf.square(predicted_results-real_prices))
        return accuracy


    def loss_function(self, predictions, actual):
        # Gets the average loss
        return tf.reduce_mean(tf.keras.losses.MSE(predictions,actual))

def train(model, train_data, train_prices, test_data, test_prices, num_epochs):

    current_epoch = 0
    total_num_of_data = np.shape(train_data)[0]
    while current_epoch < num_epochs: # Loops through all batches
        current_batch_number = 0
        while current_batch_number < total_num_of_data:
            with tf.GradientTape() as tape:
                predictions = model.call(train_data[current_batch_number:current_batch_number+model.batch_size]) # Get the probabilities for each batch
                loss = model.loss_function(predictions,train_prices[current_batch_number:current_batch_number+model.batch_size]) # Gets the loss
                # Gets the gradients for this batch
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Does gradient descent
                current_batch_number+=model.batch_size # Goes to next batch
        train_predictions = model.call(train_data)
        test_predictions = model.call(test_data)
        print("Current Train MSE on epoch",current_epoch,":",model.accuracy_function(train_predictions, train_prices))
        print("Current Test MSE on epoch",current_epoch,":",model.accuracy_function(test_predictions, test_prices))

        # TODO: Lowest so far I have found is 94
        if model.accuracy_function(train_predictions, train_prices) < 3:
            break
        current_epoch += 1
    pass

def test(model,test_data, test_prices):
    predictions = model.call(test_data)
    mse = model.loss_function(predictions, test_prices)
    print("Test MSE:",mse)
    
def get_data(test_prob=0.2, on_rnn_set = False, use_twitter=True):
    # FIXME: Just uncomment this stuff out
    if on_rnn_set:
        conn = sqlite3.connect("../data/rnn_data.db")
        data = pd.read_sql("SELECT * FROM RNNData", conn).to_numpy()
        data_without_date = data[:,1:].astype(np.float32)
        total_points = np.shape(data_without_date)[0]
        train_data = data_without_date[:int((1-test_prob)*total_points)]
        test_data = data_without_date[int((1-test_prob)*total_points):]
        train_prices = train_data[1:,3]
        test_prices = test_data[1:,3]
        train_data = train_data[:-1]
        test_data = test_data[:-1]
        return train_data, test_data, train_prices, test_prices
    else:
        # Preprocess the actual data
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
        if not use_twitter:
            train_data = train_data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            test_data = test_data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    
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
    

def preprocess(train_data, test_data, train_prices, test_prices, window_size, on_rnn_set=False):
    # FIXME: This is the previous implementation if you want to run on other data
    if on_rnn_set:
        return slidingWindow(train_data,window_size), slidingWindow(test_data, window_size), train_prices[window_size-1:], test_prices[window_size-1:]
    else:
        # New one that uses the twitter data
        return slidingWindow(train_data,window_size)[:-window_size], sliding_window_test(test_data, train_data, window_size), train_prices[window_size-1:-window_size], test_prices
    
if __name__=="__main__":
    BATCH_SIZE = 6
    TEST_PROB = 0.2
    NUM_EPOCHS = 2000
    WINDOW_SIZE = 3
    on_rnn_set = False
    INITIAL_CASH = 10000
    TRADE_COST = 0
    rnn = Stock_RNN(BATCH_SIZE)
    train_data, test_data, train_prices, test_prices = get_data(TEST_PROB, on_rnn_set=on_rnn_set, use_twitter=False)
    train_data, test_data, train_prices, test_prices = preprocess(train_data, test_data, train_prices, test_prices, WINDOW_SIZE, on_rnn_set=on_rnn_set)
    train_data, test_data, train_prices, test_prices = train_data[...,None], test_data[...,None], train_prices[...,None], test_prices[...,None]
    print(np.shape(train_data))
    print(np.shape(test_data))
    print(np.shape(train_prices))
    print(np.shape(test_prices))
    train(rnn, train_data, train_prices, test_data, test_prices, NUM_EPOCHS)
    #test(rnn, test_data, test_prices)

    rnn_twitter = Stock_RNN(BATCH_SIZE)
    train_data_twitter, test_data_twitter, train_prices_twitter, test_prices_twitter = get_data(TEST_PROB, on_rnn_set=on_rnn_set, use_twitter=True)
    train_data_twitter, test_data_twitter, train_prices_twitter, test_prices_twitter = preprocess(train_data_twitter, test_data_twitter, train_prices_twitter, test_prices_twitter, WINDOW_SIZE, on_rnn_set=on_rnn_set)
    train_data_twitter, test_data_twitter, train_prices_twitter, test_prices_twitter = train_data_twitter[...,None], test_data_twitter[...,None], train_prices_twitter[...,None], test_prices_twitter[...,None]
    train(rnn_twitter, train_data_twitter, train_prices_twitter, test_data_twitter, test_prices_twitter, NUM_EPOCHS)
    
    stockTrader = trading.StockTradingObj(INITIAL_CASH, test_prices.flatten(), rnn_twitter(test_data_twitter).numpy().flatten(), rnn(test_data).numpy().flatten(), np.arange(np.shape(test_prices)[0]), TRADE_COST)
    stockTrader.run()
    stockTrader.graph_simulation()
    