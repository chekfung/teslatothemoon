import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd

class Stock_RNN(tf.keras.Model):
	def __init__(self, batch_size):
		super(Stock_RNN, self).__init__()

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01) # Optimizer
		self.batch_size = batch_size # Take one day's worth of data at a time
		self.model = tf.keras.Sequential()
		
		self.model.add(tf.keras.layers.LSTM(128, return_sequences=True)) # LSTM layer
		self.model.add(tf.keras.layers.LSTM(64, return_sequences=False)) # LSTM layer
		self.model.add(tf.keras.layers.Dense(16, use_bias=True, kernel_initializer='uniform', activation='relu')) # Dense layer
		self.model.add(tf.keras.layers.Dense(1, use_bias=True, kernel_initializer='uniform')) # Dense layer
	@tf.function
	def call(self, stock_input):
		return self.model(stock_input)

	def accuracy_function(self, stock_inputs, real_prices):

		predicted_results = self.model(stock_inputs)
		accuracy = tf.keras.losses.MSE(predicted_results,real_prices)
		return accuracy


	def loss_function(self, predictions, actual):
		# Gets the average loss
		return tf.reduce_mean(tf.keras.losses.MSE(predictions,actual))

def train(model, train_data, train_prices, num_epochs):

	current_epoch = 0
	total_num_of_data = np.shape(train_data)[0]
	while current_epoch < num_epochs: # Loops through all batches
		current_batch_number = 0
		while current_batch_number < total_num_of_data:
			with tf.GradientTape() as tape:
				predictions = model.call(train_data[current_batch_number:current_batch_number+model.batch_size]) # Get the probabilities for each batch
				loss = model.loss_function(predictions,train_prices[current_batch_number:current_batch_number+model.batch_size]) # Gets the loss
				print("Current MSE on epoch",current_epoch,":",loss)
			# Gets the gradients for this batch
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Does gradient descent
			current_batch_number+=model.batch_size # Goes to next batch
		current_epoch += 1
	pass

def test(model,test_data, test_prices):
	predictions = model.call(test_data)
	mse = model.loss_function(predictions, test_prices)
	print("Test MSE:",mse)
	
def get_data(test_prob=0.2):
	conn = sqlite3.connect("../data/rnn_data.db")
	data = pd.read_sql("SELECT * FROM RNNData", conn).to_numpy()
	data_without_date = data[:,1:].astype(np.float64)
	total_points = np.shape(data_without_date)[0]
	train_data = data_without_date[:int((1-test_prob)*total_points)]
	test_data = data_without_date[int((1-test_prob)*total_points):]
	train_prices = train_data[1:,3]
	test_prices = test_data[1:,3]
	train_data = train_data[:-1]
	test_data = test_data[:-1]
	return train_data, test_data, train_prices, test_prices

def slidingWindow(array, window_size):
	windowed_array = []
	for i in range(0, np.shape(array)[0]-window_size+1):
		window = array[i:i+window_size]
		windowed_array.append(window)
	return np.array(windowed_array)

def preprocess(train_data, test_data, train_prices, test_prices, window_size):
	return slidingWindow(train_data,window_size), slidingWindow(test_data,window_size), slidingWindow(train_prices,window_size), slidingWindow(test_prices,window_size)
	
if __name__=="__main__":
	rnn = Stock_RNN(24)
	TEST_PROB = 0.2
	NUM_EPOCHS = 250
	WINDOW_SIZE = 24
	train_data, test_data, train_prices, test_prices = get_data(TEST_PROB)
	train_data, test_data, train_prices, test_prices = preprocess(train_data, test_data, train_prices, test_prices)
	train(rnn, train_data, train_prices, NUM_EPOCHS)
	test(rnn, test_data, test_prices)
	