import pandas 

# Trading strategy is going to be: If slope for current iteration is positive, buy all, if slope is negative, sell all
# Another assumption is that whenever we buy or sell, we immediately are able to execute that position, meaning
# that we do not need to wait for someone to buy or for someone to sell to us first.

# TODO: 
#   1.) I have to import the stock data (that can just be an internal function)
#   2.) Takes in prediction plot (I will assume that it is a 2d array for now where it first column is the stock hour datetime, and the second column is the predicted price)
#           - in order for this to work correctly, will need to get some kind of (get stock price)
#           - need a field in order to keep track of number of trades conducted by the actual boy
#           - need field to keep track of how much money the script has.

# TODO:
    # We probably want a graph function where we are able to graph what the RNN thought that it would be able to make and then graph it against what it really makes.
        # Since I am graphing, I need to keep track of each step.

class StockTradingObj:
    '''
    Initialize the stock trading object such that we are able to predict the 
    earnings that our predicted strategy array will create versus the actual 
    stock data price.

    Parameters
    initial_cash - represents how much money the stock simulator starts with
    stock_data - 1d array that represents the actual stock data  
    predicted_arr - 1d array that represents the predicted stock data from our RNN
    dates - 1d array that represents datetime when trades occur.
    trade_fee - Trading fee (in USD) executing a trade on the market
    '''
    def __init__(self, initial_cash, stock_data, predicted_arr, dates, trade_fee):
        # Data needed to do the predictions
        self.predicted_arr = predicted_arr
        self.stock_data = stock_data
        self.dates = dates
        self.predicted_arr_slope = determine_slopes(predicted_arr)

        # Keep track of actual trading prices for graphing later
        self.datetime = [] # TODO: initialize this to be the first datetime thing.
        self.actual_lst = []

        # Track number of trades, depending on whether or not we want to factor
        # that into our calculations
        self.num_trades = 0
        self.trade_fee = trade_fee

        # Cash available for trades and number of shares
        self.cash_avail = initial_cash
        self.num_shares = 0

    def graph_simulation(self):
        '''
        After running the trading simulation which populates the fields, this
        function will overlay both of the plots over each other. It will then 
        save it into the output folder.
        '''
        # TODO: I am not quite sure how to actually do this.
    
    def graph_original_predict(self):
        '''
        This function will just graph the predicted original graph such that 
        we can compare it to what the actual original stock graph looks like.
        It will overlay the two graphs on top of each other and then it will 
        be saved to the output folder.
        '''
        # TODO: I am not quite sure how to actually do this.

    def determine_slopes(self, predicted_arr):
        '''
        Go through the n length predicted array of stock price and for each 
        pair of stock prices, produce the estimated slope. 

        Returns: 
            - A n-1 length array representing the slopes of each of the pairs of 2
        '''
        length = predicted_arr.shape[0] - 1
        slopes = np.zeros(length)

        # Loop through each of the pairs to get the slope for each step
        for i in range(0, length):
            slopes[i] = (predicted_arr[i] + predicted_arr[i + 1]) / 2
        
        return slopes

    def conduct_trade(self, num_shares, cost):
        '''
        Update the actual trading parameters when we conduct a trade
        
        Params
        - num_shares: represents number of shares to buy
        - cost: represents cost of share at this price.
        '''
    
    def determine_wealth(self):
        '''
        Use this to update the actual_lst, representing the amount of money 
        in the account after the trading hour has elapsed.
        '''
    
    def run(self):
        '''
        Runs the entire simulation for the amount of time that exists
        '''

            







        