import pandas as pd
import numpy as np 
import math
import sqlite3
import matplotlib.pyplot as plt
import os
import sys
import pandas
import seaborn as sns

sns.set()

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

    In an attempt to maintain the easiest trading strategy, both the optimal
    and the RNN data essentially use the predicted slope to determine whether 
    to buy or sell. That is, if the slope between the current time and the 
    next time is positive, buy all possible stock possible. If it is negative,
    sell all stock possible.

    An assumption that we make is that whenever we buy or sell, we will 
    automatically be allowed to execute that trade and we do not have to wait 
    for another user to buy or sell.
    '''
    def __init__(self, initial_cash, stock_data, predicted_arr, baseline, dates, trade_fee=0):
        '''
        Parameters
        initial_cash - represents how much money the stock simulator starts with
        stock_data - 1d array that represents the actual stock data  
        predicted_arr - 1d array that represents the predicted stock data from our RNN
        dates - 1d array that represents datetime when trades occur.
        trade_fee - Trading fee (in USD) executing a trade on the market; If not 
                    specified, trading_fee = 0
        '''
        # Data needed to do the predictions
        self.predicted_arr = predicted_arr
        self.stock_data = stock_data
        self.baseline = baseline
        self.dates = dates
        self.predicted_arr_slope = self.determine_slopes(predicted_arr)
        self.baseline_slope = self.determine_slopes(baseline)
        self.optimal_slope = self.determine_slopes(stock_data)

        # Keep track of actual trading prices for graphing later
        self.datetime = [] 
        self.actual_lst = []
        self.baseline_lst = []
        self.optimal_lst = []
        self.random_lst = []
        
        # Trading fee information
        self.trade_fee = trade_fee
        self.initial_cash = initial_cash

        # Track number of trades, depending on whether or not we want to factor
        # that into our calculations
        self.num_trades = 0
        self.cash_avail = initial_cash
        self.num_shares = 0

        # For optimal trading (if we knew what the stock graph looked like)
        self.num_trades_o = 0
        self.cash_avail_o = initial_cash
        self.num_shares_o = 0

        self.num_trades_r = 0
        self.cash_avail_r = initial_cash
        self.num_shares_r = 0

        self.num_trades_b = 0
        self.cash_avail_b = initial_cash 
        self.num_shares_b = 0

    def graph_simulation(self):
        '''
        After running the trading simulation which populates the fields, this
        function will overlay both of the plots over each other. It will then 
        save it into the output folder.
        '''
        if (len(self.datetime) != len(self.actual_lst)) or (len(self.datetime) != len(self.optimal_lst)):
            raise Exception('Datetime and actual_lst are not equal!')

        # If everything else is fine, continue
        plt.figure(1)
        plt.plot(self.datetime, self.actual_lst)
        plt.plot(self.datetime, self.baseline_lst)
        plt.plot(self.datetime, self.optimal_lst)
        plt.plot(self.datetime, self.random_lst)
        plt.legend(['Predicted Model Trader with Twitter Data', 'Baseline Trader without Twitter Data', 'Optimal Trader', 'Random Trader'])
        plt.xlabel('Trading Hours (3/2 - 3/6)')
        plt.ylabel('Total Account Balance on Platform (USD)')
        plt.title('Twitter Sentiment Trading Strategies on TESLA over 3/2 - 3/6'.format(self.initial_cash))
        
        # If want to save the file
        #plt.savefig('output/trading_strategies.png')
        plt.show()
    
    def graph_original_predict(self):
        '''
        This function will just graph the predicted original graph such that 
        we can compare it to what the actual original stock graph looks like.
        It will overlay the two graphs on top of each other and then it will 
        be saved to the output folder.
        '''
        # First check to make sure that they are all the same length
        if (not (self.predicted_arr.shape[0] == self.stock_data.shape[0]) and \
                (self.stock_data.shape[0] == self.dates.shape[0])):
            raise Exception('Predicted, Stock, or Dates NOT all the same length.')

        # If everything is fine, start graphing
        plt.figure(0)
        plt.plot(self.dates, self.stock_data, 'r')
        plt.plot(self.dates, self.predicted_arr, 'b')
        plt.legend(['Actual Stock Data', 'Predicted Stock Data'])
        plt.xlabel('Date')
        plt.ylabel('Tesla Price (USD)')
        plt.title('Stock Price of Tesla compared to RNN Predicted Price')
        
        # If want to save the file
        plt.savefig('output/original_vs_prediction.png')
        plt.show()


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
            slopes[i] = (predicted_arr[i + 1] - self.stock_data[i]) / 2
        
        return slopes


    def buy_stock(self, num_shares, cost, op_bool=0):
        '''
        Update the actual trading parameters when we conduct a trade
        
        Params
        - num_shares: represents number of shares to buy
        - cost: represents cost of share at this price.
        - op_bool: Initially set to false; state whether or not trading with RNN or optimal
        '''
        # Check if we have actual amount to trade, otherwise error occurred in code
        trade_cost = (num_shares * cost) + self.trade_fee 

        # If using RNN trading:
        if op_bool == 0:
            if trade_cost > self.cash_avail:
                raise Exception('RNN: Calculated math incorrectly. Trying to trade on margin')

            # Update number of trades
            self.num_trades += 1
            self.num_shares += num_shares
            self.cash_avail = self.cash_avail - trade_cost

        # If using Optimal trading:
        elif (op_bool == 1):
            if trade_cost > self.cash_avail_o:
                raise Exception('Optimal: Calculated math incorrectly. Trying to trade on margin')

            # Update number of trades
            self.num_trades_o += 1
            self.num_shares_o += num_shares
            self.cash_avail_o = self.cash_avail_o - trade_cost
        
        # If using strategy without twitter sentiment (Baseline)
        elif (op_bool == 2):
            if trade_cost > self.cash_avail_b:
                raise Exception('Baseline: Calcualted math incorrectly. Trying to trade on margin')

            # Update number of trades
            self.num_trades_b += 1
            self.num_shares_b += num_shares
            self.cash_avail_b = self.cash_avail_b - trade_cost

        # If using random strategy
        else:
            if trade_cost > self.cash_avail_r:
                raise Exception('Random: Calculated math incorrectly. Trying to trade on margin')

            # Update number of trades
            self.num_trades_r += 1
            self.num_shares_r += num_shares 
            self.cash_avail_r = self.cash_avail_r - trade_cost

    def sell_stock(self, num_shares, cost, op_bool=0):
        '''
        sell_stock is essentially the same method as buy_stock, but instead, 
        we sell the stock that we have.

        Params
        - num_shares: represents number of shares to buy
        - cost: represents cost of share at this price.
        - op_bool: Initially set to false; state whether or not trading with RNN or optimal
        '''
        # If RNN trading
        if op_bool == 0:
            # Check to make sure we have enough shares
            if self.num_shares < num_shares:
                raise Exception('RNN: Trying to sell shares that we do not have!')

            trade_gain = (num_shares * cost) - self.trade_fee  

            # Update our numbers
            self.num_trades += 1
            self.num_shares = self.num_shares - num_shares
            self.cash_avail += trade_gain 
        
        # If optimal trading:
        elif (op_bool == 1):
            # Check to make sure we have enough shares
            if self.num_shares_o < num_shares:
                raise Exception('Optimal: Trying to sell shares that we do not have!')

            trade_gain = (num_shares * cost) - self.trade_fee  

            # Update our numbers
            self.num_trades_o += 1
            self.num_shares_o = self.num_shares_o - num_shares
            self.cash_avail_o += trade_gain 
        
        # If using strategy without twitter sentiment data (Baseline)
        elif(op_bool == 2):
            if self.num_shares_b < num_shares:
                raise Exception('Baseline: Trying to see shares that we do not have!')
                
            trade_gain = (num_shares * cost) - self.trade_fee 

            # Update our numbers 
            self.num_trades_b += 1
            self.num_shares_b = self.num_shares_b - num_shares
            self.cash_avail_b += trade_gain
        
        # IF random trading:
        else:
            if self.num_shares_r < num_shares:
                raise Exception('Random: Trying to sell shares that we do not have!')
        
            trade_gain = (num_shares * cost) - self.trade_fee

            # Update numbers
            self.num_trades_r += 1
            self.num_shares_r = self.num_shares_r - num_shares
            self.cash_avail_r += trade_gain

    def determine_wealth(self, stock_price, op_bool=0):
        '''
        Use this to update the actual_lst, representing the amount of money 
        in the account after the trading hour has elapsed.

        Params
        - stock_price: price of stock at time of determining wealth

        Returns
        - amount that the account is worth
        '''
        wealth = 0
        # If RNN
        if op_bool == 0:
            trading_fees = self.trade_fee * self.num_trades
            stock_balance = stock_price * self.num_shares
            wealth = stock_balance + self.cash_avail - trading_fees

        # If Optimal instead
        elif (op_bool == 1):
            trading_fees = self.trade_fee * self.num_trades_o
            stock_balance = stock_price * self.num_shares_o
            wealth = stock_balance + self.cash_avail_o - trading_fees

        elif (op_bool == 2):
            trading_fees = self.trade_fee * self.num_trades_b 
            stock_balance = stock_price * self.num_shares_b
            wealth = stock_balance + self.cash_avail_b - trading_fees

        # Determine Random's wealth
        else:
            trading_fees = self.trade_fee * self.num_trades_r
            stock_balance = stock_price * self.num_shares_r
            wealth = stock_balance + self.cash_avail_r - trading_fees
    
        return wealth

    
    def run(self):
        '''
        Runs the entire simulation using the trading strategy that we have 
        implemented, given that we predict the stock price for the next day.

        Note Assumptions from Object Declaration:
        ========================================
        In an attempt to maintain the easiest trading strategy, both the optimal
        and the RNN data essentially use the predicted slope to determine whether 
        to buy or sell. That is, if the slope between the current time and the 
        next time is positive, buy all possible stock possible. If it is negative,
        sell all stock possible.

        An assumption that we make is that whenever we buy or sell, we will 
        automatically be allowed to execute that trade and we do not have to wait 
        for another user to buy or sell.
        ========================================
        '''
        # First check that the slopes were formed correctly
        if self.predicted_arr_slope.shape[0] != self.optimal_slope.shape[0]:
            raise Exception('Length of slope arrays are not the same')

        # Determine number of iterations of loop
        num_iterations = self.predicted_arr_slope.shape[0]

        # Run through iterations of the trading simulation
        for i in range(num_iterations):
            # Current stock price:
            stock_price = float(self.stock_data[i])

            # FIXME: We might need to make the date look nicer so that it graphs nicer
            self.datetime.append(self.dates[i])

            # RNN Trading Strategy
            rnn_slope = self.predicted_arr_slope[i]

            # Determine if buy, sell, or do nothing
            if rnn_slope > 0:
                # If positive slope, buy all shares possible
                liquid_cash = self.cash_avail - self.trade_fee
                num_shares = math.floor(liquid_cash / stock_price)

                if num_shares > 0:
                    # Buy shares on the market
                    self.buy_stock(num_shares, stock_price)

            elif rnn_slope < 0:
                if self.num_shares > 0:
                    # If negative slope, sell all shares we own
                    self.sell_stock(self.num_shares, stock_price)
            
            # Add to our list what our wealth is at current time step
            curr_wealth = self.determine_wealth(stock_price)
            self.actual_lst.append(curr_wealth)

            # Optimal Trading Strategy
            optimal_slope = self.optimal_slope[i]

            # Determine if buy, sell, or do nothing
            if optimal_slope > 0:
                liquid_cash = self.cash_avail_o - self.trade_fee
                num_shares = math.floor(liquid_cash / stock_price)

                if num_shares > 0:
                    # Buy shares on the market
                    self.buy_stock(num_shares, stock_price, op_bool=1)
   
            elif optimal_slope < 0:
                if self.num_shares_o > 0:
                    # If negative slope, sell all shares we own
                    self.sell_stock(self.num_shares_o, stock_price, op_bool=1)
            
            # Add to our list what optimal wealth is at current time step
            curr_wealth = self.determine_wealth(stock_price, op_bool=1)

            self.optimal_lst.append(curr_wealth)

            # Baseline Trading Strategy
            baseline_slope = self.baseline_slope[i]

            if baseline_slope > 0:
                liquid_cash = self.cash_avail_b - self.trade_fee 
                num_shares = math.floor(liquid_cash / stock_price)

                if num_shares > 0:
                    self.buy_stock(num_shares, stock_price, op_bool=2)
            elif optimal_slope < 0:
                if self.num_shares_b > 0:
                    self.sell_stock(self.num_shares_b, stock_price, op_bool=2)

            # Add to our list what our baseline wealth is at this time stamp
            curr_wealth = self.determine_wealth(stock_price, op_bool=2)

            self.baseline_lst.append(curr_wealth)

            # Random Trading Strategy
            random = np.random.rand()
            
            # Flip a coin to determine whether to buy or to sell
            if random > 0.5:
                if self.num_shares_r > 0:
                    self.sell_stock(self.num_shares_r, stock_price, op_bool=3)
            else:
                liquid_cash = self.cash_avail_r - self.trade_fee
                num_shares = math.floor(liquid_cash / stock_price)

                if num_shares > 0:
                    # Buy shares on the market
                    self.buy_stock(num_shares, stock_price, op_bool=3)
            
            curr_wealth = self.determine_wealth(stock_price, op_bool=3)
            self.random_lst.append(curr_wealth)

# =========================================================================== #
def main():
    # Parse the three columns: predicted_arr, stock_data, dates
    INITIAL_MONEY = 10000

    # Try trading on the regression stock
    FILENAME = "polynomial_regression.csv"
    path = os.path.join(os.path.dirname(sys.path[0]), "csv", FILENAME)
    df = pd.read_csv(path)

    # Baseline: "No Twitter Predicted Price"

    # Load stock trading object
    stockTrader = StockTradingObj(INITIAL_MONEY, df['Actual Price'].to_numpy(), df['Predicted Price'].to_numpy(), df["No Twitter Predicted Price"].to_numpy(), df['Test Hour'].to_numpy(), trade_fee=0)
    stockTrader.run()
    stockTrader.graph_simulation()
    


if __name__ == '__main__':
    main()







        