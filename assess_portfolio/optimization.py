"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from scipy.optimize import minimize, rosen, rosen_der
import analysis as analysis

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    sv = 1000000
    rfr = 0.0
    sf = 252.0
    allocs = find_optimal_allocations(prices, sv, rfr, sf) # add code here to find the allocations
    port_val = analysis.get_portfolio_value(prices,allocs,sv)
    cr, adr, sddr, sr = analysis.compute_portfolio_stats(port_val, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    return allocs, cr, adr, sddr, sr

def find_optimal_allocations(prices, sv, rfr, sf):
    bounds = [(0,1)] * prices.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    minimization_result = minimize(sharpe_ratio, np.random.random(len(prices.columns)), args=(prices, sv, rfr, sf),
                      bounds=bounds, constraints=constraints)
    return minimization_result.x

#TODO: maybe use **kwargs for function signature instead.
def sharpe_ratio(allocs, *args):
    prices = args[0]
    sv = args[1]
    rfr = args[2]
    sf = args[3]
    normalized_prices = (prices / prices.iloc[0])
    port_val = (normalized_prices * allocs * sv).sum(axis=1)
    daily_rets = port_val.copy()
    daily_rets = (daily_rets / daily_rets.shift(1)) - 1
    daily_rets.ix[0, 0] = 0  # set first element to zero
    return -((daily_rets - rfr).mean() / daily_rets.std()) * np.sqrt(sf)


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbols,
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
