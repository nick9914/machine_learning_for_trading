"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1),
    syms = ['GOOG','AAPL','GLD','XOM'],
    allocs=[0.1,0.2,0.3,0.4],
    sv=1000000, rfr=0.0, sf=252.0,
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get portfolio statistics (note: std_daily_ret = volatility)
    port_val = get_portfolio_value(prices, allocs, sv)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, rfr, sf) # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        port_val_norm = port_val / port_val.iloc[0]
        prices_SPY_norm = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat([port_val_norm, prices_SPY_norm], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp)
        pass

    # Add code here to properly compute end value
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0):
    cr = (port_val[-1] / port_val[0]) - 1

    daily_rets = port_val.copy() # This is a series
    daily_rets = (daily_rets / daily_rets.shift(1)) - 1
    daily_rets.ix[0, 0] = 0 #set first element to zero

    # average daily return - how much return on investment a stock makes on average daily
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    # sharpe ratio: average return when risk is taken into account
    sr = ((daily_rets - rfr).mean() / daily_rets.std()) * np.sqrt(sf)

    return cr, adr, sddr, sr

def get_portfolio_value(prices, allocs, sv):
    normalized_prices = prices / prices.iloc[0]
    # determine the entire portfolio value on each day by summing all the values in a row (convert to a series)
    return (normalized_prices * allocs * sv).sum(axis=1)

def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0 #set daily returns for row 0 to 0
    return daily_returns

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,
        syms = symbols,
        allocs = allocations,
        sv = start_val,
        gen_plot = True)

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
    test_code()
