import pandas as pd
import numpy as np
import marketsimcode as ms
import datetime as dt
from util import get_data, plot_data
import more_itertools
import indicators as indicator


# Code implementing a ManualStrategy object (your manual strategy). It should implement testPolicy() which returns
# a trades data frame (see below).
# The main part of this code should call marketsimcode as necessary to generate the plots used in the report.

# The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding that position.


def test_policy_bench_mark():
    start_date = '01/01/2008'
    end_date = '12/31/2009'
    date_range = pd.date_range(start_date, end_date)  # TODO: replace with call to function
    trade_columns = ['Date', 'Symbol', 'Order', 'Shares']
    df_trade = pd.DataFrame(data=np.zeros(shape=(len(date_range), len(trade_columns))),
                            columns=trade_columns)
    df_trade['Date'] = date_range
    df_trade['Symbol'] = 'JPM'
    df_trade['Order'] = 'BUY'
    df_trade.iloc[0] = [date_range[0], 'JPM', 'BUY', 1000]
    return df_trade


def test_policy_theoretical_optimal_strategy(symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                             st=100000, shares_contraint=1000):
    df_trades = create_trades_df(start_date=sd, end_date=ed)
    df_trades.set_index('Date', inplace=True)
    df_trades['Symbol'] = symbol
    df_trades['Order'] = 'BUY'
    df_prices = get_data([symbol], pd.date_range(sd, ed), False)
    df_price_filter = df_prices.dropna(subset=[symbol])
    iterator = more_itertools.peekable(df_price_filter.iterrows())
    ltd_shares = 0.0
    for index, row in iterator:
        current_price = row[0]
        next_day_price = iterator.peek((np.NaN, [np.NAN]))[1][0]
        if np.isnan(next_day_price):
            break
        delta = current_price - next_day_price
        if delta > 0:
            # We want to sell, next day is going to be cheaper
            add_sell_order(df_trades, index, shares_contraint, ltd_shares)
        else:
            # We want to buy, next day the price will go up
            add_buy_order(df_trades, index=index, shares_constraint=shares_contraint, ltd_shares=ltd_shares)
        # update_life_to_date_shares
        ltd_shares = update_life_to_date_shares(ltd_shares, df_trades, index, shares_contraint)
    return df_trades.reset_index()


def update_life_to_date_shares(ltd_shares, df_trades, index, shares_constraint):
    action = df_trades.loc[index]['Order']
    action_shares = df_trades.loc[index]['Shares']
    result = 0.0
    if action == 'BUY':
        result = ltd_shares + action_shares
    elif action == 'SELL':
        result = ltd_shares - action_shares
    elif action == 'NOTHING':
        result = ltd_shares
    else:
        raise RuntimeError('Action is neither BUY, SELL, or NOTHING.')
    if abs(result) > shares_constraint:
        raise RuntimeError('shares constraint is violated.')
    else:
        return result

def add_sell_order(df_trades, index, shares_constraint, ltd_shares):
    df_trades.at[index, 'Order'] = 'SELL'
    df_trades.at[index, 'Shares'] = abs(ltd_shares + shares_constraint)


def add_buy_order(df_trades, index, shares_constraint, ltd_shares):
    df_trades.at[index, 'Order'] = 'BUY'
    df_trades.at[index, 'Shares'] = abs(ltd_shares - shares_constraint)


def get_lbd_shares(df_trades, index):
    if index <= df_trades.first_valid_index():
        return 0.0
    else:
        loc_of_index = df_trades.index.get_loc(index)
        return df_trades.iloc[loc_of_index - 1]['Shares']


def create_trades_df(start_date, end_date):
    trade_columns = ['Date', 'Symbol', 'Order', 'Shares']
    date_range = pd.date_range(start_date, end_date)
    df_trade = pd.DataFrame(data=np.zeros(shape=(len(date_range), len(trade_columns))),
                            columns=trade_columns)
    df_trade['Date'] = date_range
    return df_trade


def test_policy(symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2012, 12, 31),
                st=100000, shares_contraint=1000, look_back_period=14):
    df_trades = create_trades_df(start_date=sd, end_date=ed)
    df_trades.set_index('Date', inplace=True)
    df_trades['Symbol'] = symbol
    df_trades['Order'] = 'NOTHING'
    df_prices_sym = get_data([symbol], pd.date_range(sd, ed), False)
    df_prices_idx = get_data(['SPY'], pd.date_range(sd, ed), False, dropNonTradingSPY=False)
    df_price_filter_sym = df_prices_sym.dropna(subset=[symbol])
    df_price_filter_idx = df_prices_idx.dropna(subset=['SPY'])
    iterator = more_itertools.peekable(df_price_filter_sym.iloc[look_back_period:].iterrows())
    ltd_shares = 0.0
    prev_sym_price_over_sma = indicator.get_price_over_sma(df_price_filter_sym.iloc[:look_back_period])
    for index, row in iterator:
        # get current price to determine when we should close the position.
        df_prices_historical_sym = df_price_filter_sym.loc[:index][-look_back_period:] # Todo: Can we do this in one shot?
        df_prices_historical_idx = df_price_filter_idx.loc[:index][-look_back_period:]

        sym_price_over_sma = indicator.get_price_over_sma(df_prices_historical_sym)
        sym_bollinger_band_percent = indicator.get_bollinger_band_percent(df_prices_historical_sym)
        sym_rsi = indicator.get_rsi(df_prices_historical_sym, look_back_period)
        idx_price_over_sma = indicator.get_price_over_sma(df_prices_historical_idx)
        idx_bollinger_band_percent = indicator.get_bollinger_band_percent(df_prices_historical_idx)
        idx_rsi = indicator.get_rsi(df_prices_historical_idx, look_back_period)
        signal = get_signal(sym_price_over_sma, sym_bollinger_band_percent, sym_rsi, idx_price_over_sma,
                           idx_bollinger_band_percent, idx_rsi, prev_sym_price_over_sma)
        print(signal)
        process_signal(df_trades, index, signal, ltd_shares, shares_contraint)
        ltd_shares = update_life_to_date_shares(ltd_shares, df_trades, index, shares_contraint)
        prev_sym_price_over_sma = sym_price_over_sma
    return df_trades.reset_index()



def process_signal(df_trades, index, signal, ltd_shares, shares_constraint):
    if signal == 1:
        add_buy_order(df_trades, index, shares_constraint, ltd_shares)
    elif signal == -1:
        add_sell_order(df_trades, index, shares_constraint, ltd_shares)
    elif signal == 0:
        add_close_order(df_trades, index, ltd_shares)
    else:
        pass


def add_close_order(df_trades, index, ltd_shares):
    if ltd_shares > 0:
        df_trades.at[index, 'Order'] = 'SELL'
        df_trades.at[index, 'Shares'] = abs(ltd_shares)
    elif ltd_shares < 0:
        df_trades.at[index, 'Order'] = 'BUY'
        df_trades.at[index, 'Shares'] = abs(ltd_shares)
    else:
        # already closed. Nothing to do
        pass

def get_signal(sym_price_over_sma, sym_boillinger_band_percent, sym_rsi, idx_price_over_sma,
               idx_boillinger_band_percent, idx_rsi, prev_sym_price_over_sma):
    sym_over_sold = is_over_sold_aggressive(sym_price_over_sma, sym_boillinger_band_percent, sym_rsi)
    sym_over_bought = is_over_bought_conservative(sym_price_over_sma, sym_boillinger_band_percent, sym_rsi)
    idx_over_sold = is_over_sold_conservative(idx_price_over_sma, idx_boillinger_band_percent, idx_rsi)
    idx_over_bought = is_over_bought_aggressive(idx_price_over_sma, idx_boillinger_band_percent, idx_rsi)
    # print('sym_over_sold: ' + str(sym_over_sold) + ' sym_over_bought: ' + str(sym_over_bought))
    # Go Long when symbol is over sold but the index is not
    # Go Short whn the symbol is overbought but the index is not
    # Close the position when the symbol crosses through its SMA. Because we do not know what will happen
    if sym_over_sold and not idx_over_sold:
        return 1
    elif sym_over_bought and not idx_over_bought:
        return -1
    elif crossed_sma(sym_price_over_sma, prev_sym_price_over_sma) is True:
        return 0
    else:
        return np.NaN


def crossed_sma(price_over_sma, prev_price_over_sma):
    if price_over_sma >= 1 and prev_price_over_sma < 1:
        return True
    elif price_over_sma <= 1 and prev_price_over_sma > 1:
        return True
    else:
        return False


def is_over_sold_conservative(price_over_sma, boillinger_band_percent, rsi):
    # print('price_over_sma < 0.95: ' + str(price_over_sma < 0.95) + ' and ' + 'boillinger_band_percent < 0: ' + str(boillinger_band_percent < 0) + ' and ' + 'rsi < 30: ' + str(rsi < 30))
    return price_over_sma < 0.95 and boillinger_band_percent < 0 and rsi < 30

def is_over_bought_conservative(price_over_sma, boillinger_band_percent, rsi):
    return price_over_sma > 1.05 and boillinger_band_percent > 1 and rsi > 70

def is_over_sold_aggressive(price_over_sma, boillinger_band_percent, rsi):
    # print('price_over_sma < 0.95: ' + str(price_over_sma < 0.95) + ' and ' + 'boillinger_band_percent < 0: ' + str(boillinger_band_percent < 0) + ' and ' + 'rsi < 30: ' + str(rsi < 30))
    return price_over_sma < 0.97 and boillinger_band_percent < .37 and rsi < 60

def is_over_bought_aggressive(price_over_sma, boillinger_band_percent, rsi):
    return price_over_sma > 1.05 and boillinger_band_percent > .85 and rsi > 61.5

def test_code():
    # The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding that position.
    # bench_mark_trades_df = test_policy_bench_mark()
    # theoretical_optimal_df = test_policy_theoretical_optimal_strategy()
    indicators_df = test_policy()
    ms.market_sim_main(indicators_df, sv=100000, commission=0.0, impact=0.0)


if __name__ == "__main__":
    test_code()
