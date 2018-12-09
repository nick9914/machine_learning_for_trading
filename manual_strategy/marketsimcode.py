# An improved version of your marketsim code that accepts a "trades" data frame (instead of a file).

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(df_orders, start_val=1000000, commission=0.0, impact=0.0):
    start_date = pd.to_datetime(df_orders['Date'].min())
    end_date = pd.to_datetime(df_orders['Date'].max())
    date_range = pd.date_range(start_date, end_date)

    order_symbols = df_orders['Symbol'].unique()

    df_prices = create_df_prices(date_range, order_symbols)
    df_trades = create_df_trades(date_range, order_symbols, df_orders, df_prices, commission, impact)
    df_holdings = create_df_holdings(date_range, order_symbols, df_trades, start_val)
    df_value = create_df_value(date_range, df_holdings, df_prices)
    portvals = create_df_portval(df_value)
    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    return rv


def create_df_portval(df_value):
    # axis=0 along the rows, and axis=1 along the columns: axis='index', axis='columns'
    return df_value.sum(axis='columns')


def create_df_value(date_range, df_holdings, df_prices):
    # perform element wise multiplication
    return df_holdings.multiply(df_prices)


def create_df_holdings(date_range, order_symbols, df_trades, cash_starting_value):
    df_holdings = create_zeros_df((date_range.shape[0], len(order_symbols)), date_range, order_symbols)
    df_holdings['Cash'] = 0.0
    # create initial row
    df_holdings.iloc[0] = df_trades.iloc[0]
    # we have to add the starting value of cash
    df_holdings.iloc[0].loc['Cash'] += cash_starting_value

    for df_holding_idx in range(1, len(df_holdings)):
        prev_holding = df_holdings.iloc[df_holding_idx - 1]
        curr_trade = df_trades.iloc[df_holding_idx]
        df_holdings.iloc[df_holding_idx] = prev_holding + curr_trade

    return df_holdings


def create_df_trades(date_range, order_symbols, df_orders, df_prices, commission, impact):
    df_trades = create_zeros_df((date_range.shape[0], len(order_symbols)), date_range, order_symbols)
    df_trades['Cash'] = 0.0

    for date_symbol, grouped_orders in df_orders.groupby(['Date', 'Symbol']):
        net_shares = get_net_shares_from_grouped_orders(grouped_orders)
        df_trades.loc[pd.to_datetime(date_symbol[0]), date_symbol[1]] = net_shares
        # multiply by -1 because comission is a cost. Also += because we are grouping by date and symbol.
        df_trades.loc[pd.to_datetime(date_symbol[0]), 'Cash'] -= len(grouped_orders) * commission
        for order in grouped_orders.itertuples():
            entry_cost_j = impact * get_price_for_trade(date_symbol[0], date_symbol[1], df_prices) * order.Shares
            df_trades.loc[pd.to_datetime(date_symbol[0]), 'Cash'] -= entry_cost_j

    for trade in df_trades.itertuples():
        net_cash_for_trade = 0.0
        for sym in order_symbols:
            stock_price = get_price_for_trade(trade.Index, sym, df_prices)
            trade_quantity = getattr(trade, sym)
            if trade_quantity != 0.0 and not np.isnan(stock_price):
                # price and quantity available
                # multiply by -1 to make sells positive (getting money) and buys negative (giving money)
                net_cash_for_trade += -1 * (trade_quantity * stock_price)
            elif trade_quantity != 0.0 and np.isnan(stock_price):
                # quantity traded this day, but no price available
                raise RuntimeError('quantity traded this day, but no price available %s %s %f' % trade.Index, sym,
                                   trade_quantity)
            else:
                # trade quantity is zero, no need to get price to calculate diff in cash
                continue
        # df_trades.set_value(trade.Index, 'Cash', net_cash_for_trade)
        df_trades.at[trade.Index, 'Cash'] += net_cash_for_trade

    return df_trades


def get_price_for_trade(Index, symbol, df_prices):
    if df_prices.index.contains(Index) and symbol in df_prices:
        return df_prices.loc[Index, symbol]
    else:
        raise RuntimeError('Could not find price for this date and symbol: ' + Index + ', ' + symbol)


def get_net_shares_from_grouped_orders(trade_df):
    net_shares = 0.0
    for order in trade_df.itertuples():
        trade_share = order.Shares
        trade_order = order.Order
        if trade_order == 'BUY':
            net_shares = net_shares + trade_share
        elif trade_order == 'SELL':
            net_shares = net_shares - trade_share
        elif trade_order == 'NOTHING':
            pass
        else:
            print('Trade Order is neither BUY, SELL, nor NOTHING. Investigate')

    return net_shares


def create_df_prices(date_range, order_symbols):
    df_prices = get_data(order_symbols, date_range, False)
    # add Cash column to df_prices
    df_prices['Cash'] = 1.0
    df_prices_ffill = df_prices.fillna(method='ffill')
    df_price_ffill_bfill = df_prices_ffill.fillna(method='bfill')
    return df_price_ffill_bfill


def create_zeros_df(dimension, dates_ranges, columns):
    df = pd.DataFrame(np.zeros(dimension), index=dates_ranges)
    df.columns = columns
    return df


def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    cr = (port_val[-1] / port_val[0]) - 1

    daily_rets = port_val.copy()  # This is a series
    daily_rets = (daily_rets / daily_rets.shift(1)) - 1
    daily_rets.ix[0, 0] = 0  # set first element to zero

    # average daily return - how much return on investment a stock makes on average daily
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    # sharpe ratio: average return when risk is taken into account
    sr = ((daily_rets - rfr).mean() / daily_rets.std()) * np.sqrt(sf)

    return cr, adr, sddr, sr


def get_spy_prices_series(date_range):
    spy_prices_df = get_data(['SPY'], date_range, False, dropNonTradingSPY=False)
    spy_prices_df_ffil = spy_prices_df.fillna(method='ffill')
    spy_prices_df_ffil_bfill = spy_prices_df_ffil.fillna(method='bfill')
    return spy_prices_df_ffil_bfill[spy_prices_df_ffil.columns[0]]


def market_sim_main(df_orders, sv, commission, impact):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    # Process orders
    portvals = compute_portvals(df_orders, start_val=sv, commission=commission, impact=impact)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = pd.to_datetime(portvals.first_valid_index())
    end_date = pd.to_datetime(portvals.last_valid_index())
    date_range = pd.date_range(start_date, end_date)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    spy_prices = get_spy_prices_series(date_range)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(spy_prices)

    portvals_norm = portvals / portvals.iloc[0]
    spy_price_norm = spy_prices / spy_prices.iloc[0]

    plot_data(pd.concat([portvals_norm, spy_price_norm], keys=['Portfolio', 'SPY'], axis=1))

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
