import datetime as dt
import pandas as pd
from util import get_data
import numpy as np
import marketsimcode as ms


def create_sma_df(look_back_period, prices_df):
    sma = prices_df.cumsum()
    sma.values[look_back_period:, :] = (sma.values[look_back_period:, :] - sma.values[:-look_back_period,
                                                                           :]) / look_back_period
    sma.ix[:look_back_period, :] = np.nan
    return sma


def create_bbp_df(look_back_period, sma, prices_df):
    rolling_std_df = prices_df.rolling(window=look_back_period, min_periods=look_back_period).std()
    top_band = sma + (2 * rolling_std_df)
    bottom_band = sma - (2 * rolling_std_df)
    return (prices_df - bottom_band) / (top_band - bottom_band)


def create_price_over_sma_df(sma, prices_df):
    return prices_df / sma


def create_rsi_df(look_back_period, prices_df):
    daily_returns = prices_df.copy()
    daily_returns.values[1:, :] = prices_df.ix[1:, :].values - prices_df.ix[:-1, :].values
    daily_returns.ix[0, :] = np.nan
    price_went_up = daily_returns.where(daily_returns >= 0).fillna(0)
    price_went_down = daily_returns.where(daily_returns < 0).fillna(0)
    up_gain = price_went_up.rolling(window=look_back_period, min_periods=look_back_period).sum()
    down_loss = -1 * price_went_down.rolling(window=look_back_period, min_periods=look_back_period).sum()
    momentum = up_gain / down_loss
    # Inf results mean down_loss was 0. Those should be RSI 100. Looks like numpy handles this for us
    return 100 - (100 / (1 + momentum))


def create_spy_rsi(rsi):
    spy_rsi = rsi.copy()
    spy_rsi.values[:, :] = spy_rsi.loc[:, ['SPY']]
    return spy_rsi


def create_price_sma_cross_over_df(price_over_sma):
    # Create a binary (0 -1) array showing when price is above SMA-n day
    sma_cross = pd.DataFrame(0, index=price_over_sma.index, columns=price_over_sma.columns)
    sma_cross[price_over_sma >= 1] = 1
    sma_cross[1:] = sma_cross.diff()
    sma_cross.iloc[0] = 0
    return sma_cross


def create_orders_file():
    # The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding that position.
    # bench_mark_trades_df = test_policy_bench_mark()
    # theoretical_optimal_df = test_policy_theoretical_optimal_strategy()
    symbols = ['AAPL', 'GOOG', 'GLD', 'XOM', 'HD', 'VZ', 'KO']

    sd = dt.datetime(2001, 12, 31)
    ed = dt.datetime(2012, 01, 11)
    look_back_period = 30

    prices_df = get_data(symbols, pd.date_range(sd, ed), addSPY=True).dropna()
    sma = create_sma_df(look_back_period, prices_df.copy(deep=True))
    price_over_sma = create_price_over_sma_df(sma.copy(deep=True), prices_df.copy(deep=True))
    bbp = create_bbp_df(look_back_period, sma.copy(deep=True), prices_df.copy(deep=True))
    rsi = create_rsi_df(look_back_period, prices_df.copy(deep=True))

    spy_rsi = create_spy_rsi(rsi)
    # crate df for sma and price crosses.
    # 1: price went above sma
    # 0: No state changed
    # -1: price went below sma
    sma_cross = create_price_sma_cross_over_df(price_over_sma.copy(deep=True))

    # Strategy
    # Go Long: symbol is oversold and index is not
    # oversold: price_over_sma < 0.95 and bbp % < 0 and RSI < 30
    # Go short: symbol is overbought and index is not
    # overbought: price_over_sma > 1.05 and bbp % > 1 and RSI > 70
    # Close position: symbol crosses through SMA

    # Use the indicators to make some kind of trading decision for each day
    target_shares = prices_df.copy()
    target_shares.iloc[:, :] = np.NaN
    holdings = {sym: 0 for sym in symbols}

    target_shares[(price_over_sma < 0.95) & (bbp < 0) & (rsi < 30) & (spy_rsi > 30)] = 100
    target_shares[(price_over_sma > 1.05) & (bbp > 1) & (rsi > 70) & (spy_rsi < 70)] = -100

    target_shares[(sma_cross != 0)] = 0

    target_shares.ffill(inplace=True)
    target_shares.fillna(0, inplace=True)

    # Taking the diff will give us an order to place only when the target shares change
    orders = target_shares.copy().diff()
    orders.iloc[0] = 0

    del orders['SPY']
    orders = orders.loc[(orders != 0).any(axis=1)]

    df_trades = create_trades_df(sd, ed)
    order_list = []
    for idx, order_day in orders.iterrows():
        for sym in symbols:
            if order_day[sym] > 0:
                order_list.append([idx, sym, 'BUY', 100])
            elif order_day[sym] < 0:
                order_list.append([idx, sym, 'SELL', 100])

    for order in order_list:
        print "  ".join(str(x) for x in order)

    cols = ['Date', 'Symbol', 'Order', 'Shares']
    df_orders_test = pd.DataFrame(order_list, columns=cols)

    ms.market_sim_main(df_orders_test, sv=100000, commission=0.0, impact=0.0)


if __name__ == "__main__":
    create_orders_file()
