import pandas as pd
import numpy as np


# Code that implements your indicators as functions that operate on dataframes.
# The "main" code in indicators.py should generate the charts that illustrate your indicators in the report.

def get_sma(df_historical_prices):
    return df_historical_prices.iloc[:, 0].mean()


def get_std(df_historical_prices):
    return df_historical_prices.iloc[:, 0].std()


def get_price_over_sma(df_historical_prices):
    sma = get_sma(df_historical_prices)
    return df_historical_prices.iloc[-1][0] / sma


def get_bollinger_band_percent(df_historical_prices):
    sma = get_sma(df_historical_prices)
    std = get_std(df_historical_prices)
    top_band = sma + (2 * std)
    bottom_band = sma - (2 * std)
    current_price = df_historical_prices.iloc[-1][0]
    return (current_price - bottom_band) / (top_band - bottom_band)


def get_rsi(df_historical_prices, look_back):
    df_historical_prices_shifted = df_historical_prices[1:]
    delta = df_historical_prices_shifted.iloc[:, 0].values - df_historical_prices.iloc[:, 0].values[:-1]
    up_gain = delta[np.where(delta > 0)[0]].sum()
    down_gain = abs(delta[np.where(delta < 0)[0]].sum())
    if down_gain == 0:
        return 100
    else:
        rs = (up_gain / look_back) / (down_gain / look_back)
        return 100 - (100 / (1 + rs))