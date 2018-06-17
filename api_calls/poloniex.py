"""Interpretation of poloniex API outputs generated using poloniex library"""

from datetime import datetime
from poloniex import Poloniex
import numpy as np
import time


def retry(f, n_attempts=3):
    "Wrapper function to retry function calls in case of exceptions"
    def wrapper(*args, **kwargs):
        for i in range(n_attempts):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if i == n_attempts - 1:
                    raise
    return wrapper


def convert_datetime(date_list):
    date = datetime(date_list[0], date_list[1], date_list[2])
    unix_time = time.mktime(date.timetuple())
    return unix_time


def historical_quotes_polionex(symbol_1, symbol_2, date1, date2, period):
    """Get historical data from Poloniex for symbol_1-symbol_2 trading pair.

    Parameters
    ----------
    symbols : str
        Ticker symbol to query, ex ``"BTC"``.
    date1 : datetime.datetime
        Start date.
    date2 : datetime.datetime
        End date.

    Returns
    -------

    """
    #NEED TO ADJUST FOR API CALL TIMING
    pair = symbol_1 + '_' + symbol_2
    polo = Poloniex()
    if isinstance(date1, list):
        date1 = convert_datetime(date1)
    if isinstance(date2, list):
        date2 = convert_datetime(date2)
    history = polo.returnChartData(pair, period, date1, date2)
    return history


def api_call(coin_dict, anchor_coin, start_date, end_date, period):
    symbols, names = get_symbols(coin_dict)
    quotes = []
    for symbol in symbols:
        quote = historical_quotes_polionex(anchor_coin, symbol, start_date, end_date, period)
        quotes.append(quote)
    return symbols, names, quotes


def read_output(quote_list):
    """Interprets output of api call"""
    date_list = []
    high_list = []
    low_list = []
    open_list = []
    close_list = []
    var_list = []
    for i in quote_list:
        date = float(i['date'])
        high = float(i['high'])
        low = float(i['low'])
        open = float(i['open'])
        close = float(i['close'])
        var = high - low
        date_list.append(date)
        high_list.append(high)
        low_list.append(low)
        open_list.append(open)
        close_list.append(close)
        var_list.append(var)
    return date_list, high_list, low_list, open_list, close_list, var_list


def get_symbols(dictionary):
    symbols, names = np.array(list(dictionary.items())).T
    return symbols, names


def get_variation(quotes):
    variation = []
    for quote in quotes:
        quote_variation = []
        for i in range(len(quote)):
            high = float(quote[i]['high'])
            low = float(quote[i]['low'])
            quote_variation.append(high-low)
        variation.append(quote_variation)
    return variation


