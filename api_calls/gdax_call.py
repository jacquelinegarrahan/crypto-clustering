from gdax import public_client
from datetime import datetime
import time
import numpy as np


def get_symbols(dictionary):
    symbols, names = np.array(list(dictionary.items())).T
    return symbols, names


def convert_datetime(date_list):
    date = datetime(date_list[0], date_list[1], date_list[2])
    unix_time = time.mktime(date.timetuple())
    iso_time = date.isoformat()
    return iso_time


def get_historical_quotes_gdax(symbol_1, symbol_2, date1, date2, period):
    pair = symbol_1 + '-' + symbol_2
    gdax_call = public_client.PublicClient()
    if isinstance(date1, list):
        date1 = convert_datetime(date1)
    if isinstance(date2, list):
        date2 = convert_datetime(date2)
    history = gdax_call.get_product_historic_rates(pair, date1, date2, period)
    return history


def api_call(coin_dict, anchor_coin, start_date, end_date, period):
    symbols, names = get_symbols(coin_dict)
    quotes = []
    for symbol in symbols:
        quote = get_historical_quotes_gdax(anchor_coin, symbol, start_date, end_date, period)
        quotes.append(quote)
    return symbols, names, quotes


def get_variation(quotes):
    variation = []
    for quote in quotes:
        quote_variation = []
        for i in range(len(quote)):
            high = float(quote[i][2])
            low = float(quote[i][1])
            quote_variation.append(high-low)
        variation.append(quote_variation)
    return variation


if __name__ == "__main__":
    pair = 'BTC-USD'
    date1 = [2017, 9, 1]
    date1 = convert_datetime(date1)
    print(date1)
    date2 = [2017, 9, 5]
    date2 = convert_datetime(date2)
    period = 1000
    symbol_1 = 'BTC'
    symbol_2 = 'USD'
    gdax = public_client.PublicClient()
 #   print(gdax.get_product_historic_rates(pair, date1, date2, period))
    print(get_historical_quotes_gdax(symbol_1, symbol_2, date1, date2, period))