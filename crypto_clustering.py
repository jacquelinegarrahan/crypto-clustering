
#!/usr/bin/env python3
"""
=====================================
Clustering of crytocurrencies 
=====================================
by Jacqueline Garrahan

"""
import os, sys
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold, preprocessing
from api_calls.gdax_call import get_variation, convert_datetime, api_call
from api_calls.poloniex import Poloniex
import csv


GDAX_coin_dict = { 'BTC': 'Bitcoin', 'ETH': 'Ethereum'}

USDT_coin_dict = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'BCH': 'Bitcoin Cash',  'LTC': 'Litecoin', 'XRP': 'Ripple',
             'XMR': 'Monero', 'ZEC': 'ZCash', 'ETC': 'Ethereum Classic', 'DASH': 'Dash', 'NXT': 'NXT',
             'STR': 'Stellar', 'REP': 'Augur'}


XMR_coin_dict = {'LTC': 'Litecoin', 'DASH': 'Dash', 'ZEC': 'Zcash', 'BCN': 'Bytecoin', 'NXT': 'NXT',
                 'MAID': 'MaidSafeCoin', 'BTCD': 'BitcoinDark', 'BLK': 'BlackCoin'}
ETH_coin_dict = {'LSK': 'Lisk', 'GNT': 'Golem', 'ZEC': 'Zcash', 'BCH': 'Bitcoin Cash', 'ZRX': 'Ox',
                 'ETC': 'Ethereum Classic', 'GNO': 'Gnosis', 'STEEM': 'STEEM', 'REP': 'Augur'}

BTC_coin_dict = {'ETH': 'Ethereum', 'XMR': 'Monero', 'LTC': 'Litecoin', 'XRP': 'Ripple', 'BCH': 'Bitcoin Cash',
                 'DGB': 'DigiByte', 'LSK': 'Lisk', 'DOGE': 'Dogecoin', 'XEM': 'NEM', 'BCN': 'Bytecoin',
                  'DASH': 'Dash', 'STR': 'Stellar', 'ETC': 'Ethereum Classic',  'REP': 'Augur', 'NAV': 'NAVCoin',
                 'MAID': 'MaidSafeCoin'}
BTC_coin_dict2 ={'FCT': 'Factom', 'VIA': 'Viacoin','GNT': 'Golem', 'STEEM': 'STEEM','BTS': 'BitShares',
                 'NXT': 'NXT', 'GAME': 'GameCredits', 'SYS': 'Syscoin', 'SC': 'Siacoin', 'STRAT': 'Stratis',}

                 #'PINK': 'Pinkcoin', 'LBC': 'LBRY Credits', 'DCR': 'Decred',
                 #'VTC': 'Vertcoin', 'BURST': 'Burst', 'GNO': 'Gnosis', 'SJCX': 'Storcoin X', 'EXP': 'Expanse',
                 #'EMC2': 'Einsteinium', 'RADS': 'Radium', 'AMP': 'Synereo AMP', 'POT': 'PotCoin', 'ARDR': 'Ardor',
                 #'FLO': 'Florincoin', 'CLAM': 'CLAMS', 'XCP': 'Counterparty', 'XBC': 'BitcoinPlus',
                 #'FLDC': 'FoldingCoin', 'NOTE': 'DNotes', 'OMNI': 'Omni', 'NEOS': 'Neoscoin', 'BLK': 'BlackCoin',
                 #'VRC': 'VeriCoin', 'NMC': 'Namecoin', 'GRC': 'Gridcoin Research', 'PASC': 'PascalCoin', 'NXC':
                 #'Nexium', 'RIC': 'Riecoin', 'HUC': 'Huntercoin', 'PPC': 'Peercoin', 'BELA': 'Bela', 'XVC': 'Vcash',
                 #'XPM': 'Primecoin', 'BTCD': 'BitcoinDark', 'NAUT': 'Nautiluscoin', 'BTM': 'Bitmark',
                 #'BCY': 'BitCrystals', 'SBD': 'Steem Dollars', 'ZRX': 'Ox'}
BTC_coin_dictfull = {'ETH': 'Ethereum', 'XMR': 'Monero', 'LTC': 'Litecoin', 'XRP': 'Ripple', 'BCH': 'Bitcoin Cash',
                     'DGB': 'DigiByte', 'LSK': 'Lisk', 'DOGE': 'Dogecoin', 'XEM': 'NEM', 'BCN': 'Bytecoin',
                     'DASH': 'Dash', 'STR': 'Stellar', 'ETC': 'Ethereum Classic',  'REP': 'Augur', 'NAV': 'NAVCoin',
                     'MAID': 'MaidSafeCoin','FCT': 'Factom', 'VIA': 'Viacoin','GNT': 'Golem', 'STEEM': 'STEEM',
                     'BTS': 'BitShares', 'NXT': 'NXT', 'GAME': 'GameCredits', 'SYS': 'Syscoin', 'SC': 'Siacoin',
                     'STRAT': 'Stratis', 'PINK': 'Pinkcoin', 'LBC': 'LBRY Credits', 'DCR': 'Decred', 'VTC': 'Vertcoin',
                     'BURST': 'Burst', 'GNO': 'Gnosis', 'SJCX': 'Storcoin X', 'EXP': 'Expanse', 'EMC2': 'Einsteinium',
                     'RADS': 'Radium', 'AMP': 'Synereo AMP', 'POT': 'PotCoin', 'ARDR': 'Ardor', 'FLO': 'Florincoin',
                     'CLAM': 'CLAMS', 'XCP': 'Counterparty', 'XBC': 'BitcoinPlus', 'FLDC': 'FoldingCoin',
                     'NOTE': 'DNotes', 'OMNI': 'Omni', 'NEOS': 'Neoscoin', 'BLK': 'BlackCoin', 'VRC': 'VeriCoin',
                     'NMC': 'Namecoin', 'GRC': 'Gridcoin Research', 'PASC': 'PascalCoin', 'NXC': 'Nexium',
                     'RIC': 'Riecoin', 'HUC': 'Huntercoin', 'PPC': 'Peercoin', 'BELA': 'Bela', 'XVC': 'Vcash',
                     'XPM': 'Primecoin', 'BTCD': 'BitcoinDark', 'NAUT': 'Nautiluscoin', 'BTM': 'Bitmark',
                     'BCY': 'BitCrystals', 'SBD': 'Steem Dollars', 'ZRX': 'Ox'}


# Retrieve the data from Internet
def prepare_data(coin_dict, anchor_coin, start_date, end_date, period):
    """ date : list [year, month, day]"""
    symbols, names, quotes = api_call(coin_dict, anchor_coin, start_date, end_date, period)
    variation = get_variation(quotes)
    max_len = max(len(quote) for quote in quotes)
    new_symbols = []
    new_names = []
    new_quotes = []
    trunc_variation = []
    for i in range(len(variation)):
        if len(variation[i]) == max_len:
            trunc_variation.append(variation[i])
            new_quotes.append(quotes[i])
            new_symbols.append(symbols[i])
            new_names.append(names[i])
    new_variation = np.array(trunc_variation)
    scaled_data = preprocessing.scale(new_variation)
    X = scaled_data.copy().T
    return new_symbols, new_names, new_quotes, X

def run_cluster_analysis(coin_dict, anchor_coin, start_date, end_date, period):
    symbols, names, quotes, X = prepare_data(coin_dict, anchor_coin, start_date, end_date, period)
    edge_model = covariance.GraphLassoCV()
    edge_model.fit(X)
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    ####labels for no entry
    n_labels = labels.max()
    names = np.array(names)
    clusters = []
    for i in range(n_labels + 1):
        cluster_append = [names[labels == i]]
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
        clusters.append(cluster_append)
    print(clusters)
    return X, names, edge_model, labels, n_labels, clusters


def plot_clusters(X, names, edge_model, labels, evolution=False):
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    #plot clusters
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    correlation_edges = edge_model.precision_.copy()

    d = 1 / np.sqrt(np.diag(correlation_edges))
    correlation_edges *= d
    correlation_edges *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(correlation_edges, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=200 * d ** 2, c='k',
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(correlation_edges[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.PuBu,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor='k',
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(), )
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

    if evolution == False:
        plt.show()


def run_and_plot(coin_dict, anchor_coin, start_date, end_date, period):
    """ Runs and plots results of cluster analysis """
    X, names, edge_model, labels, n_labels, clusters = run_cluster_analysis(coin_dict, anchor_coin, start_date, end_date, period)
    plot_clusters(X, names, edge_model, labels)
    return None


def run_and_plot_evolution(coin_dict, anchor_coin, start_date, end_date, period, chunks):
    """ Divides time period into discrete chucks.
    Runs cluster analysis over data for each chunk """
    start_date = convert_datetime(start_date)
    end_date = convert_datetime(end_date)
    delta = (np.floor(end_date - start_date))/chunks
    benchmark = start_date
    for t in range(chunks):
        end_point = benchmark + delta
        X, names, edge_model, labels, n_labels, clusters = run_cluster_analysis(coin_dict, anchor_coin, benchmark,
                                                                                end_point, period)
        plot_clusters(X, names, edge_model, labels, evolution=True)
        fig = plt.gcf()
        fig.savefig(anchor_coin + '_' + str(t) + '.pdf')
        benchmark = end_point
    for i in range(len(clusters)):
        print('cluster %s =' % i, clusters[i])
    return clusters

if __name__ == "__main__":
        run_and_plot(XMR_coin_dict, 'USD', [2017, 1, 1], [2017, 2, 5], 5000)
