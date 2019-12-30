import matplotlib.pyplot as plt
import numpy as np
from io import *

def plotLevel(infos, wordIdList, columnDict, tsne_i_0, tsne_i_1, tsne_i, plotTitle):
    '''Finding the xlim and ylim of each entire tsne corpora'''
    x = [info[columnDict[tsne_i_0]] for info in infos]
    y = [info[columnDict[tsne_i_1]] for info in infos]
    # find the xlim and ylim of the entire corpora
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.scatter(x, y, alpha=1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    '''Ploting the tsne_i graph'''
    fig2, ax2 = plt.subplots(figsize=(100, 100))
    x = [item[0] for item in tsne_i]
    y = [item[1] for item in tsne_i]
    ax2.scatter(x, y, alpha=1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    for i, _ in enumerate(x):
        ax2.annotate(tsne_i[i][2], (x[i], y[i]))

    plt.title(plotTitle)
    plt.show()

def plot(infos, wordIdList, toPlot, columnDict):

    l = [
        {'tsne_i_0': 'tsne_0_0', 'tsne_i_1': 'tsne_0_1', 'tsne_i': toPlot['embeddings1'], 'plotTitle': "TSNE 0"},
        {'tsne_i_0': 'tsne_1_0', 'tsne_i_1': 'tsne_1_1', 'tsne_i': toPlot['embeddings2'], 'plotTitle': "TSNE 1"},
        {'tsne_i_0': 'tsne_2_0', 'tsne_i_1': 'tsne_2_1', 'tsne_i': toPlot['embeddings3'], 'plotTitle': "TSNE 2"},
        {'tsne_i_0': 'tsne_3_0', 'tsne_i_1': 'tsne_3_1', 'tsne_i': toPlot['embeddings4'], 'plotTitle': "TSNE 3"},
    ]

    for x in l:
        if x['tsne_i'] != 0:
            plotLevel(infos, wordIdList, columnDict, **x)
