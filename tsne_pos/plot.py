import matplotlib.pyplot as plt
import numpy as np
from io import *

def plotLevel(infos, wordIdList, columnDict, id2tag, tsne_i_0, tsne_i_1, info_indexes, plotTitle):
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
    x = [infos[infoIndex][columnDict[tsne_i_0]] for infoIndex in info_indexes]
    y = [infos[infoIndex][columnDict[tsne_i_1]] for infoIndex in info_indexes]

    ax2.scatter(x, y, alpha=1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    for i, infoIndex in enumerate(info_indexes):
        dataset = infos[infoIndex][columnDict['dataset']]
        word = wordIdList[infos[infoIndex][columnDict['id_word']]]
        pos = id2tag[(dataset, infos[infoIndex][columnDict['gold_tag']])]
        ax2.annotate(" {} {}".format(word, pos), (x[i], y[i]))

    plt.title(plotTitle)
    plt.show()

def plot(infos, wordIdList, toPlot, columnDict, id2tag):

    l = [
        {'tsne_i_0': 'tsne_0_0', 'tsne_i_1': 'tsne_0_1', 'info_indexes': toPlot['embeddings1'], 'plotTitle': "TSNE 0"},
        {'tsne_i_0': 'tsne_1_0', 'tsne_i_1': 'tsne_1_1', 'info_indexes': toPlot['embeddings2'], 'plotTitle': "TSNE 1"},
        {'tsne_i_0': 'tsne_2_0', 'tsne_i_1': 'tsne_2_1', 'info_indexes': toPlot['embeddings3'], 'plotTitle': "TSNE 2"},
        {'tsne_i_0': 'tsne_3_0', 'tsne_i_1': 'tsne_3_1', 'info_indexes': toPlot['embeddings4'], 'plotTitle': "TSNE 3"},
    ]

    for x in l:
        if len(x['info_indexes']) != 0:
            plotLevel(infos, wordIdList, columnDict, id2tag, **x)
