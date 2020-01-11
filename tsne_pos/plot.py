import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from io import *
from .utils import *

def plotLevel(cont, xlim, ylim, infos, wordIdList, columnDict, id2tag, tsne_i_0, tsne_i_1, info_indexes, plotTitle):
    '''Ploting the tsne_i graph'''
    x = [infos[infoIndex][columnDict[tsne_i_0]] for infoIndex in info_indexes]
    y = [infos[infoIndex][columnDict[tsne_i_1]] for infoIndex in info_indexes]
    plt.figure(cont)

    words = [wordIdList[infos[infoIndex][columnDict['id_word']]] for i, infoIndex in enumerate(info_indexes)]
    colors = [float(hash(s) % 256) / 256 for s in words]
    plt.scatter(x, y, alpha=1, c=colors, cmap=cm.brg)
    plt.xlim(xlim)
    plt.ylim(ylim)

    for i, infoIndex in enumerate(info_indexes):
        dataset = infos[infoIndex][columnDict['dataset']]
        word = wordIdList[infos[infoIndex][columnDict['id_word']]]
        pos = id2tag[(dataset, infos[infoIndex][columnDict['gold_tag']])]
        plt.annotate(" {} {}".format(word, pos), (x[i], y[i]))

    plt.title(plotTitle)
    plt.show()

def plot(infos, wordIdList, toPlot, columnDict, id2tag):

    l = [
        {'tsne_i_0': 'tsne_0_0', 'tsne_i_1': 'tsne_0_1', 'info_indexes': toPlot['embeddings1'], 'plotTitle': "TSNE 0"},
        {'tsne_i_0': 'tsne_1_0', 'tsne_i_1': 'tsne_1_1', 'info_indexes': toPlot['embeddings2'], 'plotTitle': "TSNE 1"},
        {'tsne_i_0': 'tsne_2_0', 'tsne_i_1': 'tsne_2_1', 'info_indexes': toPlot['embeddings3'], 'plotTitle': "TSNE 2"},
        {'tsne_i_0': 'tsne_3_0', 'tsne_i_1': 'tsne_3_1', 'info_indexes': toPlot['embeddings4'], 'plotTitle': "TSNE 3"},
    ]

    cont = 0
    for x in l:
        if len(x['info_indexes']) != 0:
            cont += 1
            xlim, ylim = calculateLimits(infos, columnDict, x['tsne_i_0'], x['tsne_i_1'])
            plotLevel(cont, xlim, ylim, infos, wordIdList, columnDict, id2tag, **x)
