import matplotlib.pyplot as plt
import numpy as np
from io import *

def plotLevel(info, tsne_i_0, tsne_i_1, tsne_i, wordIdList, plotTitle):
    '''Finding the xlim and ylim of each entire tsne corpora'''
    x = info[:][columnDict[tsne_i_0]]
    y = info[:][columnDict[tsne_i_1]]
    # find the xlim and ylim of the entire corpora
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.scatter(x, y, alpha=1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    '''Ploting the tsne0 graph'''
    fig2, ax2 = plt.subplots(figsize=(100, 100))
    x = [item[0] for item in tsne_i]
    y = [item[1] for item in tsne_i]
    ax2.scatter(x, y, alpha=1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    for i, _ in enumerate(x):
        ax2.annotate(wordIdList[i], (x[i], y[i]))

    plt.title(plotTitle)
    plt.show()

def plot(vocabFile, infoFile):

    # list of strings
    wordIdList = readVocabFile(vocabFile)
    # list of lists + columnDict
    info, columnDict = readInfoFile(infoFile)

    # list of tsnes to plot
    tsne0, tsne1, tsne2, tsne3 = query(info, columnDict)

    # gold_tags = info[:][columnDict["gold_tag"]]
    # pred_tags = info[:][columnDict["pred_tag"]]

    l = [
        'tsne_0_0', 'tsne_0_1', tsne0, "TSNE 0",
        'tsne_1_0', 'tsne_1_1', tsne1, "TSNE 1",
        'tsne_2_0', 'tsne_2_1', tsne2, "TSNE 2",
        'tsne_3_0', 'tsne_3_1', tsne3, "TSNE 3",
    ]

    for x in l:
        if len(x[2]) != 0:
            plotLevel(info, l[0], l[1], l[2], wordIdList, l[3])



'''
returns: 4 lists of lists of tsnes to plot
'''
def query(info, columnDict):
    tsne0 = []
    tsne1 = []
    tsne2 = []
    tsne3 = []
    paramsDict = readPlotParameters()

    for num in range(0, 4):
        if len(paramsDict['tsne']) == 0 or num in paramsDict['tsne']:
            for infoLine in info:
                validLine = True
                for key in paramsDict:
                    if key != 'tsne':
                        if len(key) != 0:
                            for elem in key:
                                if elem not in infoLine:
                                    validLine = False
                if validLine == True:
                    if num == 0:
                        tsne0.append((infoLine['tsne_0_0'], infoLine['tsne_0_1']))
                    elif num == 1:
                        tsne1.append((infoLine['tsne_1_0'], infoLine['tsne_1_1']))
                    elif num == 2:
                        tsne2.append((infoLine['tsne_2_0'], infoLine['tsne_2_1']))
                    elif num == 3:
                        tsne3.append((infoLine['tsne_3_0'], infoLine['tsne_3_1']))

    return tsne0, tsne1, tsne2, tsne3
