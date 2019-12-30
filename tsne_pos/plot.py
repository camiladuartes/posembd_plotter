import matplotlib.pyplot as plt
import numpy as np
from io import *

def plot(vocabFile, infoFile):

    # list of strings
    wordIdList = readVocabFile(vocabFile)
    # list of lists + columnDict
    info, columnDict = readInfoFile(infoFile)

    # list of tsnes to plot
    tsne0, tsne1, tsne2, tsne3 = query(info, columnDict)

    # gold_tags = info[:][columnDict["gold_tag"]]
    # pred_tags = info[:][columnDict["pred_tag"]]

    '''Finding the xlim and ylim of each entire tsne0 corpora'''
    if len(tsne0) != 0:
        x = info[:][columnDict["tsne_0_0"]]
        y = info[:][columnDict["tsne_0_1"]]
        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()

        '''Ploting the tsne0 graph'''
        fig2, ax2 = plt.subplots(figsize=(100, 100))
        x = [item[0] for item in tsne0]
        y = [item[1] for item in tsne0]
        ax2.scatter(x, y, alpha=1)
        ax2.set_xlim(xlim0)
        ax2.set_ylim(ylim0)
        for i, _ in enumerate(x):
            ax2.annotate(wordIdList[i], (x[i], y[i]))
        plt.title("tsne0")
        plt.show()

    '''Finding the xlim and ylim of each entire tsne1 corpora'''
    if len(tsne1) != 0:
        x = info[:][columnDict["tsne_1_0"]]
        y = info[:][columnDict["tsne_1_1"]]
        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        xlim1 = ax.get_xlim()
        ylim1 = ax.get_ylim()

        '''Ploting the tsne1 graph'''
        fig2, ax2 = plt.subplots(figsize=(100, 100))
        x = [item[0] for item in tsne1]
        y = [item[1] for item in tsne1]
        ax2.scatter(x, y, alpha=1)
        ax2.set_xlim(xlim1)
        ax2.set_ylim(ylim1)
        for i, _ in enumerate(x):
            ax2.annotate(wordIdList[i], (x[i], y[i]))
        plt.title("tsne1")
        plt.show()

    '''Finding the xlim and ylim of each entire tsne2 corpora'''
    if len(tsne2) != 0:
        x = info[:][columnDict["tsne_2_0"]]
        y = info[:][columnDict["tsne_2_1"]]
        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        xlim2 = ax.get_xlim()
        ylim2 = ax.get_ylim()

        '''Ploting the tsne2 graph'''
        fig2, ax2 = plt.subplots(figsize=(100, 100))
        x = [item[0] for item in tsne2]
        y = [item[1] for item in tsne2]
        ax2.scatter(x, y, alpha=1)
        ax2.set_xlim(xlim2)
        ax2.set_ylim(ylim2)
        for i, _ in enumerate(x):
            ax2.annotate(wordIdList[i], (x[i], y[i]))
        plt.title("tsne2")
        plt.show()

    '''Finding the xlim and ylim of each entire tsne3 corpora'''
    if len(tsne3) != 0:
        x = info[:][columnDict["tsne_3_0"]]
        y = info[:][columnDict["tsne_3_1"]]
        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        xlim3 = ax.get_xlim()
        ylim3 = ax.get_ylim()

        '''Ploting the tsne3 graph'''
        fig2, ax2 = plt.subplots(figsize=(100, 100))
        x = [item[0] for item in tsne3]
        y = [item[1] for item in tsne3]
        ax2.scatter(x, y, alpha=1)
        ax2.set_xlim(xlim3)
        ax2.set_ylim(ylim3)
        for i, _ in enumerate(x):
            ax2.annotate(wordIdList[i], (x[i], y[i]))
        plt.title("tsne3")
        plt.show()

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