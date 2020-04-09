'''
Script for plotting the trained TSNEs using WORD/POS queries
Usage:
    python plotter.py INFOS_PATH VOCAB_PATH TAGS_PATH

    INFOS_PATH: path where infos csv file is saved
    VOCAB_PATH: path where vocab csv file is saved
    TAGS_PATH: path where tags csv file is saved
'''

from tsne_pos.io import readInfoFile, readVocabFile, readTagsFile
import sys

import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time

from tqdm import tqdm

import pyautogui

def parseQueries(queries, vocab, tag2id):
    ## Different queries are separeted with a ' '
    queries = queries.split(' ')


    ## Structure where queries will be saved
    queriesDict = {}

    for query in queries:
        ## Retrieve token and its dataset+POS
        token, datasetPos = query.split('/')

        ## Retrieve token ID from saved vocab
        tokenId = vocab[token]

        ## If its set to retrieve all POS from token
        if datasetPos == '*':
            queriesDict[tokenId] = '*'
        else:
            ## Else, get its POS id
            dataset, pos = datasetPos.split('_')
            posId = tag2id[(dataset, pos)]

            ## If token is already at the structure
            if tokenId in queriesDict:
                ## And not as '*' option, add query to list
                if queriesDict[tokenId] != '*':
                    queriesDict[tokenId].append((dataset, posId))
            else:
                queriesDict[tokenId] = [(dataset, posId)]

    return queriesDict


def getInfosToPlot(queriesDict, infos, columnDict, wordIdList, id2tag):
    ## List that will contain TSNEs to be plotted
    infosToPlotColumnDict = {
        'word': 0,
        'pos': 1,
        'dataset': 2,
        'tsnes': 3,
    }
    infosToPlot = []

    ## Going through all infos and extracting infos of interest
    for infoIndex, info in tqdm(enumerate(infos), "Extracting infos of interest", total=len(infos)):
        addInfo = False
        wordId = info[columnDict['id_word']]
        posTuple = (info[columnDict['dataset']], info[columnDict['gold_tag']])

        ## If we want to plot all tokens
        if '*' in queriesDict:
            ## And all the POS
            if queriesDict[wordId] == '*':
                addInfo = True
            elif posTuple in queriesDict['*']:
                ## But not all POS, Add this info if it has a POS of interest
                addInfo = True

        ## If it is a specific token and it is in the queries structure
        if wordId in queriesDict:
            ## If we want all its POS
            if queriesDict[wordId] == '*':
                addInfo = True
            elif posTuple in queriesDict[wordId]:
                ## But not all POS, Add this info if it has a POS of interest
                addInfo = True

        if addInfo:
            wordForm = wordIdList[wordId]
            POS = id2tag[posTuple]
            tsnes = [(info[columnDict['tsne_{}_0'.format(i)]], info[columnDict['tsne_{}_1'.format(i)]])
                                    for i in range(4)]
            infosToPlot.append((wordForm, POS, info[columnDict['dataset']], tsnes))

    return infosToPlot, infosToPlotColumnDict


def calculateLimits(infos, columnDict):
    print('> Calculating plot limits')
    lims = []
    for i in range(4):
        x = [info[columnDict['tsne_{}_0'.format(i)]] for info in tqdm(infos, "X coords, TSNE {}".format(i))]
        y = [info[columnDict['tsne_{}_1'.format(i)]] for info in tqdm(infos, "Y coords, TSNE {}".format(i))]

        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        # so we don't plot all embeddings automatically:
        if plt.get_fignums():
            # window open
            plt.close(fig)
        lims.append((ax.get_xlim(), ax.get_ylim()))
    print('< Done!')
    return lims


def plotter(infos, columnDict, vocab, wordIdList, tagDicts):
    id2tag, tag2id = tagDicts
    lims = calculateLimits(infos, columnDict)

    while True:
        print('Query. Separate different queries with \' \'. Use * for all possible entries.')
        queries = input()
        plt.close('all')
        queriesDict = parseQueries(queries, vocab, tag2id)
        infosToPlot, infosToPlotColumnDict = getInfosToPlot(queriesDict, infos, columnDict, wordIdList, id2tag)

        fig_ = [4]
        # Getting screen resolution
        w, h = pyautogui.size()
        window_ = [(0,0), (0,h), (w,0), (w,h)]
        for i in range(4):
            ## Plot title
            plotTitle = 'TSNE {}'.format(i)

            ## Retrieving x and y coords for TSNEs to be plot
            x = [infoToPlot[infosToPlotColumnDict['tsnes']][i][0] for infoToPlot in infosToPlot]
            y = [infoToPlot[infosToPlotColumnDict['tsnes']][i][1] for infoToPlot in infosToPlot]

            # Turning pixels into inches (divided by 2 because there are 4 figures)
            figs_x = ((w/2)-55) * 0.010416666666819
            figs_y = ((h/2)-55) * 0.010416666666819
            fig = plt.figure(i, clear=True, figsize=(figs_x, figs_y))
            fig_.append(fig)
            fig.canvas.manager.window.move(window_[i][0], window_[i][1])

            colors = [float(hash(infoToPlot[infosToPlotColumnDict['word']]) % 256) / 256
                                for infoToPlot in infosToPlot]
            plt.scatter(x, y, alpha=1, c=colors, cmap=cm.brg, edgecolors=[(0.0, 0.0, 0.0, 1.0) for _ in colors])

            plt.xlim(lims[i][0])
            plt.ylim(lims[i][1])

            for i, infoToPlot in enumerate(infosToPlot):
                word = infoToPlot[infosToPlotColumnDict['word']]
                pos = infoToPlot[infosToPlotColumnDict['pos']]
                plt.annotate(" {} {}".format(word, pos), (x[i], y[i]))
            plt.title(plotTitle)

        print(">> Press \"q\" to close all figures.")
        def quit_figure(event):
            if event.key == 'q':
                for i in range(4):
                    plt.close(fig_[i])
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

        plt.show()


##################################### HANDLING ARGS ###################################


parser = argparse.ArgumentParser()
parser.add_argument("infosPath", help="path to infos csv file")
parser.add_argument("vocabPath", help="path to vocab csv file")
parser.add_argument("tagsPath", help="path to tags csv file")
args = parser.parse_args()

infos, columnDict = readInfoFile(args.infosPath)
wordIdList, vocab = readVocabFile(args.vocabPath)
tagDicts = readTagsFile(args.tagsPath)

plotter(infos, columnDict, vocab, wordIdList, tagDicts)
