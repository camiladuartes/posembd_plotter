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
from tsne_pos import centroid

def parseQueries(queries, vocab, tag2id):
    ## Different queries are separeted with a ' '
    queries = queries.split(' ')

    ## Structure where queries will be saved
    queriesDict = {}

    boolCentroid = False
    for query in queries:
        ## Retrieve token and its dataset+POS
        token, datasetPos = query.split('/')

        ## Check if we want centroid or not
        if query == queries[len(queries)-1]:
            if token == "centroidOn":
                boolCentroid = True
                break
        ## Retrieve token ID from saved vocab
        tokenId = vocab[token]

        ## If its set to retrieve all POS from token
        if datasetPos == '*':
            queriesDict[tokenId] = '*'
        else:
            ## Else, get its POS id
            dataset, pos = datasetPos.split('_')


            ## If token is already at the structure
            if tokenId in queriesDict:
                ## And not as '*' option, add query to list
                if queriesDict[tokenId] != '*':
                    queriesDict[tokenId].append((dataset, pos))
            else:
                queriesDict[tokenId] = [(dataset, pos)]

    return queriesDict, boolCentroid


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
        dataset = info[columnDict['dataset']]
        goldPOS = info[columnDict['gold_tag']]
        predPOS = info[columnDict['pred_tag']]
        posTuple = (dataset, goldPOS)

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
            tsnes = [(info[columnDict['tsne_{}_0'.format(i)]], info[columnDict['tsne_{}_1'.format(i)]])
                                    for i in range(4)]
            infosToPlot.append((wordForm, (goldPOS, predPOS), info[columnDict['dataset']], tsnes))
    return infosToPlot, infosToPlotColumnDict


def calculateLimits(infos, columnDict):
    print('> Calculating plot limits')
    lims = []
    for i in range(4):
        x = [info[columnDict['tsne_{}_0'.format(i)]] for info in tqdm(infos, "X coords, TSNE {}".format(i))]
        y = [info[columnDict['tsne_{}_1'.format(i)]] for info in tqdm(infos, "Y coords, TSNE {}".format(i))]

        npx = np.asarray(x)
        npy = np.asarray(y)

        lims.append(((npx.min(), npx.max()), (npy.min(), npy.max())))


    print('< Done!')
    return lims


def plotter(infos, columnDict, vocab, wordIdList, tagDicts):
    id2tag, tag2id = tagDicts
    lims = calculateLimits(infos, columnDict)

    while True:
        print('Query. Separate different queries with \' \'. Use * for all possible entries.')
        print('Type "centroidOn/*" at the end to plot the centroid of representations by POS.')
        queries = input()
        plt.close('all')
        # try:
        queriesDict, boolCentroid = parseQueries(queries, vocab, tag2id)
        # except:
            # continue
        infosToPlot, infosToPlotColumnDict = getInfosToPlot(queriesDict, infos, columnDict, wordIdList, id2tag)

        fig_ = [4]
        w, h = pyautogui.size()
        window_ = [(0,0), (0,h), (w,0), (w,h)]

        for i in range(4):
            ## Plot title
            plotTitle = 'TSNE {}'.format(i)

            ## Retrieving x and y coords for TSNEs to be plot
            x = [infoToPlot[infosToPlotColumnDict['tsnes']][i][0] for infoToPlot in infosToPlot]
            y = [infoToPlot[infosToPlotColumnDict['tsnes']][i][1] for infoToPlot in infosToPlot]

            ### Centroid
            if boolCentroid == True:
                ## Preparing dict to pass through centroid function
                dict_cent = {}
                for infoToPlot in infosToPlot:
                    x = infoToPlot[infosToPlotColumnDict['tsnes']][i][0]
                    y = infoToPlot[infosToPlotColumnDict['tsnes']][i][1]
                    word = infoToPlot[infosToPlotColumnDict['word']]
                    pos = infoToPlot[infosToPlotColumnDict['pos']]
                    if (word, pos) in dict_cent.keys():
                        dict_cent[(word, pos)].append([x, y])
                    else:
                        dict_cent[(word, pos)] = [x, y]

                ## Passing the dictionary with the word and its pos with their respective tsnes to the function
                centroids = []
                for wordPos, tsnesByPos in dict_cent.items():
                    # If there's more than one occurrence (list of lists of tsnes)
                    if all(isinstance(t, list) for t in tsnesByPos):
                        x_ = [t[0] for t in tsnesByPos]
                        y_ = [t[1] for t in tsnesByPos]
                    else:
                        x_ = tsnesByPos[0]
                        y_ = tsnesByPos[1]
                    word = wordPos[0]
                    pos = wordPos[1]
                    centroids.append(centroid(word, pos, x_, y_))

                ## Assigning x's and y's to plot
                x = []
                y = []
                for c in centroids:
                    x.append(c[1][0])
                    y.append(c[1][1])

            # Turning pixels into inches (divided by 2 because there are 4 figures)
            figs_x = ((w/2)-55) * 0.010416666666819
            figs_y = ((h/2)-55) * 0.010416666666819
            fig = plt.figure(i, clear=True, figsize=(figs_x, figs_y))
            fig_.append(fig)
            fig.canvas.manager.window.move(window_[i][0], window_[i][1])

            if boolCentroid:
                # c[0][0] here is each word in centroids
                colors = [float(hash(c[0][0]) % 256) / 256 for c in centroids]
            else:
                colors = [float(hash(infoToPlot[infosToPlotColumnDict['word']]) % 256) / 256
                                    for infoToPlot in infosToPlot]
            plt.scatter(x, y, alpha=1, c=colors, cmap=cm.brg, edgecolors=[(0.0, 0.0, 0.0, 1.0) for _ in colors])

            ax = plt.gca()
            ax.set_xlim(lims[i][0])
            ax.set_ylim(lims[i][1])

            if boolCentroid:
                for c in centroids:
                    word = c[0][0]
                    pos = c[0][1]
                    x = c[1][0]
                    y = c[1][1]
                    plt.annotate(" {} {}".format(word, pos), (x, y))
            else:
                for j, infoToPlot in enumerate(infosToPlot):
                    word = infoToPlot[infosToPlotColumnDict['word']]
                    pos = infoToPlot[infosToPlotColumnDict['pos']]
                    plt.annotate(" {} {}".format(word, pos), (x[j], y[j]))
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
