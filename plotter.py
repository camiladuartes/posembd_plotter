from tsne_pos.io import readInfoFile, readVocabFile, readTagsFile
import sys

import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time

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
    for infoIndex, info in enumerate(infos):
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
        x = [info[columnDict['tsne_{}_0'.format(i)]] for info in infos]
        y = [info[columnDict['tsne_{}_1'.format(i)]] for info in infos]

        # find the xlim and ylim of the entire corpora
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.scatter(x, y, alpha=1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # so we don't plot all embeddings automatically:
        if plt.get_fignums():
            # window open
            plt.close(fig)
        lims.append((xlim, ylim))
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


        for i in range(4):
            ## Plot title
            plotTitle = 'TSNE {}'.format(i)

            ## Retrieving x and y coords for TSNEs to be plot
            x = [infoToPlot[infosToPlotColumnDict['tsnes']][i][0] for infoToPlot in infosToPlot]
            y = [infoToPlot[infosToPlotColumnDict['tsnes']][i][1] for infoToPlot in infosToPlot]
            plt.figure(i, clear=True)

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
        plt.show(block=False)


##################################### HANDLING ARGS ###################################


parser = argparse.ArgumentParser()
parser.add_argument("infosPath", help="path of infos csv file")
parser.add_argument("vocabPath", help="path of vocab csv file")
parser.add_argument("tagsPath", help="path of tags csv file")
args = parser.parse_args()

infos, columnDict = readInfoFile(args.infosPath)
wordIdList, vocab = readVocabFile(args.vocabPath)
tagDicts = readTagsFile(args.tagsPath)

plotter(infos, columnDict, vocab, wordIdList, tagDicts)
