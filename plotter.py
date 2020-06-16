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

import plotly.express as px
import plotly.subplots as ps
import plotly.graph_objects as go

import numpy as np
import time

import pandas as pd

from tqdm import tqdm




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
            if pos != '*':
                pos = str(tag2id[(dataset, pos)])

            ## If token is already at the structure
            if tokenId in queriesDict:
                ## And not as '*' option, add query to list
                if queriesDict[tokenId] != '*':
                    queriesDict[tokenId].append((dataset, pos))
            else:
                queriesDict[tokenId] = [(dataset, pos)]

    return queriesDict, boolCentroid


def getInfosToPlot(queriesDict, infos, columnDict, wordIdList):
    ## List that will contain TSNEs to be plotted
    infosToPlot = []

    ## Going through all infos and extracting infos of interest
#     for infoIndex, info in tqdm(enumerate(infos), "Extracting infos of interest", total=len(infos)):
    for infoIndex, info in enumerate(infos):

        addInfo = False
        wordId = info[columnDict['id_word']]
        dataset = info[columnDict['dataset']]
        goldPOS = info[columnDict['gold_tag']]
        posTuple = (dataset, goldPOS)

        ## If we want to plot all tokens
        if '*' in queriesDict:
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
            infosToPlot.append(info)

    tsne_columns = [x for x in columnDict.keys() if "tsne" in x]
    int_columns = [x for x in columnDict.keys() if "id" in x or "pos" in x]
    typesDict = {x: float for x in tsne_columns}
    typesDict.update({x: int for x in int_columns})
    df = pd.DataFrame(np.array(infosToPlot), columns=columnDict.keys()).astype(typesDict)

    return df


def calcCentroids(df):
    # drop id_token, id_sent, pos_sent, id_word
    # drop either pred_tag or gold_tag
    df.drop(['id_token', 'id_sent', 'pos_sent'], axis=1, inplace=True)
    df.drop(['pred_tag'], axis=1, inplace=True)
    # curr : ''dataset',  'id_word',  'gold_tag', 'tnse_i_j'

    g = df.groupby(['id_word', 'dataset', 'gold_tag']).mean()


    return g.reset_index().dropna(axis=0)


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


##################################### HANDLING ARGS ###################################


# parser = argparse.ArgumentParser()
# parser.add_argument("infosPath", help="path to infos csv file")
# parser.add_argument("vocabPath", help="path to vocab csv file")
# parser.add_argument("tagsPath", help="path to tags csv file")
# args = parser.parse_args()

# infosPath = args.infosPath
# vocabPath = args.vocabPath
# tagsPath = args.tagsPath

infosPath = 'infos.csv'
vocabPath = 'vocab.csv'
tagsPath = 'tags.csv'

infos, columnDict = readInfoFile(infosPath)
wordIdList, vocab = readVocabFile(vocabPath)
wordIdDict = {i:x for i, x in enumerate(wordIdList)}
tagDicts = readTagsFile(tagsPath)

id2tag, tag2id = tagDicts
lims = calculateLimits(infos, columnDict)


while True:
    print('Query. Separate different queries with \' \'. Use * for all possible entries.')
    print('Type "centroidOn/*" at the end to plot the centroid of representations by POS.')
    queries = input()
    try:
        queriesDict, boolCentroid = parseQueries(queries, vocab, tag2id)
    except:
        continue

    # isso aqui retorna um dataframe
    selectedInfosDF = getInfosToPlot(queriesDict, infos, columnDict, wordIdList)


    if boolCentroid:
        selectedInfosDF = calcCentroids(selectedInfosDF)

    selectedInfosDF['id_word'] = selectedInfosDF['id_word'].map(wordIdDict)

    fig = ps.make_subplots(rows=2, cols=2, shared_xaxes=False)

    for i in range(4):
        ## Plot title
        plotTitle = 'TSNE {}'.format(i)
        range_x, range_y = lims[i]

        temp_fig = px.scatter(selectedInfosDF,
                              x="tsne_{}_0".format(i),
                              y="tsne_{}_1".format(i),
                              hover_data=["id_word", 'dataset', 'gold_tag']
                             )

        temp_fig.update_xaxes(range=list(range_x))
        temp_fig.update_yaxes(range=list(range_y))

        trace = temp_fig['data'][0]

        fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)
        fig['layout']['xaxis{}'.format(i + 1)].update(range=list(range_x))
        fig['layout']['yaxis{}'.format(i + 1)].update(range=list(range_y))

    fig.show()
