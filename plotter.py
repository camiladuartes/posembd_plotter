from tsne_pos.plot import plot, plotLevel
from tsne_pos.utils import calculateLimits
from tsne_pos.io import readInfoFile, readVocabFile, readTagsFile
import sys

import argparse

def addToPlot(toPlot, infoIndex):
    toPlot['embeddings1'].append(infoIndex)
    toPlot['embeddings2'].append(infoIndex)
    toPlot['embeddings3'].append(infoIndex)
    toPlot['embeddings4'].append(infoIndex)

def plotter(infos, columnDict, vocab, wordIdList, tagDicts):
    id2tag, tag2id = tagDicts

    while True:
        print('Query. Separate different queries with ' '. Use * for all possible entries.')
        s = input()
        queries = s.split(' ')
        queriesDict = {}
        for query in queries:
            token, datasetPos = query.split('/')
            tokenId = vocab[token]

            if datasetPos == '*':
                queriesDict[tokenId] = '*'
            else:
                dataset, pos = datasetPos.split('_')
                posId = tag2id[(dataset, pos)]

                if tokenId in queriesDict:
                    if queriesDict[tokenId] != '*':
                        queriesDict[tokenId].append((posId, dataset, pos))
                else:
                    queriesDict[tokenId] = [(posId, dataset, pos)]

        toPlot = {
            'embeddings1' : [],
            'embeddings2' : [],
            'embeddings3' : [],
            'embeddings4' : [],
        }

        for infoIndex, info in enumerate(infos):
            wordId = info[columnDict['id_word']]
            posTuple = (info[columnDict['dataset']], info[columnDict['gold_tag']])

            addInfo = False

            if '*' in queriesDict:
                if queriesDict[wordId] == '*':
                    addInfo = True
                    continue
                else:
                    for posId, dataset, pos in queriesDict['*']:
                        if posTuple == (dataset, posId):
                            addInfo = True

            if wordId in queriesDict:
                if queriesDict[wordId] == '*':
                    addInfo = True
                else:
                    for posId, dataset, pos in queriesDict[wordId]:
                        if posTuple == (dataset, posId):
                            addInfo = True


            if addInfo:
                addToPlot(toPlot, infoIndex)

        # Calculating xlim and ylim
        xlim = {}
        ylim = {}
        l = [
            {'tsne_i_0': 'tsne_0_0', 'tsne_i_1': 'tsne_0_1', 'info_indexes': toPlot['embeddings1']},
            {'tsne_i_0': 'tsne_1_0', 'tsne_i_1': 'tsne_1_1', 'info_indexes': toPlot['embeddings2']},
            {'tsne_i_0': 'tsne_2_0', 'tsne_i_1': 'tsne_2_1', 'info_indexes': toPlot['embeddings3']},
            {'tsne_i_0': 'tsne_3_0', 'tsne_i_1': 'tsne_3_1', 'info_indexes': toPlot['embeddings4']},
        ]
        for x in l:
            if len(x['info_indexes']) != 0:
                xlim[x['tsne_i_0']], ylim[x['tsne_i_1']] = calculateLimits(infos, columnDict, x['tsne_i_0'], x['tsne_i_1'])

        plot(infos, wordIdList, toPlot, columnDict, id2tag, xlim, ylim)

parser = argparse.ArgumentParser()
parser.add_argument("infosPath", help="path of infos csv file")
parser.add_argument("vocabPath", help="path of vocab csv file")
parser.add_argument("tagsPath", help="path of tags csv file")
args = parser.parse_args()
infosPath = args.infosPath
vocabPath = args.vocabPath
tagsPath = args.tagsPath

infos, columnDict = readInfoFile(infosPath)
wordIdList, vocab = readVocabFile(vocabPath)
tagDicts = readTagsFile(tagsPath)

plotter(infos, columnDict, vocab, wordIdList, tagDicts)
