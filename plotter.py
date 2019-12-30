from tsne_pos.plot import plot, plotLevel
from tsne_pos.io import readInfoFile, readVocabFile
import sys

def addToPlot(toPlot, info, columnDict):
    tokenName = wordIdList[info[columnDict['id_word']]] + ' ' + str(info[columnDict['gold_tag']])
    toPlot['embeddings1'].append((info[columnDict['tsne_0_0']], info[columnDict['tsne_0_1']], tokenName))
    toPlot['embeddings2'].append((info[columnDict['tsne_1_0']], info[columnDict['tsne_1_1']], tokenName))
    toPlot['embeddings3'].append((info[columnDict['tsne_2_0']], info[columnDict['tsne_2_1']], tokenName))
    toPlot['embeddings4'].append((info[columnDict['tsne_3_0']], info[columnDict['tsne_3_1']], tokenName))

def plotter(infos, columnDict, vocab, wordIdList):
    while True:
        print('Query. Separate different queries with ' '. Use * for all possible entries.')
        s = input()
        queries = s.split(' ')
        queriesDict = {}
        for query in queries:
            token, pos = query.split('/')
            tokenId = vocab[token]
            # posId = None
            if tokenId in queriesDict:
                queriesDict[tokenId].append(pos)
            else:
                queriesDict[tokenId] = [pos]

        print(queriesDict)

        toPlot = {
            'embeddings1' : [],
            'embeddings2' : [],
            'embeddings3' : [],
            'embeddings4' : [],
        }

        for info in infos:
            wordId = info[columnDict['id_word']]
            posId = (info[columnDict['dataset']], info[columnDict['gold_tag']])

            addInfo = False
            checkWord = ['*', wordId]
            checkPos = ['*', posId]

            for x in checkWord:
                if x in queriesDict:
                    for y in checkPos:
                        if y in queriesDict[x]:
                            addInfo = True

            if addInfo:

                addToPlot(toPlot, info, columnDict)

        plot(infos, wordIdList, toPlot, columnDict)

params = sys.argv[1:]
infosPath = params[0]
vocabPath = params[1]

infos, columnDict = readInfoFile(infosPath)
wordIdList, vocab = readVocabFile(vocabPath)

plotter(infos, columnDict, vocab, wordIdList)
