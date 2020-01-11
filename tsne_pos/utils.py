import matplotlib.pyplot as plt

def convertToText(words):
    for i in range(len(words)):
        if len(words[i]) == 1:
            if words[i][0] == '\001':
                words[i] = "BOS"
            else:
                words[i] = "EOS"
        else:
            words[i] = "".join(words[i][1:-1])

def createVocab(wordList):
    word2id = {}
    idList = []
    for word in wordList:
        if word not in word2id:
            word2id[word] = len(word2id)
        idList.append(word2id[word])
    return word2id, idList


def convertToTagNames(datasetNames, datasets, tagIds):
    for i in range(len(tagIds)):
        for dataset in datasets:
            if dataset.name == datasetNames[i]:
                return dataset.id2tag[tagIds[i]]

def calculateLimits(infos, columnDict, tsne_i_0, tsne_i_1):
    '''Finding the xlim and ylim of each entire tsne corpora'''
    x = [info[columnDict[tsne_i_0]] for info in infos]
    y = [info[columnDict[tsne_i_1]] for info in infos]
    # find the xlim and ylim of the entire corpora
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.scatter(x, y, alpha=1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # so we don't plot all embeddings automatically:
    if plt.get_fignums():
        # window open
        plt.close(fig)
    return xlim, ylim

def centroid(infos, wordIdList, columnDict, id2tag, info_indexes, word, pos):
    x, y, occurenceNumber = 0
    for i, infoIndex in enumerate(info_indexes):
        if (word == wordIdList[infos[infoIndex][columnDict['id_word']]] and
            pos == id2tag[(dataset, infos[infoIndex][columnDict['gold_tag']])]):
            x += x[i]
            y += y[i]
            occurenceNumber += 1
    x /= occurenceNumber
    y /= occurenceNumber
    centroid = x, y
    return centroid, occurenceNumber
