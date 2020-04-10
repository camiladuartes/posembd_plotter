import matplotlib.pyplot as plt
from tqdm import tqdm

def convertToText(words):
    for i in tqdm(range(len(words))):
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


def convertToTagNames(wordPos, datasets, tagIds):
    for i in range(len(tagIds)):
        for dataset in datasets:
            if dataset.name == wordPos[i][0]:
                tagIds[i] = dataset.id2tag[tagIds[i]]


def centroid(infos, wordIdList, columnDict, id2tag, info_indexes, word, pos, tsne_i_0, tsne_i_1):
    x_ = [infos[infoIndex][columnDict[tsne_i_0]] for infoIndex in info_indexes]
    y_ = [infos[infoIndex][columnDict[tsne_i_1]] for infoIndex in info_indexes]
    x = y = occurenceNumber = 0
    for i, infoIndex in enumerate(info_indexes):
        if (word == wordIdList[infos[infoIndex][columnDict['id_word']]] and
            pos == id2tag[(dataset, infos[infoIndex][columnDict['gold_tag']])]):
            x += x_[i]
            y += y_[i]
            occurenceNumber += 1
    x /= occurenceNumber
    y /= occurenceNumber
    centroid = x, y
    return centroid, occurenceNumber
