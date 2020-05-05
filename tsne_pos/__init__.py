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


def centroid(word, pos, x_, y_):
    x = y = 0
    if isinstance(x_, list):
        for i in x_:
            x += i
        for j in y_:
            y += j
        occurenceNumber = len(x_)
        x /= occurenceNumber
        y /= occurenceNumber
    else:
        x = x_
        y = y_
    centroid = [(word, pos), [x, y]]
    return centroid
