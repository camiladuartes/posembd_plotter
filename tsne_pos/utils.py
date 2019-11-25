import random, sys
import tqdm
from tsne_pos.parameters import LOG_LVL, OUTPUT_PATH, DATASETS_FOLDER, DATASETS
import pickle

def saveToPickle(filePath, obj):
    pickle_out = open(filePath, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def loadFromPickle(filePath):
    pickle_in = open(filePath, "rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()

    return obj

def printToFile(tsne2word):

    file1 = open("embeddings", "w")
    file2 = open("embeddings_meta", "w")

    for tsne, vals in tsne2word.items():
        file1.write("{}\t{}\n".format(tsne[0], tsne[1]))
        file2.write("{}\t{}\t{}\n".format(vals[0], vals[1], vals[2]))

    file1.close()
    file2.close()


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


def send_output(str, log_level):
    if log_level <= LOG_LVL:
        print(str)
    try:
        file = open(OUTPUT_PATH, "a")
        file.write(str + "\n")
        file.close()
    except:
        if log_level <= LOG_LVL:
            print("Was not able to open and write on output file")
