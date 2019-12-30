import random, sys
import tqdm
import pickle
from tsne_pos.parameters import *


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


def saveToPickle(filePath, obj):
    pickle_out = open(filePath, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def loadFromPickle(filePath):
    pickle_in = open(filePath, "rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()

    return obj

'''
returns: list of strings
'''
def readVocabFile(vocabFile):
    wordIdList = []
    vocabDict = {}
    with open(vocabFile, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            else:
                index, word = line.split(';', 1)
                index = int(index)
                word = word.strip()
                wordIdList.append(word)
                vocabDict[word] = index

    return wordIdList, vocabDict

'''
returns list of lists + columnDict

list of lists:
    info[i]: line i
    info[i][j]: line i, column j

columnDict = dict with columns ids
'''
def readInfoFile(infosPath):
    info = []
    columnDict = None
    with open(infosPath, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: columnDict = dict({y.strip():x for x, y in enumerate(line.split(';'))})
            else:
                splittedLine = line.split(';')
                infos0 = [int(splittedLine[0])]
                infos1 = [splittedLine[1]]
                infos2 = [int(x) for x in splittedLine[2:7]]
                tsnes = [float(x) for x in splittedLine[7:]]
                info.append(infos0 + infos1 + infos2 + tsnes)

    return info, columnDict

'''
embedding_iFile:
# Embedding i File
id_token embedding_i
'''
def writeEmbeddingFile(rep, embeddings):
    with open(EMBEDDINGS_TXT_PATH[rep], "w") as f:
        f.write("id_token;{}\n".format(rep))
        for index, embd in enumerate(embeddings):
            f.write("{};{}\n".format(index, embd))

'''
returns: list of lists

lists of lists:
    embeddings[i]: embedding i
    embeddings[i][j]: embedding i, dim j
'''
def readEmbeddingFile(rep):
    embeddings = []
    with open(EMBEDDINGS_TXT_PATH[rep], "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            else: embeddings.append(line.split(';')[1])

    return embeddings
