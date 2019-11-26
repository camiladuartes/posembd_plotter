import random, sys
import tqdm
from tsne_pos.parameters import LOG_LVL, OUTPUT_PATH
import pickle


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


def writeVocabFile():
    pass


def readVocabFile():
    pass


def writeInfoFile():
    pass


def readInfoFile():
    pass


def writeEmbeddingFile():
    pass


def readEmbeddingFile():
    pass


def printToFile(tsne2word):

    file1 = open("embeddings", "w")
    file2 = open("embeddings_meta", "w")

    for tsne, vals in tsne2word.items():
        file1.write("{}\t{}\n".format(tsne[0], tsne[1]))
        file2.write("{}\t{}\t{}\n".format(vals[0], vals[1], vals[2]))

    file1.close()
    file2.close()
