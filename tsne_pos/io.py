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

'''
VocabFile:
# Vocab file
id_word word
'''
def writeVocabFile():
    pass

'''
returns: list of strings
'''
def readVocabFile():
    pass

'''
InfoFile:
# Info file
id_token dataset sent_id pos_sent id_word pred_tag gold_tag tsne_dim_0 tsne_dim_1
'''
def writeInfoFile():
    pass

'''
returns list of lists + columnDict

list of lists:
    info[i]: line i
    info[i][j]: line i, column j

columnDict = {
    'id_token': 0,
    'dataset': 1,
    'sent_id': 2,
    'pos_sent': 3,
    'id_word': 4,
    'pred_tag': 5,
    'gold_tag': 6,
    'tsne_dim_0': 7,
    'tsne_dim_1': 8
}
'''
def readInfoFile():
    pass

'''
embedding_iFile:
# Embedding i File
id_token embedding_i
'''
def writeEmbeddingFile():
    pass

'''
returns: list of lists

lists of lists:
    embeddings[i]: embedding i
    embeddings[i][j]: embedding i, dim j
'''
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
