import random, sys
import tqdm
from tsne_pos.parameters import *
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
def writeVocabFile(wordIdList):
    with open(VOCAB_FILE, "w") as f:
        f.write("id_word;word\n")
        for index, word in enumerate(wordIdList):
            f.write("{};{}\n".format(index, word))


'''
returns: list of strings
'''
def readVocabFile():
    wordIdList = []
    with open(VOCAB_FILE, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            else: wordIdList.append(line.split(';')[1])

    return wordIdList


'''
InfoFile:
# Info file
id_token dataset id_sent pos_sent id_word pred_tag gold_tag tsne_0_0 tnse_0_1 tsne_1_0 tsne_1_1 tsne_2_0 tsne_2_1 tsne_3_0 tsne_3_1
'''
def writeInfoFile(tokenPos, wordIds, predTags, goldTags, tsnes):
    with open(INFOS_PATH, "w") as f:
        f.write("id_token;dataset;id_sent;pos_sent;id_word;pred_tag;gold_tag;tsne_0_0;tnse_0_1;tsne_1_0;tsne_1_1;tsne_2_0;tsne_2_1;tsne_3_0;tsne_3_1\n")
        for index in range(len(wordIds)):
            f.write("{};{};{};{};".format(index, tokenPos[0], tokenPos[1], tokenPos[2]))
            f.write("{};{};{};".format(wordIds[index], predTags[i], goldTags[i]))
            f.write("{};{};".format(tsnes["embeddings1"][0], tsnes["embeddings1"][1]))
            f.write("{};{};".format(tsnes["embeddings2"][0], tsnes["embeddings2"][1]))
            f.write("{};{};".format(tsnes["embeddings3"][0], tsnes["embeddings3"][1]))
            f.write("{};{}\n".format(tsnes["embeddings4"][0], tsnes["embeddings4"][1]))

'''
returns list of lists + columnDict

list of lists:
    info[i]: line i
    info[i][j]: line i, column j

columnDict = dict with columns ids
'''
def readInfoFile():
    info = []
    columnDict = None
    with open(INFOS_PATH, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: columnDict = dict(enumerate(f.split(';')))
            else: info.append(line.split(';'))

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
