'''
Script for training TSNEs of representation levels
Usage:
    python trainTSNE.py INPUT_PATH REPRESENTATION_LVL NUMBER_OF_EMBEDDINGS OUTPUT_PATH

    INPUT_PATH: path to pickle file containing embeddings computed at computeEmbeddings.py
    REPRESENTATION_LVL: 1, 2, 3 or 4, indicating the level of embedding to be trained
    NUMBER_OF_EMBEDDINGS: number of embeddings that will be trained at TSNEs algorithm. -1 to train all embeddings
    OUTPUT_PATH: path to pickle file that will contain the trained TSNEs
'''

import random
import sys
import numpy as np
from sklearn.manifold import TSNE
import time
import argparse

from tsne_pos.io import saveToPickle, loadFromPickle


def trainTSNEs(inFile, outFile, rep, numEmbs=-1):
    timeStart = time.time()

    embds = loadFromPickle(inFile) # loading trained embeddings

    # handling -1 entry
    if numEmbs == -1:
        numEmbs = len(embds)

    # instantiating TSNE algorithm
    tsne = TSNE(verbose=2, n_jobs=-1)

    Tembeddings = None

    if rep == '1':
        # the first representation is context unsensitive
        # therefore one word will have always the same representation

        embd2ids = {}
        uniqEmbds = []
        uniqEmbdsIndex = []
        for i, embd in enumerate(embds):
            hashableEmbd = tuple(embd.tolist())
            if hashableEmbd not in embd2ids:
                embd2ids[hashableEmbd] = []
                uniqEmbds.append(embd)
            embd2ids[hashableEmbd].append(i)


        TUniqEmbds = tsne.fit_transform(uniqEmbds)
        Tembeddings = [None for _ in range(len(embds))]

        for i, embd in enumerate(uniqEmbds):
            tsne = TUniqEmbds[i]
            hashableEmbd = tuple(embd.tolist())
            for id in embd2ids[hashableEmbd]:
                Tembeddings[id] = tsne
    else:
        # other representatioo levels are context sensitive
        Tembeddings = tsne.fit_transform(embds[:numEmbs])

    # needs
    for i in range(len(Tembeddings)):
        Tembeddings[i] = Tembeddings[i].tolist()

    saveToPickle(outFile, Tembeddings)

    timeEnd = time.time()

    print("[POS] TSNEs traning finished. Duration: {}".format(timeEnd - timeStart))


##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("inputPath", help="path to pickle file containing embeddings computed at computeEmbeddings.py")
parser.add_argument("representationLevel", help="1, 2, 3 or 4, indicating the level of embedding to be trained")
parser.add_argument("numberOfEmbeddings", help="number of embeddings that will be trained at TSNEs algorithm. -1 to train all embeddings")
parser.add_argument("outputPath", help="path to pickle file that will contain the trained TSNEs")
args = parser.parse_args()

repLvl = args.representationLevel
numEmbs = int(args.numberOfEmbeddings)

trainTSNEs(args.inputPath, args.outputPath, repLvl, numEmbs)
