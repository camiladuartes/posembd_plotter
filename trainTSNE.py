'''
Script for training TSNEs of representation levels
Usage:
    python trainTSNE.py INPUT_DIR REPRESENTATION_LVL NUMBER_OF_EMBEDDINGS OUTPUT_DIR

    INPUT_DIR: path to directory containing pickle embeddings computed at computeEmbeddings.py
    REPRESENTATION_LVL: 1, 2, 3 or 4, indicating the level of embedding to be trained
    NUMBER_OF_EMBEDDINGS: number of embeddings that will be trained at TSNEs algorithm. -1 to train all embeddings
    OUTPUT_DIR: path to directory that will contain the trained TSNEs
'''

import os
import sys
import time
import random
import argparse

import numpy as np
from sklearn.manifold import TSNE

from tsne_pos.io import saveToPickle, loadFromPickle
from tsne_pos.globals import EMBEDDINGS_PICKLE_PATH, TSNE_PICKLE_PATH


def trainTSNEs(inFile, outFile, rep, numEmbs=-1):
    timeStart = time.time()

    embds = loadFromPickle(inFile) # loading trained embeddings

    # handling -1 entry
    if numEmbs == -1:
        numEmbs = len(embds)

    # instantiating TSNE algorithm
    tsne = TSNE(verbose=2, n_jobs=-1)

    Tembeddings = None

    if rep == 0:
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

    for i in range(len(Tembeddings)):
        Tembeddings[i] = Tembeddings[i].tolist()

    saveToPickle(outFile, Tembeddings)

    timeEnd = time.time()

    print("[POS] TSNEs traning finished. Duration: {}".format(timeEnd - timeStart))


##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("inputDir", help="path to directory containing pickle embeddings computed at computeEmbeddings.py")
parser.add_argument("representationLevel", help="0, 1, 2 or 3, indicating the level of embedding to be trained")
parser.add_argument("numberOfEmbeddings", help="number of embeddings that will be trained at TSNEs algorithm. -1 to train all embeddings")
parser.add_argument("outputDir", help="path to directory that will contain the trained TSNEs")
args = parser.parse_args()

# Converting from string to ints
repLvl = int(args.representationLevel)
numEmbs = int(args.numberOfEmbeddings)

# Joining input directory to global embeddings filenames
inputPath = os.path.join(args.inputDir, EMBEDDINGS_PICKLE_PATH[repLvl])

# Creating output directory if it does not exist
if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)

# Joining output directory to global embeddings filenames
outputPath = os.path.join(args.outputDir, TSNE_PICKLE_PATH[repLvl])

trainTSNEs(inputPath, outputPath, repLvl, numEmbs)
