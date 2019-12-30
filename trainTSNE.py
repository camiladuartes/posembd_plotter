import random
import sys
import numpy as np
from sklearn.manifold import TSNE
import time


from tsne_pos.io import saveToPickle, loadFromPickle



def trainTSNEs(inFile, outFile, rep, numEmbs=-1):
    timeStart = time.time()

    embds = loadFromPickle(inFile)

    if numEmbs == -1:
        numEmbs = len(embds)

    tsne = TSNE(verbose=2, n_jobs=-1)

    Tembeddings = None

    if rep == 'embeddings1':

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
        Tembeddings = tsne.fit_transform(embds[:numEmbs])


    for i in range(len(Tembeddings)):
        Tembeddings[i] = Tembeddings[i].tolist()

    saveToPickle(outFile, Tembeddings)

    timeEnd = time.time()

    print("[POS] TSNEs traning finished. Duration: {}".format(timeEnd - timeStart))


params = sys.argv[1:]
inFile = params[0]
outFile = params[1]
rep = params[2]
numEmbs = int(params[3])

# computeEmbeddings()
trainTSNEs(inFile, outFile, rep, numEmbs)
