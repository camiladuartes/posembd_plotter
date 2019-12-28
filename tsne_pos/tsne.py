from sklearn.manifold import TSNE

from tsne_pos.io import saveToPickle, loadFromPickle
from tsne_pos.parameters import EMBEDDINGS_PICKLE_PATH, TSNE_PICKLE_PATH

import time


def trainTSNEs(inFile, outFile, rep, numEmbs=-1):
    timeStart = time.time()

    embds = loadFromPickle(inFile)

    if numEmbs == -1:
        numEmbs = len(embds)

    tsne = TSNE(verbose=2, n_jobs=-1)

    Tembeddings = None

    if rep == 'embeddings1':

        embd2ids = {}
        for i, embd in enumerate(embds):
            if embd not in embd2ids:
                embd2ids = []
            embd2ids[embd].append(i)


        uniqEmbds = list(embd2id.keys())
        TUniqEmbds = tsne.fit_transform(uniqEmbds)
        Tembeddings = [None for _ in range(len(embds))]

        for i, embd in enumerate(uniqEmbds):
            tsne = TUniqEmbds[i]
            for id in embd2ids[embd]:
                Tembeddings[id] = tsne
    else:
        Tembeddings = tsne.fit_transform(embds[:numEmbs])


    for i in range(len(Tembeddings)):
        Tembeddings[i] = Tembeddings[i].tolist()

    saveToPickle(outFile, Tembeddings)

    timeEnd = time.time()

    print("[POS] TSNEs traning finished. Duration: {}".format(timeEnd - timeStart))


def saveFiles():
    pass
