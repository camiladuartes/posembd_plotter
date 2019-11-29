from sklearn.manifold import TSNE

from tsne_pos.io import saveToPickle, loadFromPickle
from tsne_pos.parameters import EMBEDDINGS_PICKLE_PATH, TSNE_PICKLE_PATH


def trainTSNEs(inFile, outFile, rep):

    embds = loadFromPickle(inFile)
    tsne = TSNE(verbose=1)

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
        Tembeddings = tsne.fit_transform(embds)


    for i in range(len(Tembeddings)):
        Tembeddings[i] = Tembeddings[i].tolist()

    saveToPickle(outFile, Tembeddings)


def saveFiles():
    pass
