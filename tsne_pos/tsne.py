from sklearn.manifold import TSNE

from tsne_pos.io import saveToPickle, loadFromPickle
from tsne_pos.parameters import TRAIN_EMBEDDINGS, EMBEDDINGS_PATH


def trainTSNEs():
    embeddings = [rep for rep in TRAIN_EMBEDDINGS if TRAIN_EMBEDDINGS[rep]]

    # wordIdList, _, _, _ = loadFromPickle('wpgi_temp.pickle')

    for rep in embeddings:
        embs = loadFromPickle(rep + '_temp.pickle')
        tsne = TSNE(verbose=1)

        Tembeddings = tsne.fit_transform(embs)

        if rep == 'embeddins1':
            id2embd = {}
            for i, tembd in enumerate(Tembeddings):
                if i not in id2embd:
                    id2embd[i] = Tembeddings[i]
                Tembeddings[i] = id2embd[i]

        for i in range(len(Tembeddings)):
            Tembeddings[i] = tuple(Tembeddings[i].tolist())

        tsnePath = "tsne_" + EMBEDDINGS_PATH[rep]

        saveToPickle(tsnePath, Tembeddings)


def loadTSNEs():
    rep2dicts = dict()

    for rep, path in EMBEDDINGS_PATH.items():
        embeddings = loadFromPickle("tsne_" + path)
        rep2dicts[rep] = embeddings

    return rep2dicts
