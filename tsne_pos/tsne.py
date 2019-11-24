from sklearn.manifold import TSNE
from tsne_pos.utils import get_batches, convertToText, saveToPickle, loadFromPickle, createVocab
from posembd.base import get_batches
from tsne_pos.parameters import TRAIN_EMBEDDINGS, EMBEDDINGS_PATH
import torch

'''
Funcao responsavel por recuperar:
    - palavra
    - representacoes geradas
    - tags corretas e estimadas
    - posicao do token na sentenca
    - posicao da sentenca no dataset
'''
def computeEmbeddings(device, model, datasets, id2char):
    model.eval()

    embeddings = {rep: []
            for rep, to_train in TRAIN_EMBEDDINGS.items()
            if to_train == True}

    words, predTags, goldTags, wordSentId = [], [], [], []

    sentId = 0
    for itr in get_batches(datasets, "train"):
        # Getting vars
        inputs, targets, _ = itr

        sentGoldTags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]
        sentWords = [wordTensor.data.cpu().numpy() for wordTensor in inputs[0]]
        sentWords = [[id2char[c] for c in word] for word in sentWords]

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]

        # Feeding the model
        output = model(inputs)
        _, pred = torch.max(output['Macmorpho'], 2)
        pred = pred.view(1, -1)

        sentPredTags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]

        for rep in embeddings:
            embeddings[rep] += [embd for embd in output[rep].data.cpu().numpy()[0]]
        words += sentWords
        goldTags += sentGoldTags
        predTags += sentPredTags

        wordSentId += [(sentId, i) for i in range(len(sentGoldTags))]

        torch.cuda.empty_cache()

        sentId += 1

    convertToText(words)
    word2id, wordIdList = createVocab(words)
    saveToPickle('wpgi_temp.pickle', (wordIdList, predTags, goldTags, wordSentId))
    saveToPickle('vocabDict.pickle', word2id)

    for rep, embd in embeddings.items():
        print(rep)
        for i in range(len(embd)):
            embd[i] = embd[i].tolist()
        print(rep)
        for i in range(len(embd)):
            embd[i] = tuple(embd[i])
        print(rep)
        saveToPickle(rep + '_temp.pickle', embd)
        print(rep)


def trainTSNEs():
    embeddings = [rep for rep in TRAIN_EMBEDDINGS if TRAIN_EMBEDDINGS[rep]]

    wordIdList, _, _, _ = loadFromPickle('wpgi_temp.pickle')

    for rep in embeddins:
        embeddings = loadFromPickle(rep + '_temp.pickle')
        tsne = TSNE(verbose=1)

        Tembeddings = tsne.fit_transform(embeddings)

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
