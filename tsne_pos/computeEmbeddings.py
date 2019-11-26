from posembd.base import get_batches
from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel
from posembd.io import sendOutput

from tsne_pos.utils import convertToText, createVocab
from tsne_pos.parameters import *


import torch

'''
Funcao responsavel por recuperar:
    - palavra
    - representacoes geradas
    - tags corretas e estimadas
    - posicao do token na sentenca
    - posicao da sentenca no dataset
'''

def computeEmbeddings():
    device = torch.device("cuda" if torch.cudm a.is_available() else "cpu")

    torch.set_printoptions(threshold=10000)
    torch.no_grad()

    datasetsPreparer = DatasetsPreparer(DATASETS_FOLDER)
    datasets = datasetsPreparer.prepare(DATASETS)
    char2id, id2char = datasetsPreparer.getDicts()

    posModel = createPOSModel(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id, BILSTM_SIZE, datasets)
    posModel.to(device)
    posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Seting device

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
        saveToPickle(rep + '.pickle', embd)
