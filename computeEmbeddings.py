from posembd.base import get_batches
from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel
from posembd.io import sendOutput

from tsne_pos.utils import convertToText, createVocab, convertToTagNames


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_printoptions(threshold=10000)
    torch.no_grad()

    datasetsPreparer = DatasetsPreparer(DATASETS_FOLDER)
    datasets = datasetsPreparer.prepare(DATASETS)
    char2id, id2char = datasetsPreparer.getDicts()

    posModel = createPOSModel(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id, BILSTM_SIZE, datasets)
    posModel.to(device)
    posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    posModel.eval()

    embeddings = {rep: [] for rep in EMBEDDINGS_PATH}

    words, predTags, goldTags, wordSentId, datasetNames = [], [], [], [], []

    sentId = 0
    for itr in get_batches(datasets, "train"):
        # Getting vars
        inputs, targets, datasetName = itr

        sentGoldTags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]
        sentWords = [wordTensor.data.cpu().numpy() for wordTensor in inputs[0]]
        sentWords = [[id2char[c] for c in word] for word in sentWords]

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]

        # Feeding the model
        output = posModel(inputs)
        _, pred = torch.max(output['Macmorpho'], 2)
        pred = pred.view(1, -1)

        sentPredTags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]

        for rep in embeddings:
            embeddings[rep] += [embd for embd in output[rep].data.cpu().numpy()[0]]
        words += sentWords
        goldTags += sentGoldTags
        predTags += sentPredTags
        datasetNames += [datasetName for _ in range(len(sentPredTags))]

        wordSentId += [(sentId, i) for i in range(len(sentGoldTags))]

        torch.cuda.empty_cache()

        sentId += 1


    convertToTagNames(datasetNames, datasets, goldTags)
    convertToTagNames(datasetNames, datasets, predTags)
    convertToText(words)
    _, wordIdList = createVocab(words)

    wordPos = [(datasetNames[i], wordSentId[i][0], wordSentId[i][1])
                            for i in range(len(datasetNames))]

    saveToPickle(INFOS_PICKLE_PATH, (wordPos, wordIdList, predTags, goldTags))

    for rep in embeddings:
        saveToPickle(EMBEDDINGS_PATH[rep], embd)


computeEmbeddings()
