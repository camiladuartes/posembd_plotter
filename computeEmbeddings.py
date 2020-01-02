from posembd.base import get_batches
from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel
from posembd.io import sendOutput

from tsne_pos.utils import convertToText, createVocab, convertToTagNames
from tsne_pos.parameters import *
from tsne_pos.io import saveToPickle


import torch

import sys

import argparse

'''
VocabFile:
# Vocab file
id_word word
'''
def writeVocabFile(vocabFilePath, vocabDict):
    with open(vocabFilePath, "w") as f:
        f.write("id_word;word\n")
        for word, index in vocabDict.items():
            f.write("{};{}\n".format(index, word))


'''
TagsFile:
# Tags file
dataset id_tag tag
'''
def writeTagsFile(tagsFilePath, tagsFromDatasets):
    with open(tagsFilePath, "w") as f:
        f.write("dataset;id_tag;tag\n")
        for dataset, tagList in tagsFromDatasets:
            for index, tag in enumerate(tagList):
                f.write("{};{};{}\n".format(dataset, index, tag))



'''
Funcao responsavel por recuperar:
    - palavra
    - representacoes geradas
    - tags corretas e estimadas
    - posicao do token na sentenca
    - posicao da sentenca no dataset
'''

'''
parametros utiliazados
    DATASETS_FOLDER: pasta dos datasets
    DATASETS: infos dos datasets
    CHAR_EMBEDDING_DIM: tamanho do char embedding
    WORD_EMBEDDING_DIM: tamanho do word embedding
    BILSTM_SIZE: tamanho da bilstm
    MODEL_PATH: caminho para o modelo
    EMBEDDINGS_PATH: caminho onde os embeddings serao salvos
'''

def computeEmbeddings(vocabPath, infosPicklePath, tagsFilePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_printoptions(threshold=10000)
    torch.no_grad()

    datasetsPreparer = DatasetsPreparer(DATASETS_FOLDER)
    datasets = datasetsPreparer.prepare(DATASETS)
    tagsFromDatasets = [(dataset.name, dataset.id2tag) for dataset in datasets]
    char2id, id2char = datasetsPreparer.getDicts()

    writeTagsFile(tagsFilePath, tagsFromDatasets)
    exit()

    posModel = createPOSModel(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id, BILSTM_SIZE, datasets)
    posModel.to(device)
    posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    posModel.eval()

    embeddings = {rep: [] for rep in EMBEDDINGS_PICKLE_PATH}

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
    vocabDict, wordIdList = createVocab(words)

    wordPos = [(datasetNames[i], wordSentId[i][0], wordSentId[i][1])
                            for i in range(len(datasetNames))]

    writeVocabFile(vocabPath, vocabDict)
    writeTagsFile(tagsFilePath, tagsFromDatasets)
    saveToPickle(infosPicklePath, (wordPos, wordIdList, predTags, goldTags))

    for rep in embeddings:
        saveToPickle(EMBEDDINGS_PICKLE_PATH[rep], embeddings[rep])



parser = argparse.ArgumentParser()
parser.add_argument("vocabPath", help="path of vocab file")
parser.add_argument("infosPicklePath", help="path of info pickle file")
parser.add_argument("tagsFilePath", help="path of tags file")
args = parser.parse_args()
vocabPath = args.vocabPath
infosPicklePath = args.infosPicklePath
tagsFilePath = args.tagsFilePath
computeEmbeddings(vocabPath, infosPicklePath, tagsFilePath)
