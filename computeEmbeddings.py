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
DATASETS_FOLDER: path to the folder with your dataset.

Fill in the datasets infos following the structure
DATASETS = [
    ('DATSET1_NAME', ('DATASET1_TRAIN_FILE', USE_TRAIN), ('DATASET1_VAL_FILE', USE_VAL),  'DATASET1_TEST_FILE'),
    ('DATSET2_NAME', ('DATASET2_TRAIN_FILE', USE_TRAIN), ('DATASET2_VAL_FILE', USE_VAL),  'DATASET2_TEST_FILE'),
    ...
    ('DATSETN_NAME', ('DATASETN_TRAIN_FILE', USE_TRAIN), ('DATASETN_VAL_FILE', USE_VAL),  'DATASETN_TEST_FILE'),
]

'''

DATASETS_FOLDER = 'data/'

DATASETS = [
    {'name': 'Macmorpho', 'trainFile': 'macmorpho-train.mm.txt', 'useTrain': True, 'valFile': 'macmorpho-dev.mm.txt',
        'useVal': True, 'testFile': 'macmorpho-test.mm.txt'},
    {'name': 'Bosque', 'trainFile': 'pt_bosque-ud-train.mm.txt', 'useTrain': True, 'valFile': 'pt_bosque-ud-dev.mm.txt',
        'useVal': True, 'testFile': 'pt_bosque-ud-test.mm.txt'},
    {'name': 'GSD', 'trainFile': 'pt_gsd-ud-train.mm.txt', 'useTrain': True, 'valFile': 'pt_gsd-ud-dev.mm.txt',
        'useVal': True, 'testFile': 'pt_gsd-ud-test.mm.txt'},
    {'name': 'Linguateca', 'trainFile': 'lgtc-train.mm.txt', 'useTrain': True, 'valFile': 'lgtc-dev.mm.txt',
        'useVal': True, 'testFile': 'lgtc-test.mm.txt'}
]

MODEL_PATH = 'postag_sdict_WED_350_CED_70_BS_150.pt'

def retrieveModelHiperparams(modelPath):
    modelsHiperpars = modelPath.split('_')
    return {
        'WED' : modelsHiperpars[3],
        'CED' : modelsHiperpars[5],
        'BS' : modelsHiperpars[7]
    }


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

def loadModelAndDatasets(device):
    # loading datasets from datasets folder
    datasetsPreparer = DatasetsPreparer(DATASETS_FOLDER)
    datasets = datasetsPreparer.prepare(DATASETS)

    # retrieving tags and char dicts
    tagsFromDatasets = [(dataset.name, dataset.id2tag) for dataset in datasets]
    char2id, id2char = datasetsPreparer.getDicts()

    hiperparams = retrieveModelHiperparams(MODEL_PATH)
    posModel = createPOSModel(hiperparams['CED'], hiperparams['WED'], char2id, hiperparams['BS'], datasets)
    posModel.to(device)
    posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    posModel.eval()

    return datasets, posModel, id2char, tagsFromDatasets

def retrieveLists(datasets, posModel, id2char):
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


def computeEmbeddings(vocabPath, infosPicklePath, tagsFilePath):

    # pytorch logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(threshold=10000)
    torch.no_grad()

    datasets, posModel, id2char, tagsFromDatasets = loadModelAndDatasets(device)

    embeddings = {rep: [] for rep in EMBEDDINGS_PICKLE_PATH}

    words, predTags, goldTags, wordSentId, datasetNames = retrieveLists(datasets, posModel, id2char)


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


##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("vocabPath", help="path of vocab file")
parser.add_argument("infosPicklePath", help="path of info pickle file")
parser.add_argument("tagsFilePath", help="path of tags file")
args = parser.parse_args()

computeEmbeddings(args.vocabPath, args.infosPicklePath, args.tagsFilePath)
