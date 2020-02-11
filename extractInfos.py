import sys
import argparse

import torch

from posembd.base import get_batches
from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel

from tsne_pos.utils import convertToText, createVocab, convertToTagNames
from tsne_pos.globals import EMBEDDINGS_PICKLE_PATH
from tsne_pos.io import saveToPickle





# Path to folder with datasets
DATASETS_DIR = 'data/'

# Fill in the datasets infos following the structure
# DATASETS = [
#     ('DATSET1_NAME', ('DATASET1_TRAIN_FILE', USE_TRAIN), ('DATASET1_VAL_FILE', USE_VAL),  'DATASET1_TEST_FILE'),
#     ('DATSET2_NAME', ('DATASET2_TRAIN_FILE', USE_TRAIN), ('DATASET2_VAL_FILE', USE_VAL),  'DATASET2_TEST_FILE'),
#     ...
#     ('DATSETN_NAME', ('DATASETN_TRAIN_FILE', USE_TRAIN), ('DATASETN_VAL_FILE', USE_VAL),  'DATASETN_TEST_FILE'),
# ]
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

# Path to trained posembd model
MODEL_PATH = 'postag_sdict_WED_350_CED_70_BS_150.pt'

# Function for retrieving models hiperparameters from its name
def retrieveModelHiperparams(modelPath):
    modelsHiperpars = modelPath.split('_')
    return {
        'WED' : modelsHiperpars[3],
        'CED' : modelsHiperpars[5],
        'BS' : modelsHiperpars[7]
    }

# Function for writing vocab file
def writeVocabFile(vocabFilePath, vocabDict):
    # Header:
    # id_word word
    with open(vocabFilePath, "w") as f:
        f.write("id_word;word\n")
        for word, index in vocabDict.items():
            f.write("{};{}\n".format(index, word))


# Function for writing tags file
def writeTagsFile(tagsFilePath, tagsFromDatasets):
    # Header:
    # dataset id_tag tag
    with open(tagsFilePath, "w") as f:
        f.write("dataset;id_tag;tag\n")
        for dataset, tagList in tagsFromDatasets:
            for index, tag in enumerate(tagList):
                f.write("{};{};{}\n".format(dataset, index, tag))



# Function for loading model and datasets from given paths
def loadModelAndDatasets(device):

    # Loading datasets from datasets folder
    datasetsPreparer = DatasetsPreparer(DATASETS_DIR)
    datasets = datasetsPreparer.prepare(DATASETS)

    # Retrieving tags and char dicts
    tagsFromDatasets = [(dataset.name, dataset.id2tag) for dataset in datasets]
    char2id, id2char = datasetsPreparer.getDicts()

    # Loading and preparing model
    hiperparams = retrieveModelHiperparams(MODEL_PATH)
    posModel = createPOSModel(hiperparams['CED'], hiperparams['WED'], char2id, hiperparams['BS'], datasets)
    posModel.to(device)
    posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    posModel.eval()

    return datasets, posModel, id2char, tagsFromDatasets

# Function for retrieving infos lists using the model and datasets
def retrieveLists(datasets, posModel, id2char):

    # Initializing vars
    words, predTags, goldTags, wordPosInfos = [], [], [], []
    embeddings = [[] for _ in range(4)]
    sentId = 0

    for itr in get_batches(datasets, "train"):

        # Getting vars
        inputs, targets, datasetName = itr

        # Retrieving gold (labels) tags
        goldTags += [tag.data.cpu().numpy().item(0) for tag in targets[0]]

        # Retrieving words from input
        sentWords = [wordTensor.data.cpu().numpy() for wordTensor in inputs[0]]
        words += [[id2char[c] for c in word] for word in sentWords]

        # Setting the input and the target (sending to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]

        # Feeding the model and getting output
        output = posModel(inputs)
        _, pred = torch.max(output['Macmorpho'], 2)
        pred = pred.view(1, -1)

        # Retrieving predicted labels
        predTags += [tag.data.cpu().numpy().item(0) for tag in pred[0]]

        # Retrieving computed embeddings
        for i in range(4):
            embeddings[i] += [embd for embd in output["embeddings{}".format(i)].data.cpu().numpy()[0]]

        # Retrieving dataset names and word/sent pos
        wordPosInfos.append([(datasetName, len(inputs))])

        # Updating sentence id
        sentId += 1

        # Memory saving
        torch.cuda.empty_cache()

    # Word position is
    wordPos = [[(wordPosInfo[0], i, j) for j in range(wordPosInfo[1])]
            for i, wordPosInfo in enumerate(wordPosInfos)]

    return words, predTags, goldTags, wordPos, embeddings


# Main function
def extractInfos(vocabPath, infosPicklePath, tagsFilePath, embeddingsPath):

    # Pytorch logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(threshold=10000)
    torch.no_grad()

    # Loading model and datasets
    datasets, posModel, id2char, tagsFromDatasets = loadModelAndDatasets(device)

    # Retrieving lists of infos
    words, predTags, goldTags, wordPos, embeddings = retrieveLists(datasets, posModel, id2char)

    # Converting ids to words
    convertToTagNames(datasetNames, datasets, goldTags)
    convertToTagNames(datasetNames, datasets, predTags)
    convertToText(words)

    # Creating vocabulary and token-word id list
    vocabDict, wordIdList = createVocab(words)

    # Writing output files
    writeVocabFile(vocabPath, vocabDict)
    writeTagsFile(tagsFilePath, tagsFromDatasets)
    saveToPickle(infosPicklePath, (wordPos, wordIdList, predTags, goldTags))
    for i in range(4):
        saveToPickle(embeddingsPath[i], embeddings[i])


##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("vocabPath", help="path of vocab file")
parser.add_argument("infosPicklePath", help="path of info pickle file")
parser.add_argument("tagsFilePath", help="path of tags file")
parser.add_argument("embeddingsDir", help="embeddings dir")
args = parser.parse_args()

# Creating directory if it does not exist
if not os.path.exists(args.embeddingsDir):
    os.mkdir(args.embeddingsDir)

# Updating output paths for embeddings
embeddingsPath = [os.path.join(args.embeddingsDir, path) for path in EMBEDDINGS_PICKLE_PATH]

extractInfos(args.vocabPath, args.infosPicklePath, args.tagsFilePath, embeddingsPath)
