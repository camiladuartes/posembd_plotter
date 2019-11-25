from math import sqrt

import datetime
import random
import sys

from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel
from posembd.io import sendOutput


import numpy as np

import torch

from tsne_pos.parameters import *
from tsne_pos.tsne import trainTSNEs, loadTSNEs, computeEmbeddings
from tsne_pos.visualize import plot



torch.set_printoptions(threshold=10000)
torch.no_grad()

# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
#########################################################################################
#########                                                                    ############
#########                           PREPROCESSING                            ############
#########                                                                    ############
#########################################################################################
'''

datasetsPreparer = DatasetsPreparer(DATASETS_FOLDER)
datasets = datasetsPreparer.prepare(DATASETS)
char2id, id2char = datasetsPreparer.getDicts()

# dataset building
# datasets = load_datasets()
#
# # builds char-id table
# char2id, id2char = build_char_dict(datasets)
#
# # converts text to id from chars
# for dataset in datasets:
#     dataset.prepare(char2id)
#
# # prints the datasets details
# for dataset in datasets:
#     send_output(str(dataset), 1)

'''
#########################################################################################
#########                                                                    ############
#########                     DEFINING AND LOADING MODELS                    ############
#########                                                                    ############
#########################################################################################
'''

# building model
# posModel = POSTagger(CharBILSTM(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id),
#                       WordBILSTM(WORD_EMBEDDING_DIM),
#                       WordBILSTM(WORD_EMBEDDING_DIM),
#                       BILSTM_SIZE, datasets)
posModel = createPOSModel(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id, BILSTM_SIZE, datasets)
posModel.to(device)

# prints model
sendOutput(str(posModel), 1)



# Loading the model with best loss on the validation
# try:
posModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))
sendOutput("Successfully loaded trained model", 1)
# except:
#     send_output("Was not able to load trained model\n", 0)
#     exit()

'''
#########################################################################################
#########                                                                    ############
#########                          LOADING T-SNE                             ############
#########                                                                    ############
#########################################################################################
'''

computeEmbeddings(device, posModel, datasets, id2char)
trainTSNEs()

del posModel, datasets, id2char

'''
#########################################################################################
#########                                                                    ############
#########                            PLOTTING                                ############
#########                                                                    ############
#########################################################################################
'''

# plot(rep2dicts)
# printToFile(rep2dicts['embeddings1'][1])
