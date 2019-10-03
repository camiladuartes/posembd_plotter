from math import sqrt

import datetime
import random
import sys

import numpy as np

import torch

from tsne_pos.utils import send_output
from tsne_pos.dataset import build_char_dict, load_datasets
from tsne_pos.parameters import *
from tsne_pos.tsne import train_tsnes

from tsne_pos.models.CharBILSTM import CharBILSTM
from tsne_pos.models.WordBILSTM import WordBILSTM
from tsne_pos.models.POSTagger import POSTagger


torch.set_printoptions(threshold=10000)

# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
#########################################################################################
#########                                                                    ############
#########                           PREPROCESSING                            ############
#########                                                                    ############
#########################################################################################
'''

# dataset building
datasets = load_datasets()

# builds char-id table
char2id, id2char = build_char_dict(datasets)

# converts text to id from chars
for dataset in datasets:
    dataset.prepare(char2id)

# prints the datasets details
for dataset in datasets:
    send_output(str(dataset), 1)

'''
#########################################################################################
#########                                                                    ############
#########                     DEFINING AND LOADING MODELS                    ############
#########                                                                    ############
#########################################################################################
'''

# building model
pos_model = POSTagger(CharBILSTM(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id),
                      WordBILSTM(WORD_EMBEDDING_DIM),
                      WordBILSTM(WORD_EMBEDDING_DIM),
                      BILSTM_SIZE, datasets)
pos_model.to(device)

# prints model
send_output(str(pos_model), 1)



# Loading the model with best loss on the validation
try:
    pos_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    send_output("Successfully loaded trained model", 1)
except:
    send_output("Was not able to load trained model\n", 0)
    exit()

'''
#########################################################################################
#########                                                                    ############
#########                          LOADING T-SNE                             ############
#########                                                                    ############
#########################################################################################
'''

train_tsnes(device, pos_model, datasets)