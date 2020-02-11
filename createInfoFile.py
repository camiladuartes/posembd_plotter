'''
Script for creating info file
Usage:
    python createInfoFile.py INFOS_PICKLE_PATH OUTPUT_PATH

    INFOS_PICKLE_PATH: path to pickle file where the info object is saved
    OUTPUT_PATH: path where info csv file will be saved
'''

import sys
from tsne_pos.parameters import *
from tsne_pos.io import loadFromPickle
import argparse


'''
InfoFile:
# Info file
id_token dataset id_sent pos_sent id_word pred_tag gold_tag tsne_0_0 tnse_0_1 tsne_1_0 tsne_1_1 tsne_2_0 tsne_2_1 tsne_3_0 tsne_3_1
'''
def createInfoFile(infosPicklePath, infosPath, tsnePicklePaths):
    # Loading info object from pickle file
    tokenPos, wordIds, predTags, goldTags = loadFromPickle(infosPicklePath)

    # dict for accessing trained tsnes
    tsnes = {i : loadFromPickle(tsnePicklePaths['embeddings{}'.format(i)]) for i in range(1, 5)}

    # finding minimum size of trained tsnes files
    minLengthTrainedTSNEs = min([len(tsnes[i]) for i in range(2, 5)])

    # truncating objects to min length found
    for i in range(2, 5):
        tsnes[i] = tsnes[i][:minLengthTrainedTSNEs]
    tokenPos = tokenPos[:minLengthTrainedTSNEs]
    wordIds = wordIds[:minLengthTrainedTSNEs]
    predTags = predTags[:minLengthTrainedTSNEs]
    goldTags = goldTags[:minLengthTrainedTSNEs]


    with open(infosPath, "w") as f:
        # header of info file
        f.write("id_token;dataset;id_sent;pos_sent;id_word;pred_tag;gold_tag;tsne_0_0;tsne_0_1;tsne_1_0;tsne_1_1;tsne_2_0;tsne_2_1;tsne_3_0;tsne_3_1\n")
        for index in range(len(tokenPos)):
            # writing index token pos attributes
            f.write("{};{};{};{};".format(index, tokenPos[index][0], tokenPos[index][1], tokenPos[index][2]))

            # word and tags ids
            f.write("{};{};{};".format(wordIds[index], predTags[index], goldTags[index]))

            # writing tsnes
            f.write("{};{};".format(tsnes[1][index][0], tsnes[1][index][1]))
            f.write("{};{};".format(tsnes[2][index][0], tsnes[2][index][1]))
            f.write("{};{};".format(tsnes[3][index][0], tsnes[3][index][1]))
            f.write("{};{}\n".format(tsnes[4][index][0], tsnes[4][index][1]))

##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("infosPicklePath", help="path of infos pickle file")
parser.add_argument("outputFile", help="path to csv")
args = parser.parse_args()

createInfoFile(args.infosPicklePath, args.outputFile, TSNE_PICKLE_PATH)
