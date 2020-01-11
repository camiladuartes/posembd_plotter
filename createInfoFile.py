import sys
from tsne_pos.parameters import *
from tsne_pos.io import loadFromPickle
import argparse

'''
recebe os arquivos de vocabulario
'''

'''
InfoFile:
# Info file
id_token dataset id_sent pos_sent id_word pred_tag gold_tag tsne_0_0 tnse_0_1 tsne_1_0 tsne_1_1 tsne_2_0 tsne_2_1 tsne_3_0 tsne_3_1
'''
def createInfoFile(infosPicklePath, infosPath, tsnePicklePaths):
    tokenPos, wordIds, predTags, goldTags = loadFromPickle(infosPicklePath)
    tsnes = {'embeddings{}'.format(i) : loadFromPickle(tsnePicklePaths['embeddings{}'.format(i)])
                        for i in range(1, 5)}


    minTrainedsTSNEs = min([len(tsnes['embeddings{}'.format(i)]) for i in range(2, 5)])
    for i in range(2, 5):
        tsnes['embeddings{}'.format(i)] = tsnes['embeddings{}'.format(i)][:minTrainedsTSNEs]
    tokenPos = tokenPos[:minTrainedsTSNEs]
    wordIds = wordIds[:minTrainedsTSNEs]
    predTags = predTags[:minTrainedsTSNEs]
    goldTags = goldTags[:minTrainedsTSNEs]


    with open(infosPath, "w") as f:
        f.write("id_token;dataset;id_sent;pos_sent;id_word;pred_tag;gold_tag;tsne_0_0;tsne_0_1;tsne_1_0;tsne_1_1;tsne_2_0;tsne_2_1;tsne_3_0;tsne_3_1\n")
        for index in range(len(tokenPos)):
            f.write("{};{};{};{};".format(index, tokenPos[index][0], tokenPos[index][1], tokenPos[index][2]))
            f.write("{};{};{};".format(wordIds[index], predTags[index], goldTags[index]))
            f.write("{};{};".format(tsnes["embeddings1"][index][0], tsnes["embeddings1"][index][1]))
            f.write("{};{};".format(tsnes["embeddings2"][index][0], tsnes["embeddings2"][index][1]))
            f.write("{};{};".format(tsnes["embeddings3"][index][0], tsnes["embeddings3"][index][1]))
            f.write("{};{}\n".format(tsnes["embeddings4"][index][0], tsnes["embeddings4"][index][1]))


parser = argparse.ArgumentParser()
parser.add_argument("infosPicklePath", help="path of infos pickle file")
parser.add_argument("infosPath", help="path of info csv file")
args = parser.parse_args()
infosPicklePath = args.infosPicklePath
infosPath = args.infosPath

infos, columnDict = readInfoFile(infosPath)
wordIdList, vocab = readVocabFile(vocabPath)
id2tag, _ = readTagsFile(tagsPath)

createInfoFile(infosPicklePath, infosPath, TSNE_PICKLE_PATH)
