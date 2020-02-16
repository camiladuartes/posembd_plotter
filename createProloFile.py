'''
Script for creating Prolo's visualization file
Usage:
    python createProloFile.py INFOS_PATH VOCAB_PATH TAGS_PATH OUTPUT_PATH

    INFOS_PATH: path to info csv file created at createInfoFile.py
    VOCAB_PATH: path to vocab csv file created at computeEmbeddings.py
    TAGS_PATH: path to tags csv file created at computeEmbeddings.py
    OUTPUT_PATH: path where prolo csv file will be saved
'''

import argparse

from tsne_pos.io import readInfoFile, readVocabFile, readTagsFile

from tqdm import tqdm

def createProloFile(infos, columnDict, wordIdList, id2tag, outputPath):
    with open(outputPath, "w") as f:
        f.write("dataset;id_sent;palavra;gold_tag\n")
        for info in tqdm(infos):
            # for each token at info file, retrieve its dataset, sentence, word and tag id
            dataset = info[columnDict['dataset']]
            idSent = info[columnDict['id_sent']]
            idWord = info[columnDict['id_word']]
            tagId = info[columnDict['gold_tag']]

            # load the word and tag related to theirs ids
            word = wordIdList[int(idWord)]
            tag = id2tag[(dataset, int(tagId))]

            # write it on the file
            f.write("{};{};{};{}\n".format(dataset, idSent, word, tag.strip('\n')))


################################### HANDLING ARGS ##########################################

parser = argparse.ArgumentParser()
parser.add_argument("infosPath", help="path of infos csv file")
parser.add_argument("vocabPath", help="path of vocab csv file")
parser.add_argument("tagsPath", help="path of tags csv file")
parser.add_argument("outputPath", help="path of output file")
args = parser.parse_args()


## LOADING FILES
infos, columnDict = readInfoFile(args.infosPath)
wordIdList, vocab = readVocabFile(args.vocabPath)
id2tag, _ = readTagsFile(args.tagsPath)

createProloFile(infos, columnDict, wordIdList, id2tag, args.outputPath)
