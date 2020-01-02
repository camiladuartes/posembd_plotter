from tsne_pos.io import readInfoFile, readVocabFile, readTagsFile
import argparse

def createProloFile(infos, columnDict, wordIdList, id2tag, outputPath):
    with open(outputPath, "w") as f:
        f.write("dataset;id_sent;palavra;gold_tag\n")
        for info in infos:
            dataset = info[columnDict['dataset']]
            idSent = info[columnDict['id_sent']]
            idWord = info[columnDict['id_word']]
            tagId = info[columnDict['gold_tag']]

            word = wordIdList[int(idWord)]
            tag = id2tag[(dataset, int(tagId))]

            f.write("{};{};{};{}\n".format(dataset, idSent, word, tag.strip('\n')))



parser = argparse.ArgumentParser()
parser.add_argument("infosPath", help="path of infos csv file")
parser.add_argument("vocabPath", help="path of vocab csv file")
parser.add_argument("tagsPath", help="path of tags csv file")
parser.add_argument("outputPath", help="path of output file")
args = parser.parse_args()
infosPath = args.infosPath
tagsPath = args.tagsPath
outputPath = args.outputPath
vocabPath = args.vocabPath

infos, columnDict = readInfoFile(infosPath)
wordIdList, vocab = readVocabFile(vocabPath)
id2tag, _ = readTagsFile(tagsPath)

createProloFile(infos, columnDict, wordIdList, id2tag, outputPath)
