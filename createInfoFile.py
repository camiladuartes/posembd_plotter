'''
recebe os arquivos de vocabulario
'''

'''
InfoFile:
# Info file
id_token dataset id_sent pos_sent id_word pred_tag gold_tag tsne_0_0 tnse_0_1 tsne_1_0 tsne_1_1 tsne_2_0 tsne_2_1 tsne_3_0 tsne_3_1
'''
def createInfoFile(infosPicklePath, tsnePicklePaths):
    wordPos, wordIdList, predTags, goldTags = loadFromPickle(infosPicklePath)
    tsnes = {'embedding{}'.format(i) : loadFromPickle(tsnePicklePaths['embedding{}'.format(i)])
                        for i in range(1, 5)}


    minTrainedsTSNEs = min(len(tsnes['embedding{}'.format(i)] for i in range(2, 5)))
    for i in range(2, 5):
        tsnes['embedding{}'.format(i)] = tsnes['embedding{}'.format(i)][:minTrainedsTSNEs]
    wordPos = wordPos[:minTrainedsTSNEs]
    wordIdList = wordIdList[:minTrainedsTSNEs]
    predTags = predTags[:minTrainedsTSNEs]
    goldTags = goldTags[:minTrainedsTSNEs]


    with open(infosPath, "w") as f:
        f.write("id_token;dataset;id_sent;pos_sent;id_word;pred_tag;gold_tag;tsne_0_0;tnse_0_1;tsne_1_0;tsne_1_1;tsne_2_0;tsne_2_1;tsne_3_0;tsne_3_1\n")
        for index in range(len(wordPos)):
            f.write("{};{};{};{};".format(index, tokenPos[0], tokenPos[1], tokenPos[2]))
            f.write("{};{};{};".format(wordIds[index], predTags[i], goldTags[i]))
            f.write("{};{};".format(tsnes["embeddings1"][index][0], tsnes["embeddings1"][index][1]))
            f.write("{};{};".format(tsnes["embeddings2"][index][0], tsnes["embeddings2"][index][1]))
            f.write("{};{};".format(tsnes["embeddings3"][index][0], tsnes["embeddings3"][index][1]))
            f.write("{};{}\n".format(tsnes["embeddings4"][index][0], tsnes["embeddings4"][index][1]))


params = sys.argv[1:]
infosPicklePath = params[0]

createInfoFile(infosPicklePath, TSNE_PICKLE_PATH)
