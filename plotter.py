from tsne_pos.io import readInfoFile, readVocabFile


def plotter(infos, vocab):
    while True:
        print('Query. Separate different queries with ' '. Use * for all possible entries.')
        s = input()
        queries = s.split(' ')

        for query in queries:
            word, pos = query.split('/')

            if word == '*':
                pass
            else:
                pass

            if pos == '*':
                pass
            else:
                pass



params = sys.argv[1:]
infosPath = params[0]
vocabPath = params[1]

infos = readInfoFile(infosPath)
vocab = readVocabFile(vocabPath)


plotter(infos, vocab)
