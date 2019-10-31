from sklearn.manifold import TSNE
from tsne_pos.utils import get_batches, convertToText
import pickle
from tsne_pos.parameters import TRAIN_EMBEDDINGS, EMBEDDINGS_PATH
import torch

def train_tsnes(device, model, datasets, id2char):

    model.eval()
    for rep, to_train  in TRAIN_EMBEDDINGS.items():
        if to_train == False:
            continue

        embeddings = []
        words = []
        pred_tags = []
        gold_tags = []
        for itr in get_batches(datasets, "train"):
            # Getting vars
            inputs, targets, _ = itr

            sent_gold_tags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]
            sent_words = [word_tensor.data.cpu().numpy() for word_tensor in inputs[0]]
            sent_words = [[id2char[c] for c in word] for word in sent_words]

            # Setting the input and the target (seding to GPU if needed)
            inputs = [[word.to(device) for word in sample] for sample in inputs]

            # Feeding the model
            output = model(inputs)
            _, pred = torch.max(output['Macmorpho'], 2)
            pred = pred.view(1, -1)

            sent_embeddings = [embd for embd in output[rep].data.cpu().numpy()[0]]
            sent_pred_tags = [tag.data.cpu().numpy().item(0) for tag in targets[0]]

            embeddings += sent_embeddings
            words += sent_words
            gold_tags += sent_gold_tags
            pred_tags += sent_pred_tags

            del output, inputs, _, sent_words, sent_embeddings
            torch.cuda.empty_cache()

        tsne = TSNE()

        for i in range(len(embeddings)):
            embeddings[i] = tuple(embeddings[i].tolist())

        print("ok1")
        unique_embeddings = None

        if rep == 'embeddings1':
            unique_embeddings = list(set(embeddings))
        else:
            unique_embeddings = embeddings


        t_embeddings = tsne.fit_transform(unique_embeddings)
        print("ok2")

        for i in range(len(t_embeddings)):
            t_embeddings[i] = tuple(t_embeddings[i].tolist())

        print("ok3")
        # montando o dicionario
        emb2tsne = dict(zip(unique_embeddings, t_embeddings))

        convertToText(words)

        tsne2word = {}
        for i in range(len(words)):
            tsne2word[emb2tsne[embeddings[i]]] = (words[i], gold_tags[i], pred_tags[i])

        path_emb2tsne = "emb2tsne_" + EMBEDDINGS_PATH[rep]
        path_tsne2word = "tsen2word_" + EMBEDDINGS_PATH[rep]

        try:
            pickle_out = open(path_emb2tsne, "wb")
            pickle.dump(emb2tsne, pickle_out)
            pickle_out.close()
            pickle_out = open(path_tsne2word, "wb")
            pickle.dump(tsne2word, pickle_out)
            pickle_out.close()
        except:
            print("Wasn't able to save to pickle file")

        del embeddings, t_embeddings

def load_tsnes():
    rep2dicts = dict()

    for rep, path in EMBEDDINGS_PATH.items():
        emb2tsne, tsne2word = None, None

        path_emb2tsne = "emb2tsne_" + path
        path_tsne2word = "tsen2word_" + path

        try:
            pickle_in = open(path_emb2tsne, "rb")
            emb2tsne = pickle.load(pickle_in)
            pickle_in.close()
            pickle_in = open(path_tsne2word, "rb")
            tsne2word = pickle.load(pickle_in)
            pickle_in.close()

            rep2dicts[rep] = (emb2tsne, tsne2word)
        except:
            print("Wasn't able to load pickle file for {}".format(rep))

    return rep2dicts
