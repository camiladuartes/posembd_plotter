from sklearn.manifold import TSNE
from tsne_pos.utils import get_batches
import pickle
from tsne_pos.parameters import TRAIN_EMBEDDINGS, EMBEDDINGS_PATH
import torch

def train_tsnes(device, model, datasets):

    model.eval()
    for rep, to_train  in TRAIN_EMBEDDINGS.items():
        if to_train == False:
            continue

        sent_embeddings = []
        for itr in get_batches(datasets, "train"):

            # Getting vars
            inputs, _, _ = itr

            # Setting the input and the target (seding to GPU if needed)
            inputs = [[word.to(device) for word in sample] for sample in inputs]

            # Feeding the model

            sent_embeddings.append(model(inputs)[rep].data.cpu().numpy()[0])
            # print(model(inputs)[rep])
            # print(model(inputs)[rep].cpu())

            del model(inputs)[rep], inputs, _
            torch.cuda.empty_cache()

        tsne = TSNE()

        lembeddings = [emdb for sent in sent_embeddings for emdb in sent]
        del sent_embeddings

        t_embeddings = tsne.fit_transform(lembeddings)


        lembeddings = [str(e) for e in lembeddings]
        # montando o dicionario
        d = dict(zip(lembeddings, t_embeddings))

        try:
            pickle_out = open(EMBEDDINGS_PATH[rep], "wb")
            pickle.dump(d, pickle_out)
            pickle_out.close()
        except:
            print("Wasn't able to save to pickle file")

        del lembeddings, t_embeddings

def load_tsne(rep):
    d = None
    try:
        pickle_in = open(EMBEDDINGS_PATH[rep], "rb")
        d = pickle.load(pickle_in)
        pickle_in.close()
    except:
        print("Wasn't able to load pickle file")
    return d
