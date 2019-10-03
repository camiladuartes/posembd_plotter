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

        embeddings = []
        for itr in get_batches(datasets, "train"):

            # Getting vars
            inputs, _, _ = itr

            # Setting the input and the target (seding to GPU if needed)
            inputs = [[word.to(device) for word in sample] for sample in inputs]

            # Feeding the model
            embeddings.append(model(inputs)[rep].cpu().clone())
            # print(model(inputs)[rep].cpu())

            del model(inputs)[rep], inputs, _
            torch.cuda.empty_cache()

        tsne = TSNE()
        t_embeddings = tsne.fit_transform(embeddings)

        # montando o dicionario
        d = dict(zip(embeddings, t_embeddings))

        try:
            pickle_out = open(EMBEDDINGS_PATH[rep], "wb")
            pickle.dump(d, pickle_out)
            pickle_out.close()
        except:
            print("Wasn't able to save to pickle file")

        del embeddings, t_embeddings

def load_tsne(path, rep):
    d = None
    try:
        pickle_in = open(path[rep], "rb")
        d = pickle.load(pickle_in)
        pickle_in.close()
    except:
        print("Wasn't able to load pickle file")
    return d
