import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from tsne_pos.parameters import PLOT_FLAG


def visualize(title, emb2tsne, tsne2word, colors):

    tsnes = list(emb2tsne.values())

    x, y = list(zip(*tsnes))
    words = [tsne2word[(x[i], y[i])][0] for i in range(len(x))]
    gold_tags = [tsne2word[(x[i], y[i])][1] for i in range(len(x))]
    pred_tags = [tsne2word[(x[i], y[i])][2] for i in range(len(x))]

    fig, ax = plt.subplots(figsize=(100,100))
    ax.scatter(x, y, c=gold_tags, cmap=ListedColormap(colors), alpha=1)

    for i, _ in enumerate(x):
        ax.annotate(words[i], (x[i], y[i]))

    plt.title(title)
    plt.show()


def plot(rep2dicts):

    colors = np.random.rand(30, 3)

    for rep, flag in PLOT_FLAG.items():
        if not flag:
            continue

        visualize(rep, rep2dicts[rep][0], rep2dicts[rep][1], colors)
