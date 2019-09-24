import matplotlib.pyplot as plt

def visualize(title, embeddings, colors, labels):
    x, y = zip(*embeddings)
    plt.scatter(x, y, c=colors, alpha=0.5, label=labels)
    plt.title(title)
    plt.show()

