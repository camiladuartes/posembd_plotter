from math import sqrt

import datetime
import random
import sys


import numpy as np

from tsne_pos.tsne import trainTSNEs, loadTSNEs
from tsne_pos import computeEmbeddings
from tsne_pos.visualize import plot


computeEmbeddings()
trainTSNEs()


# plot(rep2dicts)
# printToFile(rep2dicts['embeddings1'][1])
