from math import sqrt

import datetime
import random
import sys


import numpy as np

from tsne_pos.tsne import trainTSNEs
from tsne_pos import computeEmbeddings
from tsne_pos.visualize import plot

params = sys.argv[1:]
inFile = params[0]
outFile = params[1]
rep = params[2]

# computeEmbeddings()
trainTSNEs(inFile, outFile, rep)


# plot(rep2dicts)
# printToFile(rep2dicts['embeddings1'][1])
