from math import sqrt

import datetime
import random
import sys


import numpy as np

from tsne_pos.tsne import trainTSNEs

params = sys.argv[1:]
inFile = params[0]
outFile = params[1]
rep = params[2]
numEmbs = params[3]

# computeEmbeddings()
trainTSNEs(inFile, outFile, rep, numEmbs)
