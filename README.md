# POS Embeddings Plotter

Link para arquivos prontos:
https://drive.google.com/drive/folders/1jEFnWcQGGss2gYocmNIY7XzYx5d9xJ3P

## Scripts
The scripts must be used in a specific order, since the output of one might be the input of another. Their order and decriptions will be given below.

```
1 - extractInfos.py: will extract content from the POS model and the datasets
2 - trainTSNE.py: will train the TSNE of the embeddings extracted at (1)
3 - createInfoFile.py: will create a .csv from the output of (1) and (2)
4 - plotter.py: will plot the trained TSNEs using outputs from (1) and (3)

* - createProloFile.py: aditional script for creating a .csv for visualization using outputs from (1) and (3)
```

### extractInfos.py
```
External packages needed and how to install/update it using pip:
torch (pytorch) - pip install -U torch
posembd - pip install git+https://github.com/pauloamed/posembd.git
```

You will need the following files/dirs
```
data/ - a directory with POS datasets (following the posembd structure)
modelPath - a path to the posembd model .pt file
```

To use it, execute
```
python extractInfos.py VOCAB_PATH INFOS_PICKLE_PATH TAGS_PATH EMBEDDINGS_DIR

Where
VOCAB_PATH: path where vocab csv file will be saved
INFOS_PICKLE_PATH: path to pickle file where the info object will be saved
TAGS_PATH: path where tags csv file will be saved
EMBEDDINGS_DIR: directory where the pickle files from the embeddings will be saved
```


### trainTSNE.py
```
External packages needed and how to install/update it using pip:
numpy - pip install -U numpy
sklearn - pip install -U scikit-learn
```

You will need the following files/dirs
```
INPUT_DIR/ - the directory where the embeddings extracted from the datasets using the posembd model at step (1) are saved as pickle files
```

To use it, execute
```
python trainTSNE.py INPUT_DIR REPRESENTATION_LVL NUMBER_OF_EMBEDDINGS OUTPUT_DIR

Where
INPUT_DIR: path to directory containing pickle embeddings computed at computeEmbeddings.py
REPRESENTATION_LVL: 0, 1, 2 or 3, indicating the level of embedding to be trained
NUMBER_OF_EMBEDDINGS: number of embeddings that will be trained at TSNEs algorithm. -1 to train all embeddings
OUTPUT_DIR: path to directory that will contain the trained TSNEs
```

### createInfoFile.py
```
External packages needed and how to install/update it using pip:
tqdm - pip install -U tqdm
```

You will need the following files/dirs
```
TSNES_DIR/ - the directory where the trained tsnes from step (2) are saved as pickle files
```

To use it, execute
```
python createInfoFile.py INFOS_PICKLE_PATH TSNES_DIR OUTPUT_PATH

Where
INFOS_PICKLE_PATH: path to pickle file where the info object is saved
TSNES_DIR: directory where the trained tsnes pickles are saved
OUTPUT_PATH: path where info csv file will be saved
```

### plotter.py
```
External packages needed and how to install/update it using pip:
tqdm - pip install -U tqdm
matplotlib - pip install -U matplotlib
numpy - pip install -U numpy
```

You will need the following files/dirs
```
INFOS_PATH: path to the infos csv file created at (3)
VOCAB_PATH: path to the infos csv file created at (1)
TAGS_PATH: path to the infos csv file created at (1)
```

To use it, execute
```
python plotter.py INFOS_PATH VOCAB_PATH TAGS_PATH

Where
INFOS_PATH: path where infos csv file is saved
VOCAB_PATH: path where vocab csv file is saved
TAGS_PATH: path where tags csv file is saved
```
### createProloFile.py
```
External packages needed and how to install/update it using pip:
tqdm - pip install -U tqdm
```

You will need the following files/dirs
```
INFOS_PATH: path to the infos csv file created at (3)
VOCAB_PATH: path to the infos csv file created at (1)
TAGS_PATH: path to the infos csv file created at (1)
```

To use it, execute
```
python createProloFile.py INFOS_PATH VOCAB_PATH TAGS_PATH OUTPUT_PATH

Where
INFOS_PATH: path to info csv file created at createInfoFile.py
VOCAB_PATH: path to vocab csv file created at computeEmbeddings.py
TAGS_PATH: path to tags csv file created at computeEmbeddings.py
OUTPUT_PATH: path where prolo csv file will be saved
```
