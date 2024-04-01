import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import nltk
import spacy

# set this path to where you want to save results
BASE_DIR = "/projectnb/ds598/students/demoyu/sp2024_midterm/"

# Do not edit. This points to the dataset folder
DATA_BASE_DIR = "/projectnb/ds598/materials/datasets/vizwiz/captions/"

os.environ["SPACY_DATA"] = BASE_DIR + "/misc2/spacy_data"

nltk_data_directory = BASE_DIR + "misc2/nltk_data"
nltk.data.path.append(nltk_data_directory)
nltk.download("punkt", download_dir=nltk_data_directory)

# Set the Transformers cache directory
os.environ["TRANSFORMERS_CACHE"] = BASE_DIR + "misc2"

# Set the Hugging Face home directory (this includes datasets cache)
os.environ["HF_HOME"] = BASE_DIR + "misc2"
os.environ["TORCH_HOME"] = BASE_DIR + "misc2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DATA PATHS
# Data was loaded here from the VizWiz website 
# `wget https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip` then unzip annotations.zip
TRAIN_ANNOTATION_FILE = (
    DATA_BASE_DIR + "annotations/train.json"
)

# Data was loaded here from the VizWiz website
# `wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip`
TRAIN_IMAGE_FOLDER = DATA_BASE_DIR + "train"
VAL_ANNOTATION_FILE = DATA_BASE_DIR + "annotations/val.json"

# Data was loaded here from the VizWiz website
# `wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip`
VAL_IMAGE_FOLDER = DATA_BASE_DIR + "val"

TEST_ANNOTATION_FILE = (
    DATA_BASE_DIR + "annotations/test.json"
)

# Data was loaded here from the VizWiz website
# `wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip`
TEST_IMAGE_FOLDER = DATA_BASE_DIR + "test"


DEMO_MEAN = np.array([123.675, 116.280, 103.530]) / 255
DEMO_STD = np.array([58.395, 57.120, 57.375]) / 255

# SAVE PATHS
DEMO_SAVE_PATH = BASE_DIR + "RESULTS2/git"
CNNLSTM_SAVE_PATH = BASE_DIR + "RESULTS2/cnn_lstm"
