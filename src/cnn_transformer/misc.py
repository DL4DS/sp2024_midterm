import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import CNNLSTMDataset  # import from local file dataset.py
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from model import EncoderDecoder

image_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CNNLSTMDataset(
    processor=None,
    annotation_file=TRAIN_ANNOTATION_FILE,
    image_folder=TRAIN_IMAGE_FOLDER,
    transforms=image_transforms,
)
val_dataset = CNNLSTMDataset(
    processor=None,
    annotation_file=VAL_ANNOTATION_FILE,
    image_folder=VAL_IMAGE_FOLDER,
    transforms=image_transforms,
)

print("Train dataset size: ", len(train_dataset))
print("Validation dataset size: ", len(val_dataset))
print(train_dataset)