import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import CNNLSTMDataset
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

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

create_directory(CNNLSTM_SAVE_PATH)  # src/base/helpers.py
create_directory(CNNLSTM_SAVE_PATH + "/examples")

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

test_dataset = CNNLSTMDataset(
    processor=None,
    annotation_file=TEST_ANNOTATION_FILE,
    image_folder=TEST_IMAGE_FOLDER,
    transforms=image_transforms,
    training=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MODEL_PATH = '/projectnb/ds598/admin/xthomas/midterm_baselines/RESULTS/cnn_lstm/best_model.pth'
MODEL_PATH = CNNLSTM_SAVE_PATH + "/best_model.pth"

model = EncoderDecoder(embed_size=400, hidden_size=512, vocab_size=len(train_dataset.vocab), num_layers=2, drop_prob=0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH))


print(len(test_dataset))

test_captions_dict = {}
caption_val = []
for data in tqdm(test_dataset, total=len(test_dataset)):

    img = data[0].to(device).unsqueeze(0)
    img_id = data[1]

    with torch.no_grad():
        feature = model.encoder(img)
        caption = model.decoder.generate_caption(feature.unsqueeze(0), max_len=50, vocab=train_dataset.vocab)
    test_captions_dict[img_id.item()] = caption
    caption_val.append(
        {"image_id": img_id.item(), "caption": caption}
    )  # Used for VizWizEvalCap

with open(CNNLSTM_SAVE_PATH + "/test_captions.json", "w") as f:
    json.dump(caption_val, f, indent=4)

print("Test captions saved to disk!!")
