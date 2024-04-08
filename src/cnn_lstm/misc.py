import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
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

train_dataset = Subset(train_dataset, range(1000))
val_dataset = Subset(val_dataset, range(100))

def collate_fn(batch):
    images, captions, img_ids = zip(*batch)
    
    # Stack images into a single tensor
    imgs = torch.cat([img.unsqueeze(0) for img in images], dim=0)

    targets = pad_sequence(captions, batch_first=True, padding_value=0)
    img_ids_tensor = torch.tensor(img_ids, dtype=torch.int64) if not isinstance(img_ids[0], torch.Tensor) else torch.stack(img_ids, 0)

    return imgs, targets, img_ids_tensor


train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

print("Train dataset size: ", len(train_dataset))
print("Validation dataset size: ", len(val_dataset))

vocab_size = len(train_dataset.dataset.vocab)
device = "cuda"

model = EncoderDecoder(embed_size=200,hidden_size=512,vocab_size=vocab_size,num_layers=2,drop_prob=0.3).to(device)
model.load_state_dict(torch.load(f"{CNNLSTM_SAVE_PATH}_glove/best_model.pth"))

batch = next(iter(val_dataloader))
print(batch)

with torch.no_grad():
    image, caption, img_id = batch
    image, caption = image.to(device), caption.to(device)
    print(image.size(), caption.size(), img_id)
    features = model.encoder(image.to(device))
    generated_caption = model.decoder.generate_caption(features.unsqueeze(0),vocab=train_dataset.dataset.vocab)

    print('features:', features)
    print('features size:', features.size())
    print('generated_caption:', generated_caption)
