import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from src.demo_model.dataset import DemoDataset
from tqdm import tqdm
from transformers import ViTFeatureExtractor, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

# Create necessary directories
create_directory(DEMO_SAVE_PATH)
create_directory(DEMO_SAVE_PATH + "/examples")

# Load ViT for image feature extraction
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# Load GPT-2 for caption generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Test dataset
test_dataset = DemoDataset(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    annotation_file=TEST_ANNOTATION_FILE,
    image_folder=TEST_IMAGE_FOLDER,
    transforms=None,  # Add transforms if required
)

# Caption generation
test_captions_dict = {}
caption_val = []
for data in tqdm(test_dataset, total=len(test_dataset)):
    images = data["images"]
    img_id = data["image_ids"]

    inputs = feature_extractor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    test_captions_dict[img_id] = caption
    caption_val.append({"image_id": img_id, "caption": caption})

# Save captions
with open(DEMO_SAVE_PATH + "/test_captions.json", "w") as f:
    json.dump(caption_val, f, indent=4)

print("Test captions saved to disk!")
