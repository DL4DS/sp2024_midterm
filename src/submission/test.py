import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import DemoDataset
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

create_directory(DEMO_SAVE_PATH)  # src/base/helpers.py
create_directory(DEMO_SAVE_PATH + "/examples")

# The path below points to the location where the model was saved
MODEL_PATH = f"{DEMO_SAVE_PATH}/best_model"

# Load your fine tuned model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)

## TODO
# You can use the AutoProcessor.from_pretrained() method to load the HuggingFace
# processor for the model you are using. This will allow you to use the processor
# to encode and decode text and images.
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor
#
# Of course you should use the same model you trained with.
try:
    processor = AutoProcessor.from_pretrained("microsoft/git-base", cache_dir=CACHE_DIR)
except Exception as e:
    print("You need to pick a pre-trained model from HuggingFace.")
    print("Exception: ", e)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_dataset = DemoDataset(
    processor=processor,
    annotation_file=TEST_ANNOTATION_FILE,
    image_folder=TEST_IMAGE_FOLDER,
    transforms=None,
    test=True,
)

print(len(test_dataset))

test_captions_dict = {}
caption_val = []
for data in tqdm(test_dataset, total=len(test_dataset)):
    pixel_values = data["pixel_values"]
    pixel_values = pixel_values.unsqueeze(0)
    img_id = data["image_ids"]

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        if torch.cuda.device_count() > 1:
            output = model.module.generate(pixel_values=pixel_values, max_length=50)
        else:
            output = model.generate(pixel_values=pixel_values, max_length=50)

    caption = processor.decode(output[0], skip_special_tokens=True)

    test_captions_dict[img_id.item()] = caption
    caption_val.append(
        {"image_id": img_id.item(), "caption": caption}
    )  # Used for VizWizEvalCap

with open(DEMO_SAVE_PATH + "/test_captions.json", "w") as f:
    json.dump(caption_val, f, indent=4)

print("Test captions saved to disk!!")
