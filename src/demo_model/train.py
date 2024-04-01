import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from .dataset import DemoDataset   ## This is a local import from dataset.pyA
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from torch.optim import AdamW  # Use this instead of transformers.AdamW
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")


# from transformers import BlipProcessor, BlipForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load the VisionEncoderDecoderModel
encoder_decoder = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", "bert-base-uncased"
)

# Make sure to set the decoder_start_token_id
encoder_decoder.config.decoder_start_token_id = tokenizer.cls_token_id
encoder_decoder.config.pad_token_id = tokenizer.pad_token_id


CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

create_directory(DEMO_SAVE_PATH)
create_directory(DEMO_SAVE_PATH + "/examples")

train_dataset = DemoDataset(
    annotation_file=TRAIN_ANNOTATION_FILE,
    image_folder=TRAIN_IMAGE_FOLDER,
    feature_extractor=feature_extractor,  # For images
    tokenizer=tokenizer,                  # For text
    transforms=None
)
val_dataset = DemoDataset(
    annotation_file=VAL_ANNOTATION_FILE,
    image_folder=VAL_IMAGE_FOLDER,
    feature_extractor=feature_extractor,  # For images
    tokenizer=tokenizer,                  # For text
    transforms=None
)

train_dataset = Subset(train_dataset, range(100))
val_dataset = Subset(val_dataset, range(10))
#train_dataset = Subset(train_dataset, range(len(train_dataset)))
#val_dataset = Subset(val_dataset, range(len(val_dataset)))

print("SANITY CHECK!!")
print(f"LEN TRAIN IMAGE IDS: {len(train_dataset.dataset.image_ids)}")
print(f"LEN VAL IMAGE IDS: {len(val_dataset.dataset.image_ids)}")
print("SANITY CHECK DONE!!")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32)

optimizer = AdamW(encoder_decoder.parameters(), lr=5e-5)


encoder_decoder.to("cuda" if torch.cuda.is_available() else "cpu")
encoder_decoder.train()
# Logger
logger = Logger(f"{DEMO_SAVE_PATH}/logs.log")
# Move the model to the chosen device
encoder_decoder.to(device)

method = "CIDEr"  # method used for comparsions
if torch.cuda.device_count() > 1:
    encoder_decoder = torch.nn.DataParallel(encoder_decoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")



def train(logger, train_dataloader, model, optimizer, device, feature_extractor, tokenizer):
    model.train()
    progress_bar = tqdm(train_dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
        )

        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        loss.backward()

        optimizer.step()

        # Update progress bar with loss info
        progress_bar.set_postfix({"loss": loss.item()})

    return loss.item()
subset_val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=10) 

def evaluate(
    logger, epoch, save_path, best_score, val_dataloader, model, processor, device
):
    model.eval()
    caption_val = []
    plot_captions_dict = {}
    for idx, batch in enumerate(val_dataloader):
        image_ids = batch.pop("image_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        with torch.no_grad():
            outputs = model.generate(pixel_values=pixel_values, max_length=50)

        # Decode the generated ids to text
        generated_captions = processor.batch_decode(outputs, skip_special_tokens=True)

        # Store the generated captions
        for img_id, caption in zip(image_ids, generated_captions):
            caption_val.append(
                {"image_id": img_id.item(), "caption": caption}
            )  # Used for VizWizEvalCap
            plot_captions_dict[img_id.item()] = caption  # Used for plotting

    # Save the generated captions to a json file
    with open(f"{save_path}/generated_captions.json", "w") as f:
        json.dump(caption_val, f, indent=4)

    vizwizRes = val_dataset.dataset.vizwiz.loadRes(
        f"{save_path}/generated_captions.json"
    )
    vizwizEval = VizWizEvalCap(val_dataset.dataset.vizwiz, vizwizRes)
    vizwizEval.evaluate()

    logger.info(f"Validation scores at epoch: {epoch}")
    for method in vizwizEval.eval:
        logger.info(f"  Method: {method}, Score: {vizwizEval.eval[method]:.4f}")

    return vizwizEval, vizwizRes, plot_captions_dict

def get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method="CIDEr"):
    # Get 5 best and 5 worst captions every epoch
    # Use first 3 idxs to plot throughout the training

    img_id_scores = vizwizEval.imgToEval

    best_img_ids = sorted(
        img_id_scores, key=lambda x: img_id_scores[x][method], reverse=True
    )[:5]
    worst_img_ids = sorted(img_id_scores, key=lambda x: img_id_scores[x][method])[:5]
    first_3_img_ids = list(img_id_scores.keys())[:3]

    best_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in best_img_ids
    ]
    worst_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in worst_img_ids
    ]
    first_3_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in first_3_img_ids
    ]
    best_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(best_img_paths, best_img_ids)
    ]  # get img path, generated caption and ground truth caption
    worst_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(worst_img_paths, worst_img_ids)
    ]
    first_3_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(first_3_img_paths, first_3_img_ids)
    ]

    # Save the images and captions
    save_image_captions(
        best_img_and_captions, f"{DEMO_SAVE_PATH}/examples/epoch_{epoch}/best/"
    )
    save_image_captions(
        worst_img_and_captions, f"{DEMO_SAVE_PATH}/examples/epoch_{epoch}/worst/"
    )
    save_image_captions(
        first_3_img_and_captions, f"{DEMO_SAVE_PATH}/examples/epoch_{epoch}/first_3/"
    )

for epoch in range(3):  # Example: 3 epochs, adjust as necessary
    logger.info(f"Epoch: {epoch+1}")
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(encoder_decoder.device)
        labels = batch["input_ids"].to(encoder_decoder.device) # Rename input_ids to labels

        # The VisionEncoderDecoderModel expects `labels` for the text part during training
        # Forward pass
        encoder_decoder.train()
        outputs = encoder_decoder(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    logger.info(f"Average training loss: {avg_loss}")

    # Save the model every epoch
    encoder_decoder.save_pretrained(f"{DEMO_SAVE_PATH}/model_epoch_{epoch+1}")

    if (epoch + 1) % 3 == 0:  # Adjust as necessary
        vizwizEval, vizwizRes, plot_captions_dict = evaluate(
            logger, epoch, DEMO_SAVE_PATH, best_score, val_dataloader, encoder_decoder, tokenizer,
        )
        score = vizwizEval.eval[method]
        if score > best_score:
            best_score = score
            encoder_decoder.save_pretrained(f"{DEMO_SAVE_PATH}/best_model")
            logger.info(f"New best score: {best_score}. Model saved")

        get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method)

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Move your model to the chosen device
encoder_decoder.to(device)