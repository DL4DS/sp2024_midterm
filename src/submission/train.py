import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import DemoDataset   ## This is a local import from dataset.pyA
from tqdm import tqdm
from transformers import AutoProcessor, AutoImageProcessor
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

################################################################################
# This is template code that will not run as is since a model is not defined but
# is has much of the infrastructure needed to fine-tune a model on the VizWiz
# dataset.
#
# At a minimum you will have to complete code indicated by TODO comments.
################################################################################

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

create_directory(DEMO_SAVE_PATH)
create_directory(DEMO_SAVE_PATH + "/examples")

## TODO
# You can use the AutoProcessor.from_pretrained() method to load the HuggingFace
# processor for the model you are using. This will allow you to use the processor
# to encode and decode text and images.
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor
try:
    processor = AutoProcessor.from_pretrained("microsoft/git-base", cache_dir=CACHE_DIR)
except Exception as e:
    print("You need to pick a pre-trained model from HuggingFace.")
    print("Exception: ", e)

train_dataset = DemoDataset(
    processor=processor,
    annotation_file=TRAIN_ANNOTATION_FILE,
    image_folder=TRAIN_IMAGE_FOLDER,
    transforms=None,
)
val_dataset = DemoDataset(
    processor=processor,
    annotation_file=VAL_ANNOTATION_FILE,
    image_folder=VAL_IMAGE_FOLDER,
    transforms=None,
)

### Use the Subset while debugging ###
# train_dataset = Subset(train_dataset, range(100))
# val_dataset = Subset(val_dataset, range(10))

### Since, subset is used above, the dataset object needs to be called with a .dataset, to access the original dataset. So while using the full dataset, the below is done. ###
train_dataset = Subset(train_dataset, range(len(train_dataset)))
val_dataset = Subset(val_dataset, range(len(val_dataset)))

print("SANITY CHECK!!")
print(f"LEN TRAIN IMAGE IDS: {len(train_dataset.dataset.image_ids)}")
print(f"LEN VAL IMAGE IDS: {len(val_dataset.dataset.image_ids)}")
print("SANITY CHECK DONE!!")


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32)

## TODO
# You can use the AutoModelForCausalLM.from_pretrained() method to load the HuggingFace
# model you want to fine-tune. This will allow you to use the model to train and evaluate
# on the VizWiz dataset.
try:
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base", cache_dir=CACHE_DIR)
except Exception as e:
    print("You need to pick a pre-trained model from HuggingFace.")
    print("Exception: ", e)

## TODO Select your model optimizer
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)   # pick one from torch.optim
except Exception as e:
    print("You need to pick an optimizer from torch.optim.")
    print("Exception: ", e)

# Wrap the model with DataParallel only if more than one GPU is available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

method = "CIDEr"  # method used for comparsions

logger = Logger(f"{DEMO_SAVE_PATH}/logs.log")


def train(loger, train_dataloader, model, optimizer, device, processor):
    model.train()

    for idx, batch in progress_bar:
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask
        )

        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        loss.backward()

        optimizer.step()

        # Update progress bar with loss info
        progress_bar.set_postfix({"loss": loss.item()})

    return loss.item()


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
            if torch.cuda.device_count() > 1:
                outputs = model.module.generate(pixel_values=pixel_values, max_length=50)
            else:
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


best_score = 0
for epoch in range(15):
    print(f"Epoch: {epoch+1}")
    # Wrap the dataloader with tqdm for a progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc="Training"
    )

    # Train the model
    loss = train(logger, train_dataloader, model, optimizer, device, processor)
    logger.info(f"Loss at epoch {epoch}: {loss}")

    # Evaluate the model every 3 epochs
    if epoch % 3 == 0:
        vizwizEval, vizwizRes, plot_captions_dict = evaluate(
            logger,
            epoch,
            DEMO_SAVE_PATH,
            best_score,
            val_dataloader,
            model,
            processor,
            device,
        )
        score = vizwizEval.eval[method]
        if score > best_score:
            best_score = score
            if torch.cuda.device_count() > 1:
                model.module.save_pretrained(f"{DEMO_SAVE_PATH}/best_model")
            else:
                model.save_pretrained(f"{DEMO_SAVE_PATH}/best_model")
            logger.info(f"New best score: {best_score}. Model saved")

        get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method)
