import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import DemoDataset   ## This is a local import from dataset.pyA
from tqdm import tqdm
from transformers import AutoProcessor, CLIPProcessor
from transformers import AutoModelForCausalLM, CLIPModel
import os
import json
from utils import save_ckp, load_ckp, check_parent_path_exist
from transformers import GPT2LMHeadModel, GPT2Tokenizer


CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")
LOAD_MODEL = False

create_directory(DEMO_SAVE_PATH)
create_directory(DEMO_SAVE_PATH + "/subset_examples")

# 1. choose pretrained processor (for image processing)
# pretrained_processor_name = "microsoft/git-base-coco"
pretrained_processor_name = "microsoft/git-large"
# pretrained_processor_name = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(pretrained_processor_name, cache_dir=CACHE_DIR)
# processor = CLIPProcessor.from_pretrained(pretrained_processor_name, cache_dir=CACHE_DIR)

# 2. choose pretrained model (for text handling)
# pretrained_model_name = "microsoft/git-base-coco"
pretrained_model_name = "microsoft/git-large"
# pretrained_model_name = "openai/clip-vit-base-patch32"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, cache_dir=CACHE_DIR)
# model = CLIPModel.from_pretrained(pretrained_model_name, cache_dir=CACHE_DIR)
# 3. choose optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
optimizer = torch.optim.AdamW(model.parameters(), 1e-5)

# Load Train and Val Dataset
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
train_dataset = Subset(train_dataset, range(6*10*6))
val_dataset = Subset(val_dataset, range(6*2*6))

### Since, subset is used above, the dataset object needs to be called with a .dataset, to access the original dataset. So while using the full dataset, the below is done. ###
# train_dataset = Subset(train_dataset, range(len(train_dataset)))
# val_dataset = Subset(val_dataset, range(len(val_dataset)))

print("SANITY CHECK!!")
print(f"LEN TRAIN IMAGE IDS: {len(train_dataset.dataset.image_ids)}")
print(f"LEN VAL IMAGE IDS: {len(val_dataset.dataset.image_ids)}")
print("SANITY CHECK DONE!!")


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=6)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=18)

# Wrap the model with DataParallel only if more than one GPU is available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load GPT-2 model and tokenizer
gpt2_model_name = 'gpt2'
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

method = "CIDEr"  # method used for comparsions

logger = Logger(f"{DEMO_SAVE_PATH}/subset_logs.log")

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Compute contrastive loss between image and text embeddings using cosine similarity.

    Args:
    - image_embeddings (torch.Tensor): Embeddings for images, shape: (batch_size, embedding_dim)
    - text_embeddings (torch.Tensor): Embeddings for texts, shape: (batch_size, embedding_dim)
    - temperature (float): Temperature for scaling cosine similarity

    Returns:
    - torch.Tensor: Scalar loss value.
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Compute cosine similarity
    # Cosine similarity matrix [batch_size, batch_size]
    # Each element [i, j] represents the similarity between i-th image and j-th text
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T) / temperature

    # Labels for matched pairs
    labels = torch.arange(image_embeddings.size(0)).long().to(image_embeddings.device)

    # Cross-entropy loss for images as anchors
    loss_i = F.cross_entropy(similarity_matrix, labels)

    # Cross-entropy loss for texts as anchors
    loss_t = F.cross_entropy(similarity_matrix.T, labels)

    # Average the two losses
    loss = (loss_i + loss_t) / 2.0

    return loss

def train_clip(logger, train_dataloader, model, optimizer, device, processor, save_path):
    model.train()
    loss = 0
    
    for step, batch in progress_bar:
        # Access each element by its key
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        # Optionally use image_ids for logging or tracking

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # Assuming outputs contain the embeddings you need to calculate your loss
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds
        
        # Calculate your loss based on the embeddings
        loss = contrastive_loss(image_embeddings, text_embeddings)

        loss.backward()
        optimizer.step()

        loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return loss

def evaluate_clip(logger, epoch, save_path, best_score, val_dataloader, clip_model, gpt2_model, gpt2_tokenizer, device):
    clip_model.eval()
    gpt2_model.eval()

    caption_val = []
    plot_captions_dict = {}

    for batch in tqdm(
        enumerate(val_dataloader), total=len(val_dataloader), desc="Validation"
    ):
        image_ids = batch["image_ids"]
        pixel_values = batch["pixel_values"].to(device)

        # Assuming you have a strategy to create prompts for GPT-2 based on the image content
        prompts = ["Describe this image:"] * len(pixel_values)  # Placeholder prompts

        generated_captions = []
        for prompt in prompts:
            encoded_inputs = gpt2_tokenizer.encode_plus(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=50  # Ensure you specify a max_length
            )
            input_ids = encoded_inputs["input_ids"].to(device)
            attention_mask = encoded_inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = gpt2_model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    max_length=50, 
                    num_return_sequences=1
                )
            caption = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_captions.append(caption)

        for img_id, caption in zip(image_ids, generated_captions):
            caption_val.append({"image_id": img_id.item(), "caption": caption})  # Convert tensor to int with .item()
            plot_captions_dict[img_id] = caption

    # Save the generated captions to a json file
    with open(f"{save_path}/subset_generated_captions.json", "w") as f:
        json.dump(caption_val, f, indent=4)

    vizwizRes = val_dataset.dataset.vizwiz.loadRes(
        f"{save_path}/subset_generated_captions.json"
    )
    vizwizEval = VizWizEvalCap(val_dataset.dataset.vizwiz, vizwizRes)
    vizwizEval.evaluate()

    logger.info(f"Validation scores at epoch: {epoch}")
    for method in vizwizEval.eval:
        logger.info(f"  Method: {method}, Score: {vizwizEval.eval[method]:.4f}")
    
    score = vizwizEval.eval[method]
    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch + 1,
        'valid_score_max': score,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    checkpoint_path = f"{save_path}/subset_checkpointing/checkpoint/current_checkpoint.pt"
    best_model_path = f"{save_path}/subset_checkpointing/best_model/best_model.pt"
    # save checkpoint
    check_parent_path_exist(checkpoint_path)
    check_parent_path_exist(best_model_path)
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    
    if score >= best_score:
        print('Valid score increases ({:.6f} --> {:.6f}).  Saving model ...'.format(best_score, score))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        best_score = score

    return vizwizEval, vizwizRes, plot_captions_dict


def train(loger, train_dataloader, model, optimizer, device, processor, save_path):
    model.train()
    # loss_min = 1e5
    for idx, batch in progress_bar:
        
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            pixel_values=pixel_values, 
            labels=input_ids
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
    for idx, batch in tqdm(
        enumerate(val_dataloader), total=len(val_dataloader), desc="Validation"
    ):
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
    with open(f"{save_path}/subset_generated_captions.json", "w") as f:
        json.dump(caption_val, f, indent=4)

    vizwizRes = val_dataset.dataset.vizwiz.loadRes(
        f"{save_path}/subset_generated_captions.json"
    )
    vizwizEval = VizWizEvalCap(val_dataset.dataset.vizwiz, vizwizRes)
    vizwizEval.evaluate()

    logger.info(f"Validation scores at epoch: {epoch}")
    for method in vizwizEval.eval:
        logger.info(f"  Method: {method}, Score: {vizwizEval.eval[method]:.4f}")
    
    score = vizwizEval.eval[method]
    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch + 1,
        'valid_score_max': score,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    checkpoint_path = f"{save_path}/subset_checkpointing/checkpoint/current_checkpoint.pt"
    best_model_path = f"{save_path}/subset_checkpointing/best_model/best_model.pt"
    # save checkpoint
    check_parent_path_exist(checkpoint_path)
    check_parent_path_exist(best_model_path)
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    
    if score >= best_score:
        print('Valid score increases ({:.6f} --> {:.6f}).  Saving model ...'.format(best_score, score))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        best_score = score

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
        best_img_and_captions, f"{DEMO_SAVE_PATH}/subset_examples/epoch_{epoch}/best/"
    )
    save_image_captions(
        worst_img_and_captions, f"{DEMO_SAVE_PATH}/subset_examples/epoch_{epoch}/worst/"
    )
    save_image_captions(
        first_3_img_and_captions, f"{DEMO_SAVE_PATH}/subset_examples/epoch_{epoch}/first_3/"
    )


best_score = 0
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    # Wrap the dataloader with tqdm for a progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc="Training"
    )

    # Train the model
    loss = train(logger, train_dataloader, model, optimizer, device, processor, DEMO_SAVE_PATH)
    logger.info(f"Loss at epoch {epoch}: {loss}")

    # Evaluate the model every epoch
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
    # vizwizEval, vizwizRes, plot_captions_dict = evaluate_clip(logger, epoch, DEMO_SAVE_PATH, best_score, val_dataloader, model, gpt2_model, gpt2_tokenizer, device)
    
    score = vizwizEval.eval[method]
    logger.info(f"Valid Score at epoch {epoch}: {score}")

    if score > best_score:
        best_score = score
        model.save_pretrained(f"{DEMO_SAVE_PATH}/subset_best_model")
        logger.info(f"New best score: {best_score}. Model saved")

    get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method)

